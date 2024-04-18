import os

import torch.cuda
import transformers
from loguru import logger
from torch.utils.data import Dataset
import json
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from peft import LoraConfig, get_peft_model

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

accelerator = Accelerator(fsdp_plugin=fsdp_plugin)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class FineTuningDataset(Dataset):
    def __init__(self, path_to_dataset: str):
        self.dataset: list[dict] = []
        with open(path_to_dataset, "r") as f:
            content = f.read()
            for line in content.split("\n"):
                if line:
                    self.dataset.append(json.loads(line))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> str:
        return self.dataset[idx]['code']


def tokenize(code: list[str], tokenizer, max_length=1024):
    result = tokenizer(
        code,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        add_special_tokens=True
    )
    result["labels"] = result["input_ids"].clone()
    return result


def collate_fn(batch, tokenizer):
    code = [item for item in batch]
    return tokenize(code, tokenizer)


def main():
    model_name = "microsoft/phi-1_5"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda"
    model = model.to(device)

    train_dataset = FineTuningDataset("./data/kotlin/train.jsonl")
    val_dataset = FineTuningDataset("./data/kotlin/val.jsonl")

    batch_size = 1
    logger.info(f"Training with batch size {batch_size}")
    logger.info(f"Training with {torch.cuda.device_count()} GPUs")
    logger.info(f"Training for 1 epoch: {len(train_dataset) // batch_size} steps")

    model = accelerator.prepare_model(model)

    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "dense",
            "fc1",
            "fc2",
            "lm_head"
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    print_trainable_parameters(model)

    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    os.environ["WANDB_PROJECT"] = "test task jb"
    os.environ["WANDB_LOG_MODEL"] = "false"

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda data: collate_fn(data, tokenizer),
        args=transformers.TrainingArguments(
            output_dir="./output/fine_tuned_model",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            evaluation_strategy="steps",
            num_train_epochs=1,
            logging_dir="./output/fine_tuned_model/logs",
            save_steps=1000 // batch_size,
            eval_steps=1000 // batch_size,
            save_total_limit=2,
            report_to="wandb",
            run_name="fine_tuning_phil-1_5_on_kotlin",
            logging_steps=10,
            learning_rate=2.5e-5,
        )
    )

    trainer.train()


if __name__ == "__main__":
    main()
