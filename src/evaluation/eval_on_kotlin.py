import json
import re
from dataclasses import dataclass
from fuzzywuzzy import fuzz

import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm


class KotlinDataset(Dataset):
    def __init__(self, path_to_dataset: str):
        self.dataset: list[dict] = []
        with open(path_to_dataset, "r") as f:
            content = f.read()
            for line in content.split("\n"):
                if line:
                    self.dataset.append(json.loads(line))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[str, str]:
        context, to_complete = self.dataset[idx]["context"], self.dataset[idx]["line_to_complete"]

        return context, to_complete

@dataclass
class EvaluationResult:
    exact_match: float
    edit_sim: float


def trim_generated(generated: str) -> str:
    """
    Trim the generated string, so, that only the fuction implementation is left
    :param generated: generated string
    :return: trimmed string
    """
    brackets_balance = 0
    for i, c in enumerate(generated):
        if c == '{':
            brackets_balance += 1
        elif c == '}':
            brackets_balance -= 1

        if brackets_balance == -1:
            return generated[:i + 1]

    return generated

def evaluate(model, tokenizer, path_to_dataset: str, device, verbose: bool) -> EvaluationResult:
    dataset = KotlinDataset(path_to_dataset)

    exact_match = []
    edit_sim = []

    for i in tqdm(range(len(dataset)), total=len(dataset), disable=not verbose):
        context, to_complete = dataset[i]
        context_tokens = tokenizer(context, return_tensors="pt")
        generated = model.generate(input_ids=context_tokens['input_ids'].to(device),
                                   max_length=context_tokens['input_ids'].shape[1] + 100,
                                   eos_token_id=tokenizer.eos_token_id,
                                   attention_mask=torch.ones_like(context_tokens['input_ids']).to(device),
                                   num_beams=3,
                                   pad_token_id=0,
                                   early_stopping=True)

        # keep only generated part
        generated = generated[0, context_tokens['input_ids'].shape[1]:]

        generated_line = tokenizer.decode(generated, skip_special_tokens=True)

        generated_line = trim_generated(generated_line)

        generated_line = generated_line.strip()
        to_complete = to_complete.strip()

        if verbose:
            logger.debug(f"Context: \n<{context}>\n")
            logger.debug(f"Generated: <{generated_line}>")
            logger.debug(f"Expected: <{to_complete}>")

        # computing metrics
        exact_match.append(generated_line == to_complete)
        edit_sim.append(fuzz.ratio(generated_line, to_complete) / 100)

    return EvaluationResult(exact_match=sum(exact_match) / len(exact_match), edit_sim=sum(edit_sim) / len(edit_sim))
