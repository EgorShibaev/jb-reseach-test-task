import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from loguru import logger
import argparse

import torch
import transformers

from eval_on_codeXGLUE import evaluate as evaluate_codeXGLUE
from eval_on_kotlin import evaluate as evaluate_kotlin


def eval_and_save_metrics(model, tokenizer, path_to_dataset, device, output_path, eval_fn, verbose=True):
    logger.info(f"Evaluating on {path_to_dataset}")

    evaluation_result = eval_fn(model, tokenizer, path_to_dataset, device, verbose=verbose)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("Evaluation result:")
    print(evaluation_result)
    print(f"Saving evaluation result to {output_path}")
    with open(output_path, "w") as f:
        json.dump(asdict(evaluation_result), f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=str, default="all", choices=["all", "python", "kotlin"])
    args = parser.parse_args()

    model_name = "microsoft/phi-1_5"

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda"
    model = model.to(device)

    if args.scope == "all" or args.scope == "python":
        eval_and_save_metrics(
            model, tokenizer, "./data/test.jsonl", device,
            Path(f"output/{datetime.now().isoformat()}/evaluation_result_python.json"),
            evaluate_codeXGLUE
        )

    if args.scope == "all" or args.scope == "kotlin":
        eval_and_save_metrics(
            model, tokenizer, "./data/kotlin/test.jsonl", device,
            Path(f"output/{datetime.now().isoformat()}/evaluation_result_kotlin.json"),
            evaluate_kotlin
        )


if __name__ == "__main__":
    main()
