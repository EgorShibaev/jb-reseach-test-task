import json
import re
from dataclasses import dataclass
from fuzzywuzzy import fuzz

import torch
from loguru import logger
from torch.utils.data import Dataset
from tqdm import tqdm


def post_process(code):
    code = code.replace("<NUM_LIT>", "0").replace("<STR_LIT>", "").replace("<CHAR_LIT>", "")
    pattern = re.compile(r"<(STR|NUM|CHAR)_LIT:(.*?)>", re.S)
    lits = re.findall(pattern, code)
    for lit in lits:
        code = code.replace(f"<{lit[0]}_LIT:{lit[1]}>", lit[1])
    return code


def process_special_tokens(code, replacements):
    for k, v in replacements.items():
        code = code.replace(k, v)
    return code


class PythonDataset(Dataset):
    def __init__(self, path_to_dataset: str):
        self.dataset: list[dict] = []
        with open(path_to_dataset, "r") as f:
            content = f.read()
            for line in content.split("\n"):
                if line:
                    self.dataset.append(json.loads(line))

        for i in range(len(self.dataset)):
            self.dataset[i]["signature"] = post_process(self.dataset[i]["signature"])
            self.dataset[i]['signature'] = process_special_tokens(
                self.dataset[i]['signature'],
                {
                    "<EOL>": "\n",
                    "<INDENT>": "",
                    "<DEDENT>": ""
                }
            )

            self.dataset[i]["body"] = post_process(self.dataset[i]["body"])
            self.dataset[i]['body'] = process_special_tokens(
                self.dataset[i]['body'],
                {
                    "<EOL>": " ",
                    "<INDENT>": "",
                    "<DEDENT>": ""
                }
            )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple[str, str]:
        # add \t before each line in the docstring
        docstring = self.dataset[idx]['docstring']
        docstring = "\t" + docstring
        context = f"{self.dataset[idx]['signature']}\n\t\"\"\"\n{docstring}\n\t\"\"\"\n"
        to_complete = self.dataset[idx]['body']
        return context, to_complete


@dataclass
class EvaluationResult:
    exact_match: float
    edit_sim: float


def evaluate(model, tokenizer, path_to_dataset: str, device, verbose) -> EvaluationResult:
    dataset = PythonDataset(path_to_dataset)

    exact_match = []
    edit_sim = []

    for i in tqdm(range(len(dataset)), total=len(dataset), disable=not verbose):
        context, to_complete = dataset[i]
        context_tokens = tokenizer(context, return_tensors="pt")
        generated = model.generate(input_ids=context_tokens['input_ids'].to(device),
                                   max_length=context_tokens['input_ids'].shape[1] + 100,
                                   eos_token_id=tokenizer.eos_token_id,
                                   attention_mask=torch.ones_like(context_tokens['input_ids']).to(device),
                                   pad_token_id=0,
                                   early_stopping=True,
                                   do_sample=True,
                                   top_k=50)

        # keep only generated part
        generated = generated[0, context_tokens['input_ids'].shape[1]:]

        generated_line = tokenizer.decode(generated, skip_special_tokens=True)

        # remove all new lines and indentations
        generated_line = generated_line.replace("\n", " ")
        generated_line = generated_line.replace("\t", "")
        generated_line = generated_line.replace(" " * 4, "")

        # if new method is generated, keep only needed one

        for starting_of_another_function in ['def', 'class', '@']:
            if starting_of_another_function in generated_line:
                index = generated_line.index(starting_of_another_function)
                generated_line = generated_line[:index]

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
