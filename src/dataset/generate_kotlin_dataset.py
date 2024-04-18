import argparse
import os
import json
from pathlib import Path
import re
from typing import Dict, Iterable

from sklearn.model_selection import train_test_split


def save_train_val_dataset(dataset, path):
    """
    Save the train or validation dataset to a file.
    They are saved in the format: [{'code': ...}, {'code': ...}, ...].
    Such format is expected by the training script.
    :param dataset: list of codes
    :param path: path to save the dataset
    :return: None
    """
    dataset = [{"code": code} for code in dataset]
    with open(path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item, indent=None))
            f.write("\n")


def split_code_into_blocks(code, block_len=1000):
    """
    Split the code into blocks of the given length
    :param code: code to split
    :param block_len: length of the block
    :return: list of code blocks
    """
    code_blocks = []
    for i in range(0, len(code), block_len):
        code_blocks.append(code[i:i + block_len])
    return code_blocks

def save_test_dataset(dataset, path):
    """
    Save the test dataset to a file.
    They are saved in the format: [{'context': ..., 'line_to_complete': ...}, ...].
    Such format is expected by the evaluation script.
    :param dataset: list of test cases
    :param path: path to save the dataset
    :return: None
    """
    with open(path, "w") as f:
        for item in dataset:
            f.write(json.dumps(item, indent=None))
            f.write("\n")


def extract_code_block(code, start_position):
    """
    Extracts the code block starting at the given position according to the opening and closing braces
    :param code: Java or Kotlin code
    :param start_position: position of the opening brace of the code block
    :return: code block and end position of the code block
    """
    open_braces = 0
    code_block_start = start_position
    code_block_end = -1

    for i in range(start_position, len(code)):
        if code[i] == "{":
            open_braces += 1
        elif code[i] == "}":
            open_braces -= 1
            if open_braces == 0:
                code_block_end = i + 1
                break

    if code_block_end == -1:
        raise ValueError(f"Code block extraction failed")

    while code[code_block_start] == "\n":
        code_block_start += 1

    return code[code_block_start:code_block_end], code_block_end


def generate_test_case(code: str, max_context_len: int = 1000) -> Iterable[Dict[str, str]]:
    """
    Generate a test case for the given code.
    the task is to write definition of some function
    :param max_context_len: maximum length of the context
    :param code: code to generate test case for
    :return: a tuple of (context, line to complete)
    """

    # find all function definitions
    method_pattern = re.compile(rf"\n\s*(\w+\s+)*fun\s+(\w+)\s*\([^\)]*\)[^\n]*\{{")

    function_defs = [(m.start(), m.group(0)) for m in method_pattern.finditer(code)]

    if not function_defs:
        return None

    function_defs = [
        (start, extract_code_block(code, start)[0])
        for start, _ in function_defs

    ]

    for start, function_def in function_defs:

        context = code[:start]
        signature = function_def[:function_def.index("{")].strip()
        body = function_def[function_def.index("{") + 1:]

        context += f"\n{signature} {{\n"

        if len(context) > max_context_len or len(body) > max_context_len:
            continue

        if body.strip() == "}":
            continue

        yield {
            "context": context,
            "line_to_complete": body
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_path", type=Path)
    parser.add_argument("--dataset_path", type=Path)

    args = parser.parse_args()

    args.dataset_path.mkdir(parents=True, exist_ok=True)

    # do over all .kt files in the repo

    kotlin_files = []
    for root, dirs, files in os.walk(args.repo_path):
        for file in files:
            if file.endswith(".kt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    code = f.read()
                    # delete /* ... */ comment in the beginning of the file
                    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
                    kotlin_files.append(code)

    train, test = train_test_split(kotlin_files, test_size=0.1, random_state=42)
    train, val = train_test_split(train, test_size=0.1, random_state=42)

    train = [code_block for code in train for code_block in split_code_into_blocks(code)]
    val = [code_block for code in val for code_block in split_code_into_blocks(code)]

    save_train_val_dataset(train, os.path.join(args.dataset_path, "train.jsonl"))
    save_train_val_dataset(val, os.path.join(args.dataset_path, "val.jsonl"))

    test_cases = []
    for code in test:
        test_cases.extend(generate_test_case(code))

    save_test_dataset(test_cases, os.path.join(args.dataset_path, "test.jsonl"))

    print(f"Train size: {len(train)}")
    print(f"Validation size: {len(val)}")
    print(f"Test size: {len(test_cases)}")


if __name__ == "__main__":
    main()
