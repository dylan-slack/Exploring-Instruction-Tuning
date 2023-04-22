"""Data functions"""
import json
import re
import glob
import random

import numpy as np
from datasets import Dataset


def clean(answer: str):
    return re.sub(r"<<.*?>>", "", answer).replace("####", "The answer is")


def read_jsonl(path: str):
    with open(path) as fh:
        load = [json.loads(line) for line in fh.readlines() if line]
    for d in load:
        d["answer"] = clean(d["answer"])
    return load


def load_data(train: str = "./grade-school-math/grade_school_math/data/train.jsonl",
              test: str = "grade-school-math/grade_school_math/data/test.jsonl"):
    train_data = Dataset.from_list(read_jsonl(train))
    test_data = Dataset.from_list(read_jsonl(test))
    return train_data, test_data


def load_bbh_tasks(bbh_folder: str = "./Big-Bench-Hard/bbh",
                   sample: int = -1):
    tasks = glob.glob(bbh_folder + "/*.json")
    all_data = []
    for t in tasks:
        with open(t) as json_file:
            data = json.load(json_file)["examples"]
            all_data.extend(data)
    if sample > 0:
        all_data = np.random.choice(all_data, replace=False, size=sample)

    def replace(d):
        d["question"] = d.pop("input")
        d["answer"] = d.pop("target")
        return d

    all_data = list(map(replace, all_data))
    dataset = Dataset.from_list(all_data)
    return dataset


def format_dolly(data: dict):
    assert "instruction" in data.keys() and "context" in data.keys()
    if random.choice([True, False]):
        output = f"{data['context']} {data['instruction']}"
    else:
        output = f"{data['instruction']} {data['context']}"
    return output


def load_dolly_data(dolly_file: str = "./dolly/data/databricks-dolly-15k.jsonl"):
    all_data = []
    with open(dolly_file, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            question = format_dolly(json_line)
            answer = json_line["response"]
            all_data.append(
                {"question": question, "answer": answer}
            )
    return Dataset.from_list(all_data)
