"""Main"""
import argparse
from functools import partial
from typing import Optional, List, Dict

import datasets
import numpy as np
import transformers
from datasets import Dataset

import wandb

import data


def accuracy(preds, tokenizer, acc_metric):
    predictions, labels = preds
    predictions = np.argmax(predictions[0], axis=-1)
    labels[labels == -100] = 0
    predictions[predictions == -100] = 0
    pred_labels = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    gt_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    return acc_metric.compute(predictions=pred_labels, references=gt_labels)


def train(train_data, test_data, args):
    # Load
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForSeq2SeqLM.from_pretrained(args.model)
    # Data processing
    partial_preprocess_function = partial(
        preprocess_function,
        tokenizer=tokenizer,
        args=args
    )
    train_data = train_data.map(
        partial_preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on train dataset",
    )
    test_data = test_data.map(
        partial_preprocess_function,
        batched=True,
        num_proc=8,
        remove_columns=test_data.column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on test dataset",
    )

    training_args = transformers.TrainingArguments(
        output_dir="models",
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        do_train=args.train,
        num_train_epochs=args.epochs,
        save_strategy='steps',
        save_steps=args.eval_steps*4,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.accum,
        eval_accumulation_steps=1,
        report_to="wandb"
    )
    acc_metric = datasets.load_metric("accuracy_metric.py")
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=partial(
            accuracy,
            tokenizer=tokenizer,
            acc_metric=acc_metric
        ),
    )
    trainer.evaluate()
    trainer.train()


def preprocess_function(examples, tokenizer, args):
    if args.debug:
        inputs = examples["question"][:3]
        targets = examples["answer"][:3]
    else:
        inputs = examples["question"]
        targets = examples["answer"]
    model_inputs = tokenizer(
        inputs,
        max_length=args.max_source_length,
        padding="do_not_pad" if args.train_batch_size == 1 else "max_length",
        truncation=True
    )
    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(
        text_target=targets,
        max_length=args.max_target_length,
        padding="do_not_pad" if args.train_batch_size == 1 else "max_length",
        truncation=True
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main(args):
    train_data = data.load_dolly_data()
    test_data = data.load_bbh_tasks(sample=200)
    train(train_data, test_data, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let's do math.")
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, default="./models")
    parser.add_argument('-t', '--train', action="store_true")
    parser.add_argument('-b', '--train_batch_size', type=int, default=16)
    parser.add_argument('-d', '--debug', action="store_true")
    parser.add_argument('--eval-steps', type=int, default=200)
    parser.add_argument('--accum', type=int, default=4)
    parser.add_argument('--overwrite-cache', action="store_true")
    parser.add_argument('--max-source-length', type=int, default=1024)
    parser.add_argument('--max-target-length', type=int, default=256)
    all_args = parser.parse_args()

    wandb.init(config=all_args,
               project="Instruction Tuning Exploration",
               name="test")

    main(all_args)