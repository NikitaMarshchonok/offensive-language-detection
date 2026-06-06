"""Optional transformer fine-tuning for offensive language detection.

This script trains a transformer classifier such as DistilBERT. It is optional
because it requires more time and compute than the classical baselines.

Example:
    python -m src.train_transformer --source hf_tweet_eval --epochs 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.data_loader import load_hf_tweet_eval, load_local_dataset
from src.preprocessing import clean_text_for_transformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


def load_data(args):
    if args.source == "hf_tweet_eval":
        splits = load_hf_tweet_eval()
        return splits.train, splits.validation, splits.test

    if not args.train_path:
        raise ValueError("For --source local you must provide --train_path")

    splits = load_local_dataset(
        train_path=args.train_path,
        test_path=args.test_path,
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
    )
    return splits.train, None, splits.test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_binary": precision_score(labels, predictions, zero_division=0),
        "recall_binary": recall_score(labels, predictions, zero_division=0),
        "f1_binary": f1_score(labels, predictions, zero_division=0),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a transformer classifier.")
    parser.add_argument("--source", choices=["hf_tweet_eval", "local"], default="hf_tweet_eval")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise ImportError(
            "Transformer training requires datasets, transformers, accelerate and torch. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    output_dir = Path(args.output_dir)
    transformer_dir = output_dir / "models" / "transformer"
    metrics_dir = output_dir / "metrics"
    transformer_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_df, validation_df, test_df = load_data(args)
    if validation_df is None:
        validation_df = test_df.copy()

    for df in [train_df, validation_df, test_df]:
        df["text"] = df["text"].apply(clean_text_for_transformer)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(validation_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)
    test_ds = Dataset.from_pandas(test_df[["text", "label"]], preserve_index=False).map(tokenize, batched=True)

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    test_ds = test_ds.rename_column("label", "labels")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)

    training_kwargs = dict(
        output_dir=str(transformer_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )

    # Transformers versions differ: some use evaluation_strategy, newer versions
    # may use eval_strategy. This keeps the script more robust.
    try:
        training_args = TrainingArguments(evaluation_strategy="epoch", **training_kwargs)
    except TypeError:
        training_args = TrainingArguments(eval_strategy="epoch", **training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Fine-tuning transformer model. Raw text examples are not printed.")
    trainer.train()
    test_metrics = trainer.evaluate(test_ds)

    trainer.save_model(str(transformer_dir / "final"))
    tokenizer.save_pretrained(str(transformer_dir / "final"))

    with (metrics_dir / "distilbert_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(test_metrics, file, indent=2, ensure_ascii=False)

    print("Transformer metrics:")
    print(test_metrics)


if __name__ == "__main__":
    main()
