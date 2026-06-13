"""Fine-tune a transformer classifier for offensive language detection.

This script provides the strongest modern method in the project: a pre-trained
transformer encoder is fine-tuned directly on the OLID training set and
evaluated on the official OLID Level A test set.

By default it uses the locally cached `sentence-transformers/all-MiniLM-L6-v2`
checkpoint. If internet access is available, another Hugging Face checkpoint
can be used with `--model_name`, for example `distilbert-base-uncased` or
`vinai/bertweet-base`.

Example:
    python -m src.train_transformer_finetune \
        --train_path data/raw/olid-training-v1.0.tsv \
        --test_path data/processed/olid-test-levela-labeled.tsv \
        --text_col tweet \
        --label_col subtask_a \
        --output_dir outputs_final
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data_loader import describe_dataset, load_local_dataset
from src.evaluate import evaluate_predictions, save_confusion_matrix_figure, save_metrics
from src.preprocessing import clean_text_for_transformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class TextClassificationDataset(Dataset):
    """Torch dataset that tokenizes texts for transformer classification."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def predict(model, dataloader: DataLoader, device: torch.device) -> tuple[list[int], list[int]]:
    """Return true labels and model predictions."""
    model.eval()
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            labels = batch["labels"].cpu().tolist()
            batch = move_batch_to_device(batch, device)
            outputs = model(**batch)
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_labels.extend(labels)
            all_predictions.extend(predictions)

    return all_labels, all_predictions


def train_and_evaluate(args: argparse.Namespace) -> dict:
    """Fine-tune transformer model, evaluate it, and save artifacts."""
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
    except ImportError as exc:
        raise ImportError(
            "Transformer fine-tuning requires transformers and torch. "
            "Install them with: pip install -r requirements.txt"
        ) from exc

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    splits = load_local_dataset(
        train_path=args.train_path,
        test_path=args.test_path,
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
    )
    train_df = splits.train.copy()
    test_df = splits.test.copy()

    for df in [train_df, test_df]:
        df["text"] = df["text"].apply(clean_text_for_transformer)

    with (metrics_dir / "finetuned_transformer_dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {"train": describe_dataset(train_df), "test": describe_dataset(test_df)},
            file,
            indent=2,
            ensure_ascii=False,
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Starting fine-tuned transformer experiment.", flush=True)
    print(f"Loading transformer classifier: {args.model_name}", flush=True)
    print(f"Device: {device}", flush=True)
    print("Raw text examples are not printed for ethical and safety reasons.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        local_files_only=args.local_files_only,
    ).to(device)

    train_dataset = TextClassificationDataset(
        texts=train_df["text"].tolist(),
        labels=train_df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    test_dataset = TextClassificationDataset(
        texts=test_df["text"].tolist(),
        labels=test_df["label"].astype(int).tolist(),
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Fine-tuning epoch {epoch + 1}/{args.epochs}")
        for batch in progress:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.item())
            progress.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}", flush=True)

    y_true, predictions = predict(model, test_loader, device)
    metrics = evaluate_predictions(y_true, predictions)
    metrics["transformer_encoder"] = args.model_name
    metrics["training"] = "full sequence-classification fine-tuning"
    metrics["epochs"] = args.epochs
    metrics["learning_rate"] = args.learning_rate

    model_name = "finetuned_minilm_transformer"
    save_metrics(metrics, metrics_dir / f"{model_name}.json")
    save_confusion_matrix_figure(
        y_true=y_true,
        y_pred=predictions,
        output_path=figures_dir / f"{model_name}_confusion_matrix.png",
        title="Confusion Matrix: Fine-tuned Transformer",
    )

    final_model_dir = models_dir / model_name
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    joblib.dump({"model_name": args.model_name, "max_length": args.max_length}, models_dir / f"{model_name}_metadata.joblib")

    comparison_path = metrics_dir / "model_comparison.csv"
    new_row = pd.DataFrame(
        [
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision_binary": metrics["precision_binary"],
                "recall_binary": metrics["recall_binary"],
                "f1_binary": metrics["f1_binary"],
                "f1_macro": metrics["f1_macro"],
            }
        ]
    )
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
        comparison = comparison[comparison["model"] != model_name]
        comparison = pd.concat([comparison, new_row], ignore_index=True)
    else:
        comparison = new_row
    comparison.sort_values("f1_macro", ascending=False).to_csv(comparison_path, index=False)

    print("Fine-tuned transformer metrics:")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune transformer classifier for offensive-language detection.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--allow_download",
        action="store_true",
        help="Allow Hugging Face to download the transformer model if it is not cached.",
    )
    args = parser.parse_args()
    args.local_files_only = not args.allow_download
    return args


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
