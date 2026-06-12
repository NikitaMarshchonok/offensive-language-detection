"""Train a transformer-embedding classifier for offensive language detection.

This script uses a pre-trained transformer encoder to create contextual sentence
embeddings, then trains a Logistic Regression classifier on top of those
embeddings. It is lighter than full fine-tuning, but it still gives the project
a recent transformer-based method that can be compared with TF-IDF baselines.

Example:
    python -m src.train_transformer_embeddings \
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
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

from src.data_loader import describe_dataset, load_local_dataset
from src.evaluate import evaluate_predictions, save_confusion_matrix_figure, save_metrics
from src.preprocessing import clean_text_for_transformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings using the attention mask."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> np.ndarray:
    """Encode texts into L2-normalized transformer embeddings."""
    model.eval()
    embeddings: list[np.ndarray] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
        batch_texts = texts[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            pooled = mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        embeddings.append(pooled.cpu().numpy())

    return np.vstack(embeddings)


def train_and_evaluate(args: argparse.Namespace) -> dict:
    """Create transformer embeddings, train classifier, and save artifacts."""
    print("Starting transformer embedding experiment.", flush=True)
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Transformer embeddings require transformers and torch. "
            "Install them with: pip install -r requirements.txt"
        ) from exc
    print("Transformers imports completed.", flush=True)

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
    print(f"Loaded data: train={len(train_df)}, test={len(test_df)}", flush=True)

    for df in [train_df, test_df]:
        df["text"] = df["text"].apply(clean_text_for_transformer)

    summary_path = metrics_dir / "transformer_embedding_dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"train": describe_dataset(train_df), "test": describe_dataset(test_df)},
            file,
            indent=2,
            ensure_ascii=False,
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Loading transformer encoder: {args.model_name}", flush=True)
    print(f"Device: {device}", flush=True)
    print("Raw text examples are not printed for ethical and safety reasons.", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model_name, local_files_only=args.local_files_only).to(device)

    X_train = encode_texts(
        texts=train_df["text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    X_test = encode_texts(
        texts=test_df["text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )

    classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    classifier.fit(X_train, train_df["label"])
    predictions = classifier.predict(X_test)

    metrics = evaluate_predictions(test_df["label"], predictions)
    metrics["transformer_encoder"] = args.model_name
    metrics["classifier"] = "LogisticRegression"
    metrics["representation"] = "mean-pooled transformer sentence embeddings"

    model_name = "transformer_minilm_logistic_regression"
    save_metrics(metrics, metrics_dir / f"{model_name}.json")
    save_confusion_matrix_figure(
        y_true=test_df["label"],
        y_pred=predictions,
        output_path=figures_dir / f"{model_name}_confusion_matrix.png",
        title="Confusion Matrix: Transformer Embeddings + Logistic Regression",
    )
    joblib.dump(classifier, models_dir / f"{model_name}.joblib")

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

    print("Transformer embedding metrics:")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer-embedding offensive-language classifier.")
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--model_name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
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
