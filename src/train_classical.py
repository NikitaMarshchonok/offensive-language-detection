"""Train classical ML baselines for offensive language detection.

Models:
1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. TF-IDF + Complement Naive Bayes

Example commands
----------------
Hugging Face TweetEval fallback:
    python -m src.train_classical --source hf_tweet_eval

Local OLID-style file:
    python -m src.train_classical --source local --train_path data/raw/olid-training-v1.0.tsv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data_loader import describe_dataset, load_hf_tweet_eval, load_local_dataset
from src.evaluate import evaluate_predictions, save_confusion_matrix_figure, save_metrics
from src.preprocessing import clean_text_for_classical_ml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "outputs"


def build_models() -> dict[str, Pipeline]:
    """Create the classical ML pipelines."""
    tfidf = TfidfVectorizer(
        preprocessor=clean_text_for_classical_ml,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50_000,
        sublinear_tf=True,
    )

    return {
        "tfidf_logistic_regression": Pipeline(
            steps=[
                ("tfidf", tfidf),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "tfidf_linear_svm": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=clean_text_for_classical_ml,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", LinearSVC(class_weight="balanced", random_state=42)),
            ]
        ),
        "tfidf_complement_naive_bayes": Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        preprocessor=clean_text_for_classical_ml,
                        ngram_range=(1, 2),
                        min_df=2,
                        max_df=0.95,
                        max_features=50_000,
                        sublinear_tf=True,
                    ),
                ),
                ("classifier", ComplementNB(alpha=0.5)),
            ]
        ),
    }


def load_data(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test data according to CLI arguments."""
    if args.source == "hf_tweet_eval":
        splits = load_hf_tweet_eval()
        train_df = pd.concat([splits.train, splits.validation], ignore_index=True)
        test_df = splits.test
        return train_df, test_df

    if not args.train_path:
        raise ValueError("For --source local you must provide --train_path")

    splits = load_local_dataset(
        train_path=args.train_path,
        test_path=args.test_path,
        text_col=args.text_col,
        label_col=args.label_col,
        test_size=args.test_size,
    )
    return splits.train, splits.test


def train_and_evaluate(args) -> pd.DataFrame:
    """Train models, evaluate them, and save artifacts."""
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    models_dir = output_dir / "models"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(args)

    dataset_summary = {
        "train": describe_dataset(train_df),
        "test": describe_dataset(test_df),
    }
    with (metrics_dir / "dataset_summary.json").open("w", encoding="utf-8") as file:
        json.dump(dataset_summary, file, indent=2, ensure_ascii=False)

    print("Dataset summary was saved to outputs/metrics/dataset_summary.json")
    print("Raw text examples are not printed for ethical and safety reasons.")

    X_train = train_df["text"]
    y_train = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]

    rows = []
    for model_name, pipeline in build_models().items():
        print(f"Training model: {model_name}")
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        metrics = evaluate_predictions(y_test, predictions)
        save_metrics(metrics, metrics_dir / f"{model_name}.json")
        save_confusion_matrix_figure(
            y_true=y_test,
            y_pred=predictions,
            output_path=figures_dir / f"{model_name}_confusion_matrix.png",
            title=f"Confusion Matrix: {model_name}",
        )
        joblib.dump(pipeline, models_dir / f"{model_name}.joblib")

        rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision_binary": metrics["precision_binary"],
                "recall_binary": metrics["recall_binary"],
                "f1_binary": metrics["f1_binary"],
                "f1_macro": metrics["f1_macro"],
            }
        )

    comparison = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    comparison.to_csv(metrics_dir / "model_comparison.csv", index=False)
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    return comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical offensive-language classifiers.")
    parser.add_argument("--source", choices=["hf_tweet_eval", "local"], default="hf_tweet_eval")
    parser.add_argument("--train_path", type=str, default=None)
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--text_col", type=str, default=None)
    parser.add_argument("--label_col", type=str, default=None)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_evaluate(args)


if __name__ == "__main__":
    main()
