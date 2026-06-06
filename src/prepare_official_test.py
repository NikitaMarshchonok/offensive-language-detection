"""Prepare the official OLID Level A test set.

This script joins:
1. data/raw/testset-levela.tsv
2. data/raw/labels-levela.csv

and creates:
data/processed/olid-test-levela-labeled.tsv

The script does not print raw tweet texts for ethical and safety reasons.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def prepare_official_test_set() -> Path:
    """Create a labeled official OLID Level A test file."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    test_path = RAW_DIR / "testset-levela.tsv"
    labels_path = RAW_DIR / "labels-levela.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"Missing file: {test_path}")

    if not labels_path.exists():
        raise FileNotFoundError(f"Missing file: {labels_path}")

    test = pd.read_csv(test_path, sep="\t")

    labels = pd.read_csv(
        labels_path,
        header=None,
        names=["id", "subtask_a"],
    )

    labels = labels[labels["id"].astype(str).str.lower() != "id"]

    test["id"] = test["id"].astype(str).str.strip()
    labels["id"] = labels["id"].astype(str).str.strip()

    merged = test.merge(labels, on="id", how="inner")

    output_path = PROCESSED_DIR / "olid-test-levela-labeled.tsv"
    merged[["id", "tweet", "subtask_a"]].to_csv(output_path, sep="\t", index=False)

    print(f"Saved: {output_path}")
    print(f"Shape: {merged.shape}")
    print("Label distribution:")
    print(merged["subtask_a"].value_counts())

    return output_path


if __name__ == "__main__":
    prepare_official_test_set()
