"""Data loading utilities for the Offensive Language Detection project.

The module supports two workflows:
1. Local OLID-style files, where the text column is often named `tweet` and
   the label column is often named `subtask_a`.
2. Hugging Face TweetEval offensive dataset as an easy-to-run fallback.

The code intentionally does not print raw texts, because the dataset may contain
harmful language. Only aggregate statistics should be displayed in reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


TEXT_CANDIDATES = ["tweet", "text", "content", "sentence", "comment"]
LABEL_CANDIDATES = ["subtask_a", "label", "class", "target", "y"]


@dataclass
class DatasetSplits:
    """Container for train/validation/test splits."""

    train: pd.DataFrame
    validation: Optional[pd.DataFrame]
    test: pd.DataFrame


def _detect_separator(file_path: str | Path) -> str:
    """Guess a reasonable separator from the file extension."""
    suffix = Path(file_path).suffix.lower()
    if suffix == ".tsv":
        return "\t"
    return ","


def read_table(file_path: str | Path) -> pd.DataFrame:
    """Read a CSV/TSV file into a DataFrame.

    Parameters
    ----------
    file_path:
        Path to a local CSV or TSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    sep = _detect_separator(file_path)
    return pd.read_csv(file_path, sep=sep)


def _find_column(df: pd.DataFrame, candidates: list[str], explicit: Optional[str] = None) -> str:
    """Find a column by explicit name or common alternatives."""
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Column '{explicit}' was not found. Available columns: {list(df.columns)}")
        return explicit

    lower_to_original = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]

    raise ValueError(
        f"Could not infer column. Tried: {candidates}. Available columns: {list(df.columns)}"
    )


def map_label(value) -> int:
    """Map common offensive-language labels to binary integers.

    Returns
    -------
    int
        1 for offensive, 0 for non-offensive.
    """
    if pd.isna(value):
        raise ValueError("Label contains missing value")

    normalized = str(value).strip().lower()

    offensive_values = {"off", "offensive", "1", "true", "yes", "hate", "abusive"}
    non_offensive_values = {"not", "not_offensive", "non-offensive", "normal", "0", "false", "no"}

    if normalized in offensive_values:
        return 1
    if normalized in non_offensive_values:
        return 0

    # Hugging Face TweetEval uses integer labels, commonly 0/1.
    if normalized in {"0.0", "1.0"}:
        return int(float(normalized))

    raise ValueError(f"Unknown label value: {value!r}")


def standardize_dataframe(
    df: pd.DataFrame,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
) -> pd.DataFrame:
    """Return a clean DataFrame with exactly two columns: text and label."""
    text_col = _find_column(df, TEXT_CANDIDATES, text_col)
    label_col = _find_column(df, LABEL_CANDIDATES, label_col)

    result = df[[text_col, label_col]].copy()
    result.columns = ["text", "label"]
    result["text"] = result["text"].astype(str).fillna("")
    result["label"] = result["label"].apply(map_label).astype(int)
    result = result[result["text"].str.strip().ne("")].reset_index(drop=True)
    return result


def load_local_dataset(
    train_path: str | Path,
    test_path: Optional[str | Path] = None,
    text_col: Optional[str] = None,
    label_col: Optional[str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplits:
    """Load local data and return train/test splits.

    If a separate test file is not provided, the function creates a stratified
    train/test split from the training file.
    """
    train_df = standardize_dataframe(read_table(train_path), text_col=text_col, label_col=label_col)

    if test_path:
        test_df = standardize_dataframe(read_table(test_path), text_col=text_col, label_col=label_col)
        return DatasetSplits(train=train_df, validation=None, test=test_df)

    train_split, test_split = train_test_split(
        train_df,
        test_size=test_size,
        random_state=random_state,
        stratify=train_df["label"],
    )
    return DatasetSplits(
        train=train_split.reset_index(drop=True),
        validation=None,
        test=test_split.reset_index(drop=True),
    )


def load_hf_tweet_eval() -> DatasetSplits:
    """Load the Hugging Face TweetEval offensive split.

    This is an easy-to-run fallback when the original OLID files are not stored
    locally. It requires internet access and the `datasets` package.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The `datasets` package is required for Hugging Face loading. "
            "Install it with: pip install datasets"
        ) from exc

    dataset = load_dataset("tweet_eval", "offensive")

    def to_df(split_name: str) -> pd.DataFrame:
        df = pd.DataFrame(dataset[split_name])
        return standardize_dataframe(df, text_col="text", label_col="label")

    return DatasetSplits(
        train=to_df("train"),
        validation=to_df("validation"),
        test=to_df("test"),
    )


def describe_dataset(df: pd.DataFrame) -> dict:
    """Return aggregate dataset statistics without exposing raw text."""
    counts = df["label"].value_counts().sort_index().to_dict()
    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "label_distribution": {
            "not_offensive_0": int(counts.get(0, 0)),
            "offensive_1": int(counts.get(1, 0)),
        },
        "avg_text_length_chars": float(df["text"].astype(str).str.len().mean()),
    }
