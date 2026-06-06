"""Evaluation helpers for the Offensive Language Detection project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


CLASS_NAMES = ["NOT", "OFF"]


def evaluate_predictions(y_true: Iterable[int], y_pred: Iterable[int]) -> Dict:
    """Compute a standard set of classification metrics."""
    y_true = np.asarray(list(y_true))
    y_pred = np.asarray(list(y_pred))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_binary": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_binary": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_binary": float(f1_score(y_true, y_pred, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def save_metrics(metrics: Dict, output_path: str | Path) -> None:
    """Save metrics as JSON."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)


def save_confusion_matrix_figure(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    output_path: str | Path,
    title: str,
) -> None:
    """Create and save a confusion matrix figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)

    fig, ax = plt.subplots(figsize=(6, 5))
    display.plot(ax=ax, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
