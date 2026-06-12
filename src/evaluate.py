"""Evaluation helpers for the Offensive Language Detection project."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
from sklearn.metrics import (
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

    from PIL import Image, ImageDraw, ImageFont

    cm = confusion_matrix(y_true, y_pred)

    width, height = 760, 560
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((30, 24), title, fill="black", font=font)
    draw.text((330, 70), "Predicted label", fill="black", font=font)
    draw.text((28, 260), "True label", fill="black", font=font)

    x0, y0 = 220, 120
    cell_w, cell_h = 190, 150
    max_value = max(int(cm.max()), 1)

    for index, label in enumerate(CLASS_NAMES):
        draw.text((x0 + index * cell_w + 72, y0 - 28), label, fill="black", font=font)
        draw.text((x0 - 62, y0 + index * cell_h + 65), label, fill="black", font=font)

    for row in range(2):
        for col in range(2):
            value = int(cm[row, col])
            intensity = int(235 - 155 * (value / max_value))
            color = (intensity, intensity + 12, 255)
            left = x0 + col * cell_w
            top = y0 + row * cell_h
            right = left + cell_w
            bottom = top + cell_h
            draw.rectangle((left, top, right, bottom), fill=color, outline="black", width=2)
            draw.text((left + 82, top + 64), str(value), fill="black", font=font)

    draw.text((220, 455), f"Labels: {CLASS_NAMES[0]} = not offensive, {CLASS_NAMES[1]} = offensive", fill="black", font=font)
    image.save(output_path)
