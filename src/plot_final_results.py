"""Create final result plots for the project report and presentation.

The script uses Pillow instead of matplotlib so it remains lightweight and does
not depend on system font-cache discovery.
"""

from pathlib import Path
from shutil import copy2

import pandas as pd
from PIL import Image, ImageDraw, ImageFont


METRICS_PATH = Path("outputs_final/metrics/model_comparison.csv")
FIGURES_DIR = Path("outputs_final/figures")
REPORT_ASSETS_DIR = Path("reports/assets")
REPORTS_DIR = Path("reports")


MODEL_LABELS = {
    "tfidf_logistic_regression": "TF-IDF + Logistic Regression",
    "tfidf_linear_svm": "TF-IDF + Linear SVM",
    "tfidf_complement_naive_bayes": "TF-IDF + Complement NB",
    "transformer_minilm_logistic_regression": "Transformer Embeddings + Logistic Regression",
    "finetuned_minilm_transformer": "Fine-tuned Transformer",
}


def create_bar_plot(df: pd.DataFrame, metric: str, title: str, output_name: str) -> None:
    """Create and save a simple bar chart for one metric."""
    plot_df = df.copy()
    plot_df["model_label"] = plot_df["model"].map(MODEL_LABELS).fillna(plot_df["model"])

    width = 1100
    height = 620
    margin_left = 90
    margin_right = 50
    margin_top = 80
    margin_bottom = 190
    chart_width = width - margin_left - margin_right
    chart_height = height - margin_top - margin_bottom

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()

    draw.text((margin_left, 30), title, fill="black", font=font)
    draw.line((margin_left, margin_top, margin_left, margin_top + chart_height), fill="black", width=2)
    draw.line(
        (margin_left, margin_top + chart_height, margin_left + chart_width, margin_top + chart_height),
        fill="black",
        width=2,
    )

    for tick in range(0, 11, 2):
        score = tick / 10
        y = margin_top + chart_height - int(chart_height * score)
        draw.line((margin_left - 5, y, margin_left + chart_width, y), fill=(220, 220, 220), width=1)
        draw.text((35, y - 6), f"{score:.1f}", fill="black", font=font)

    values = plot_df[metric].astype(float).tolist()
    labels = plot_df["model_label"].tolist()
    n = max(len(values), 1)
    slot = chart_width / n
    bar_width = min(130, int(slot * 0.55))

    for index, (value, label) in enumerate(zip(values, labels)):
        center = margin_left + int(slot * index + slot / 2)
        bar_height = int(chart_height * value)
        left = center - bar_width // 2
        right = center + bar_width // 2
        top = margin_top + chart_height - bar_height
        bottom = margin_top + chart_height
        draw.rectangle((left, top, right, bottom), fill=(90, 132, 190), outline="black")
        draw.text((left + 8, top - 18), f"{value:.3f}", fill="black", font=font)

        words = label.split()
        lines = []
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) > 18:
                lines.append(current)
                current = word
            else:
                current = candidate
        if current:
            lines.append(current)
        for line_index, line in enumerate(lines[:4]):
            draw.text((center - 75, bottom + 15 + line_index * 16), line, fill="black", font=font)

    image.save(REPORT_ASSETS_DIR / output_name)


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")

    REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METRICS_PATH)

    create_bar_plot(df, "accuracy", "Model Comparison by Accuracy", "model_comparison_accuracy.png")
    create_bar_plot(df, "f1_macro", "Model Comparison by Macro F1-score", "model_comparison_macro_f1.png")
    create_bar_plot(df, "f1_binary", "Model Comparison by Offensive-Class F1-score", "model_comparison_offensive_f1.png")

    for figure_path in FIGURES_DIR.glob("*.png"):
        copy2(figure_path, REPORT_ASSETS_DIR / figure_path.name)

    copy2(METRICS_PATH, REPORTS_DIR / "model_comparison_official_test.csv")

    print("Report assets were created successfully.")
    print(f"Assets directory: {REPORT_ASSETS_DIR}")


if __name__ == "__main__":
    main()
