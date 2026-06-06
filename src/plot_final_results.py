"""Create final result plots for the project report and presentation.

This script reads the final model comparison table and creates visualizations:
1. Accuracy comparison
2. Macro F1 comparison
3. Offensive-class F1 comparison

It also copies confusion matrices into reports/assets.
"""

from pathlib import Path
from shutil import copy2

import matplotlib.pyplot as plt
import pandas as pd


METRICS_PATH = Path("outputs_final/metrics/model_comparison.csv")
FIGURES_DIR = Path("outputs_final/figures")
REPORT_ASSETS_DIR = Path("reports/assets")
REPORTS_DIR = Path("reports")


MODEL_LABELS = {
    "tfidf_logistic_regression": "TF-IDF + Logistic Regression",
    "tfidf_linear_svm": "TF-IDF + Linear SVM",
    "tfidf_complement_naive_bayes": "TF-IDF + Complement NB",
}


def create_bar_plot(df: pd.DataFrame, metric: str, title: str, output_name: str) -> None:
    """Create and save a single bar chart for one metric."""
    plot_df = df.copy()
    plot_df["model_label"] = plot_df["model"].map(MODEL_LABELS).fillna(plot_df["model"])

    ax = plot_df.plot(
        kind="bar",
        x="model_label",
        y=metric,
        legend=False,
        rot=20,
        figsize=(9, 5),
    )

    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)

    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f")

    plt.tight_layout()
    plt.savefig(REPORT_ASSETS_DIR / output_name, dpi=200)
    plt.close()


def main() -> None:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"Missing metrics file: {METRICS_PATH}")

    REPORT_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METRICS_PATH)

    create_bar_plot(
        df=df,
        metric="accuracy",
        title="Model Comparison by Accuracy",
        output_name="model_comparison_accuracy.png",
    )

    create_bar_plot(
        df=df,
        metric="f1_macro",
        title="Model Comparison by Macro F1-score",
        output_name="model_comparison_macro_f1.png",
    )

    create_bar_plot(
        df=df,
        metric="f1_binary",
        title="Model Comparison by Offensive-Class F1-score",
        output_name="model_comparison_offensive_f1.png",
    )

    for figure_path in FIGURES_DIR.glob("*.png"):
        copy2(figure_path, REPORT_ASSETS_DIR / figure_path.name)

    copy2(METRICS_PATH, REPORTS_DIR / "model_comparison_official_test.csv")

    print("Report assets were created successfully.")
    print(f"Assets directory: {REPORT_ASSETS_DIR}")


if __name__ == "__main__":
    main()
