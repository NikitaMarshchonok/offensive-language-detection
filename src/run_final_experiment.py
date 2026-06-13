"""Run the full final experiment for the practical project.

This script:
1. prepares the official OLID Level A test set;
2. trains classical TF-IDF baseline models;
3. trains a transformer-embedding model;
4. saves metrics, confusion matrices, and trained models.

Raw tweet texts are not printed for ethical and safety reasons.
"""

from __future__ import annotations

import subprocess
import sys
import argparse


def run_command(command: list[str]) -> None:
    """Run a terminal command and stop if it fails."""
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final offensive-language detection experiments.")
    parser.add_argument(
        "--include_transformer",
        action="store_true",
        help="Also run the transformer-embedding experiment. This requires a working transformers/torch setup.",
    )
    parser.add_argument(
        "--include_finetuning",
        action="store_true",
        help="Also run full transformer sequence-classification fine-tuning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_command([sys.executable, "-m", "src.prepare_official_test"])

    run_command(
        [
            sys.executable,
            "-m",
            "src.train_classical",
            "--source",
            "local",
            "--train_path",
            "data/raw/olid-training-v1.0.tsv",
            "--test_path",
            "data/processed/olid-test-levela-labeled.tsv",
            "--text_col",
            "tweet",
            "--label_col",
            "subtask_a",
            "--output_dir",
            "outputs_final",
        ]
    )

    if args.include_transformer:
        run_command(
            [
                sys.executable,
                "-m",
                "src.train_transformer_embeddings",
                "--train_path",
                "data/raw/olid-training-v1.0.tsv",
                "--test_path",
                "data/processed/olid-test-levela-labeled.tsv",
                "--text_col",
                "tweet",
                "--label_col",
                "subtask_a",
                "--output_dir",
                "outputs_final",
            ]
        )

    if args.include_finetuning:
        run_command(
            [
                sys.executable,
                "-m",
                "src.train_transformer_finetune",
                "--train_path",
                "data/raw/olid-training-v1.0.tsv",
                "--test_path",
                "data/processed/olid-test-levela-labeled.tsv",
                "--text_col",
                "tweet",
                "--label_col",
                "subtask_a",
                "--output_dir",
                "outputs_final",
            ]
        )


if __name__ == "__main__":
    main()
