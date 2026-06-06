"""Run the full final experiment for the practical project.

This script:
1. prepares the official OLID Level A test set;
2. trains classical ML models;
3. saves metrics, confusion matrices, and trained models.

Raw tweet texts are not printed for ethical and safety reasons.
"""

from __future__ import annotations

import subprocess
import sys


def run_command(command: list[str]) -> None:
    """Run a terminal command and stop if it fails."""
    print("\n$", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
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


if __name__ == "__main__":
    main()
