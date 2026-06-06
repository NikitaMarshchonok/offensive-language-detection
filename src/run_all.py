"""One-command runner for the practical project.

This script runs the classical ML part of the project and creates the first set
of outputs needed for the report:
- dataset summary
- metrics JSON files
- model comparison CSV
- confusion matrix images
- saved sklearn models
"""

from __future__ import annotations

from src.train_classical import main


if __name__ == "__main__":
    main()
