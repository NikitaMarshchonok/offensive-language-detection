# Data folder

This project supports two dataset options.

## Option A — Recommended official topic dataset: OLID / OffensEval 2019

Use the Offensive Language Identification Dataset (OLID), the official dataset for OffensEval / SemEval 2019 Task 6.

Expected local file example:

```text
data/raw/olid-training-v1.0.tsv
```

Typical OLID columns:

```text
id
tweet
subtask_a
subtask_b
subtask_c
```

For this practical project we use **Sub-task A** only:

```text
NOT = not offensive
OFF = offensive
```

Run:

```bash
python -m src.train_classical --source local --train_path data/raw/olid-training-v1.0.tsv
```

If your local file uses different column names:

```bash
python -m src.train_classical \
  --source local \
  --train_path data/raw/my_dataset.csv \
  --text_col text \
  --label_col label
```

## Option B — Easy fallback: Hugging Face TweetEval offensive

If you want to run the project immediately without manually downloading files:

```bash
python -m src.train_classical --source hf_tweet_eval
```

This requires internet access and the `datasets` package.

## Ethical note

The raw dataset may contain harmful language. The project code intentionally does not print raw text examples into terminal output, reports, or charts.
