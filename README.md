# Offensive Language Detection in Social Media Texts

This project implements an end-to-end offensive-language detection pipeline for social-media text. The revised version follows recent scientific work by treating TF-IDF models as **classical baselines** and adding a **transformer-based approach** for comparison.

The binary task is:

- `NOT` — not offensive
- `OFF` — offensive

The project was developed as a practical Software Engineering / Information Retrieval course project.

---

## Research Question

How effective are classical TF-IDF machine-learning baselines for offensive language detection, and how can a recent transformer-based approach be integrated into the same experimental pipeline?

---

## Scientific Basis

The revised report is based on offensive-language and abusive-language detection literature, including:

- OLID / OffensEval for offensive-language identification in tweets.
- BERT, Sentence-BERT, BERTweet, and HateBERT as transformer-based NLP foundations.
- Recent SemEval-2023 EDOS work on explainable abusive/sexist language detection.
- Recent cross-lingual and multilingual offensive-language detection studies from 2024.

See [reports/final_report.md](reports/final_report.md) for the full related-work section and references.

---

## Dataset

The project uses the **Offensive Language Identification Dataset (OLID)** from **OffensEval / SemEval 2019**.

The selected task is **Level A binary classification**:

- `NOT` — not offensive
- `OFF` — offensive

Expected local files:

```text
data/raw/olid-training-v1.0.tsv
data/raw/testset-levela.tsv
data/raw/labels-levela.csv
```

The official labeled test file is prepared by:

```bash
python -m src.prepare_official_test
```

---

## Methods

### Classical Baselines

The following TF-IDF models are implemented as reproducible baselines:

1. TF-IDF + Logistic Regression
2. TF-IDF + Linear SVM
3. TF-IDF + Complement Naive Bayes

These methods are fast, interpretable, and useful for comparison, but they are not presented as recent state of the art.

### Transformer-Based Approach

The modern approach is implemented in:

```text
src/train_transformer_finetune.py
```

It fine-tunes `sentence-transformers/all-MiniLM-L6-v2` as a transformer sequence-classification model directly on the OLID training set. This is the main recent method in the revised project.

The repository also includes a lighter transformer-embedding baseline:

```text
src/train_transformer_embeddings.py
```

It uses the same transformer encoder to create contextual sentence embeddings, then trains Logistic Regression on top of those embeddings.

---

## Evaluation Metrics

Models are evaluated with:

- Accuracy
- Precision for the offensive class
- Recall for the offensive class
- F1-score for the offensive class
- Macro F1-score
- Confusion matrix

The main metric is **Macro F1-score**, because the dataset is imbalanced and accuracy alone can be misleading.

---

## Verified Results

The verified local experiment was evaluated on the official OLID Level A test set.

The official test set contains:

```text
860 examples
620 NOT
240 OFF
```

| Model | Accuracy | Precision OFF | Recall OFF | F1 OFF | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Fine-tuned Transformer | 0.8430 | 0.7692 | 0.6250 | 0.6897 | 0.7923 |
| Transformer Embeddings + Logistic Regression | 0.7779 | 0.5939 | 0.6458 | 0.6188 | 0.7310 |
| TF-IDF + Logistic Regression | 0.7791 | 0.6033 | 0.6083 | 0.6058 | 0.7262 |
| TF-IDF + Complement Naive Bayes | 0.7988 | 0.7134 | 0.4667 | 0.5642 | 0.7167 |
| TF-IDF + Linear SVM | 0.7744 | 0.6009 | 0.5708 | 0.5855 | 0.7153 |

The fine-tuned transformer achieves the best Accuracy, Macro F1-score, and offensive-class F1-score. TF-IDF + Logistic Regression is the strongest classical baseline.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

Run the verified baseline pipeline:

```bash
python -m src.run_final_experiment
```

Run the transformer-embedding experiment as well:

```bash
python -m src.run_final_experiment --include_transformer
```

Run the full fine-tuned transformer experiment as well:

```bash
python -m src.run_final_experiment --include_finetuning
```

Run all verified experiments:

```bash
python -m src.run_final_experiment --include_transformer --include_finetuning
```

Run the fine-tuned transformer experiment directly:

```bash
python -m src.train_transformer_finetune \
  --train_path data/raw/olid-training-v1.0.tsv \
  --test_path data/processed/olid-test-levela-labeled.tsv \
  --text_col tweet \
  --label_col subtask_a \
  --output_dir outputs_final
```

Run the transformer-embedding experiment directly:

```bash
python -m src.train_transformer_embeddings \
  --train_path data/raw/olid-training-v1.0.tsv \
  --test_path data/processed/olid-test-levela-labeled.tsv \
  --text_col tweet \
  --label_col subtask_a \
  --output_dir outputs_final
```

If the transformer model is not cached locally and internet access is available, add:

```bash
--allow_download
```

Generate report assets:

```bash
python -m src.plot_final_results
```

---

## Project Structure

```text
offensive-language-detection/
├── data/
├── notebooks/
├── outputs_final/
├── reports/
│   ├── assets/
│   ├── final_report.md
│   └── presentation_outline.md
├── src/
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── prepare_official_test.py
│   ├── train_classical.py
│   ├── train_transformer.py
│   ├── train_transformer_embeddings.py
│   └── run_final_experiment.py
├── requirements.txt
└── README.md
```

---

## Ethical Note

This project works with offensive-language data. Raw offensive examples are not printed in the terminal output, README, report, or visual assets. The goal is responsible model evaluation, not reproduction of harmful content.

Automatic moderation systems can make mistakes. They should support human moderation rather than fully replace it.

---

## Main Conclusion

The project now aligns with recent offensive-language detection research. TF-IDF models are used as classical baselines, and a fine-tuned transformer classifier is included and evaluated for modern comparison. The best overall model is **Fine-tuned Transformer**, with Macro F1-score `0.7923` and offensive-class F1-score `0.6897`. The best classical baseline is **TF-IDF + Logistic Regression**, with Macro F1-score `0.7262`.
