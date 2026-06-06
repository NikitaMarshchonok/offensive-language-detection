# Offensive Language Detection in Social Media Texts

Practical project for the course **Advanced Techniques in Information Retrieval** / Software Engineering master's degree.

## 1. Project idea

The goal of this project is to build and evaluate a machine learning system that detects offensive language in short social-media texts.

The task is binary classification:

```text
Input: social-media text
Output: OFFENSIVE or NOT_OFFENSIVE
```

The project follows the required practical-project pipeline:

```text
Dataset → Preprocessing → Representation Model → Learning Model → Evaluation → Report → Presentation
```

## 2. Dataset

Recommended dataset:

**OLID — Offensive Language Identification Dataset**, used in **OffensEval / SemEval 2019 Task 6**.

For this project we focus on **Sub-task A**:

```text
OFF = offensive
NOT = not offensive
```

The project also supports the Hugging Face `tweet_eval` offensive split as an easy fallback for running the code quickly.

## 3. Methods

The project compares two families of approaches:

### Classical Machine Learning

1. **TF-IDF + Logistic Regression**
2. **TF-IDF + Linear SVM**

### Transformer-Based Model

3. **DistilBERT / BERT fine-tuning**

The transformer model is optional because it requires more compute.

## 4. Evaluation metrics

The models are evaluated using:

```text
Accuracy
Precision
Recall
F1-score
Macro F1-score
Confusion Matrix
```

Main metric: **Macro F1-score**, because offensive-language datasets may be imbalanced.

## 5. Project structure

```text
offensive-language-detection/
│
├── data/
│   ├── raw/                 # local datasets, not committed to Git
│   ├── processed/           # processed data, not committed to Git
│   └── README.md
│
├── notebooks/
│   └── 01_offensive_language_detection.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── evaluate.py
│   ├── train_classical.py
│   ├── train_transformer.py
│   └── run_all.py
│
├── outputs/
│   ├── figures/             # confusion matrix images
│   ├── metrics/             # JSON/CSV metrics
│   └── models/              # trained models, not committed to Git
│
├── reports/
│   ├── final_report_template.md
│   └── presentation_outline.md
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Installation

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## 7. Run classical models

### Option A — with Hugging Face fallback

```bash
python -m src.train_classical --source hf_tweet_eval
```

### Option B — with local OLID file

Put the OLID training file here:

```text
data/raw/olid-training-v1.0.tsv
```

Then run:

```bash
python -m src.train_classical --source local --train_path data/raw/olid-training-v1.0.tsv
```

## 8. Run optional transformer model

```bash
python -m src.train_transformer --source hf_tweet_eval --epochs 2
```

For local OLID:

```bash
python -m src.train_transformer --source local --train_path data/raw/olid-training-v1.0.tsv --epochs 2
```

## 9. Expected outputs

After running the classical training script, the project creates:

```text
outputs/metrics/dataset_summary.json
outputs/metrics/tfidf_logistic_regression.json
outputs/metrics/tfidf_linear_svm.json
outputs/metrics/model_comparison.csv
outputs/figures/tfidf_logistic_regression_confusion_matrix.png
outputs/figures/tfidf_linear_svm_confusion_matrix.png
outputs/models/tfidf_logistic_regression.joblib
outputs/models/tfidf_linear_svm.joblib
```

## 10. Ethical note

The dataset may contain harmful language. For this reason, the code avoids printing raw examples in terminal output, report templates, and figures. The evaluation focuses on aggregate metrics only.

## 11. References

- Zampieri et al. (2019). SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media.
- OLID / OffensEval 2019 official dataset page.
- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
