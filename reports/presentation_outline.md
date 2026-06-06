# Presentation Outline — Offensive Language Detection

## Slide 1 — Title

**Offensive Language Detection in Social Media Texts Using Machine Learning and Transformer-Based Models**

Student: Nikita Marshchonok  
M.Sc. Software Engineering

---

## Slide 2 — Problem

- Social media contains large amounts of user-generated content.
- Some posts may include offensive or harmful language.
- Manual moderation is expensive and difficult to scale.
- Automatic detection can support moderation workflows.

---

## Slide 3 — Goal and Research Question

**Goal:** Build and evaluate a practical offensive-language detection system.

**Research question:**

How effective are classical ML models compared to transformer-based models for offensive language detection?

---

## Slide 4 — Dataset

- Dataset: OLID / OffensEval 2019
- Domain: English social-media posts
- Task used in this project: Sub-task A
- Labels:
  - NOT — not offensive
  - OFF — offensive

Ethical note: raw harmful examples are not shown.

---

## Slide 5 — Pipeline

```text
Dataset → Preprocessing → Representation → Model Training → Evaluation → Results
```

---

## Slide 6 — Preprocessing

For classical models:

- Lowercasing
- URL normalization/removal
- User mention normalization
- Symbol cleaning
- Extra-space normalization

For transformer models:

- Minimal cleaning
- Transformer tokenizer

---

## Slide 7 — Models

| Model | Representation |
|---|---|
| Logistic Regression | TF-IDF |
| Linear SVM | TF-IDF |
| DistilBERT / BERT | Transformer embeddings |

---

## Slide 8 — Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion Matrix

Main metric: **Macro F1-score**

---

## Slide 9 — Results

Insert the final comparison table from:

```text
outputs/metrics/model_comparison.csv
```

Insert confusion matrix images from:

```text
outputs/figures/
```

---

## Slide 10 — Discussion

- Classical models are fast and easy to deploy.
- Transformer models understand context better but require more resources.
- False positives and false negatives are important ethical concerns.
- The system should support human moderation, not replace it fully.

---

## Slide 11 — Conclusion

- A complete offensive-language detection pipeline was implemented.
- TF-IDF models provide strong baselines.
- Transformer-based models can be added for stronger contextual understanding.
- The project satisfies the practical requirements: dataset, preprocessing, representation, learning model, evaluation, report, and presentation.
