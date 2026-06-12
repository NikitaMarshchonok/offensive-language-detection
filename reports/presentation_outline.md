# Presentation Outline — Offensive Language Detection

## Slide 1 — Title

**Offensive Language Detection in Social Media Texts: Classical Baselines and a Transformer-Based Approach**

Student: Nikita Marshchonok  
M.Sc. Software Engineering

---

## Slide 2 — Problem

- Social media contains large amounts of user-generated content.
- Some posts include offensive or harmful language.
- Manual moderation is expensive and difficult to scale.
- Automatic detection can support human moderation workflows.

---

## Slide 3 — Research Question

How effective are classical TF-IDF machine-learning baselines for offensive-language detection, and how can a recent transformer-based approach be integrated into the same experimental pipeline?

---

## Slide 4 — Scientific Basis

- OLID / OffensEval introduced a standard offensive-language detection benchmark.
- Recent research uses BERT-family and transformer-based models.
- SemEval-2023 EDOS systems use models such as BERT, RoBERTa, DistilBERT, HateBERT, and XLM-T.
- 2024 work emphasizes cross-lingual and multilingual offensive-language detection.

---

## Slide 5 — Dataset

- Dataset: OLID / OffensEval 2019
- Domain: English tweets
- Task: Level A binary classification
- Labels:
  - NOT — not offensive
  - OFF — offensive

Ethical note: raw harmful examples are not shown.

---

## Slide 6 — Pipeline

```text
OLID Dataset → Preprocessing → Representation → Model Training → Evaluation → Report Assets
```

---

## Slide 7 — Methods

| Model | Representation | Role |
|---|---|---|
| Logistic Regression | TF-IDF | Classical baseline |
| Linear SVM | TF-IDF | Classical baseline |
| Complement Naive Bayes | TF-IDF | Classical baseline |
| Logistic Regression | Transformer embeddings | Recent transformer-based approach |

---

## Slide 8 — Transformer-Based Approach

- Encoder: `sentence-transformers/all-MiniLM-L6-v2`
- Representation: contextual sentence embeddings
- Classifier: Logistic Regression
- Purpose: compare modern contextual representations with sparse TF-IDF baselines
- Full DistilBERT-style fine-tuning script is also included

---

## Slide 9 — Evaluation Metrics

- Accuracy
- Precision for OFF
- Recall for OFF
- F1-score for OFF
- Macro F1-score
- Confusion matrix

Main metric: **Macro F1-score**

---

## Slide 10 — Verified Baseline Results

| Model | Accuracy | F1 OFF | Macro F1 |
|---|---:|---:|---:|
| Transformer Embeddings + Logistic Regression | 0.7779 | 0.6188 | 0.7310 |
| TF-IDF + Logistic Regression | 0.7791 | 0.6058 | 0.7262 |
| TF-IDF + Complement Naive Bayes | 0.7988 | 0.5642 | 0.7167 |
| TF-IDF + Linear SVM | 0.7744 | 0.5855 | 0.7153 |

Best overall model: **Transformer Embeddings + Logistic Regression**.
Best classical baseline: **TF-IDF + Logistic Regression**.

---

## Slide 11 — Discussion

- TF-IDF baselines are fast and reproducible.
- They are limited in contextual understanding.
- Transformer embeddings better match recent NLP research and improve offensive-class F1.
- Offensive-class recall remains challenging.
- The system should support human moderation rather than replace it.

---

## Slide 12 — Conclusion

- The revised project is based on recent scientific work.
- Older TF-IDF methods are clearly positioned as baselines.
- A transformer-based approach is implemented and evaluated for modern comparison.
- The report includes updated literature, method descriptions, references, and verified baseline results.
