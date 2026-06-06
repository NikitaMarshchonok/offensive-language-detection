# Offensive Language Detection in Social Media Texts Using Machine Learning and Transformer-Based Models

**Student:** Nikita Marshchonok  
**Degree:** M.Sc. Software Engineering  
**Course:** Advanced Techniques in Information Retrieval  
**Project type:** Practical programming project  

---

## Abstract

This project investigates automatic offensive language detection in short social-media texts. The main task is binary classification: identifying whether a given post is offensive or not offensive. The project compares classical machine learning methods based on TF-IDF representations with a modern transformer-based approach. The system is evaluated using accuracy, precision, recall, F1-score, macro F1-score, and confusion matrices.

---

## 1. Introduction

Social-media platforms contain large amounts of user-generated text. Some posts may include offensive, abusive, or harmful language. Manual moderation is expensive and does not scale well. Therefore, automatic detection systems can support moderation workflows by identifying potentially problematic content.

The goal of this project is to build a practical machine learning pipeline for offensive language detection and compare classical and transformer-based approaches.

---

## 2. Research Question

**How effective are classical machine learning models compared to transformer-based models for detecting offensive language in social-media texts?**

---

## 3. Dataset

The recommended dataset is the **Offensive Language Identification Dataset (OLID)**, used in OffensEval / SemEval 2019 Task 6. OLID contains English tweets annotated according to a hierarchical three-level annotation scheme.

This project focuses on **Sub-task A**:

| Label | Meaning |
|---|---|
| NOT | Not offensive |
| OFF | Offensive |

For ethical and safety reasons, this report does not include raw offensive examples from the dataset.

---

## 4. Preprocessing

For classical machine learning models, the following preprocessing steps were applied:

1. Lowercasing
2. URL normalization/removal
3. User mention normalization
4. Hashtag sign removal
5. Removal of non-text symbols
6. Extra whitespace normalization

For transformer-based models, preprocessing was kept minimal because transformer tokenizers are designed to preserve contextual information.

---

## 5. Text Representation Models

### 5.1 TF-IDF Representation

TF-IDF converts each text into a sparse numerical vector based on term frequency and inverse document frequency. It gives higher weight to terms that are important in a document but not too common across the whole corpus.

### 5.2 Transformer-Based Representation

Transformer models such as BERT and DistilBERT create contextual representations of text. Unlike TF-IDF, the representation of a word depends on its surrounding context.

---

## 6. Learning Models

The project compares the following models:

| Model | Representation | Description |
|---|---|---|
| Logistic Regression | TF-IDF | Strong classical baseline for text classification |
| Linear SVM | TF-IDF | Effective classifier for high-dimensional sparse text features |
| DistilBERT / BERT | Transformer tokenizer and encoder | Modern contextual model, optional due to compute requirements |

---

## 7. Evaluation Methodology

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Macro F1-score
- Confusion Matrix

The main metric is **Macro F1-score**, because offensive-language datasets may be imbalanced and accuracy alone may be misleading.

---

## 8. Experimental Results

After running the code, insert the generated results from:

```text
outputs/metrics/model_comparison.csv
```

Example table format:

| Model | Accuracy | Precision | Recall | F1 | Macro F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | TBD | TBD | TBD | TBD | TBD |
| TF-IDF + Linear SVM | TBD | TBD | TBD | TBD | TBD |
| DistilBERT / BERT | Optional | Optional | Optional | Optional | Optional |

Confusion matrices should be inserted from:

```text
outputs/figures/
```

---

## 9. Comparison of Two Research Approaches

### 9.1 Classical approach

Classical offensive-language detection systems often use manually engineered features or TF-IDF representations combined with supervised machine learning classifiers such as Logistic Regression or SVM. These methods are fast, interpretable, and easier to deploy.

### 9.2 Transformer-based approach

Transformer-based models such as BERT represent text contextually and can capture more complex semantic relationships. They usually require more compute, but they may perform better in cases where context is important.

### 9.3 Practical comparison

| Criterion | Classical ML | Transformer-Based Model |
|---|---|---|
| Speed | Fast | Slower |
| Compute cost | Low | Higher |
| Interpretability | Better | Lower |
| Context understanding | Limited | Stronger |
| Deployment simplicity | Easier | More complex |
| Expected performance | Good baseline | Often stronger |

---

## 10. Discussion

Discuss:

1. Which model achieved the best Macro F1-score?
2. Did the model confuse offensive and non-offensive posts?
3. Was recall for the offensive class high enough?
4. What are the limitations of the dataset?
5. How could the system be improved?

Possible improvements:

- More training data
- Domain-specific transformer model
- Better class imbalance handling
- Human-in-the-loop moderation
- Multilingual extension, for example Hebrew offensive-language detection

---

## 11. Ethical Considerations

Offensive-language detection is a sensitive task. Automated systems can make mistakes and should not be used as the only decision-making mechanism. False positives may unfairly flag normal speech, while false negatives may miss harmful content. Therefore, such systems should support human moderation rather than fully replace it.

The report avoids displaying raw harmful examples and focuses on aggregate results.

---

## 12. Conclusion

This project implemented a practical offensive-language detection pipeline using classical machine learning and an optional transformer-based method. The work covers the complete pipeline: dataset loading, preprocessing, representation, model training, evaluation, and result analysis. Classical TF-IDF models provide strong and efficient baselines, while transformer-based models offer a more advanced context-aware approach.

---

## References

1. Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., & Kumar, R. (2019). SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media.
2. Zampieri, M. et al. Offensive Language Identification Dataset (OLID), OffensEval / SemEval 2019.
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
