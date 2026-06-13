# Offensive Language Detection in Social Media Texts

## A Comparative Study of Classical TF-IDF Baselines and Fine-Tuned Transformer Models

**Student:** Nikita Marshchonok  
**Degree:** M.Sc. Software Engineering  
**Dataset:** OLID / OffensEval 2019, Level A  

---

## Abstract

This project studies automatic offensive language detection in short social-media texts. The task is binary classification: identifying whether a post is offensive (`OFF`) or not offensive (`NOT`). The revised project is explicitly based on recent scientific work in offensive, abusive, and hateful language detection. Classical TF-IDF models are kept only as reproducible baseline methods, while the main modern method is a fine-tuned transformer classifier.

The project uses the OLID dataset from OffensEval / SemEval 2019 and evaluates all models on the official Level A test set. Five models are compared: three TF-IDF baselines, a transformer-embedding classifier, and a fine-tuned transformer classifier. The best result is achieved by the fine-tuned transformer model with **Accuracy = 0.8430**, **Macro F1 = 0.7923**, and **F1 for the offensive class = 0.6897**. This substantially improves over the best classical baseline, TF-IDF + Logistic Regression, which achieves **Macro F1 = 0.7262**.

**Keywords:** offensive language detection, hate speech detection, text classification, OLID, TF-IDF, transformer, fine-tuning, BERT, social media.

---

## 1. Introduction

Social-media platforms contain large amounts of user-generated text. Some posts include offensive, abusive, or harmful language. Manual moderation is expensive, emotionally difficult, and hard to scale. Automatic offensive-language detection can support moderation workflows by identifying potentially harmful content for human review.

Early text-classification systems often relied on sparse lexical representations such as bag-of-words or TF-IDF combined with classifiers such as Logistic Regression, Support Vector Machines, or Naive Bayes. These methods are still useful because they are simple, fast, reproducible, and interpretable. However, they are no longer considered the most recent approach for offensive-language detection. Recent research has moved toward transformer-based models, including BERT, RoBERTa, BERTweet, HateBERT, XLM-R, and large language models.

The main purpose of this revised project is therefore to compare older classical baselines with a recent transformer-based method and to ground the project in relevant scientific literature.

---

## 2. Research Question

The project addresses the following research question:

> How effective are classical TF-IDF machine-learning baselines compared with transformer-based models for detecting offensive language in social-media texts?

The practical goal is to build a reproducible experimental pipeline that includes dataset preparation, preprocessing, model training, evaluation, result visualization, and a scientific report.

---

## 3. Scientific Background and Related Work

Offensive-language detection became a widely studied benchmark problem through the Offensive Language Identification Dataset (OLID) and OffensEval / SemEval 2019 Task 6. Zampieri et al. (2019a, 2019b) introduced a hierarchical annotation scheme for offensive language in tweets. Level A of OLID is a binary task that distinguishes offensive (`OFF`) and not offensive (`NOT`) posts. This is the task used in the present project.

Transformer-based models are now a dominant direction in NLP text classification. BERT introduced deep bidirectional transformer pretraining and showed strong results across many language-understanding tasks (Devlin et al., 2019). Sentence-BERT adapted BERT-style models for sentence-level embeddings (Reimers and Gurevych, 2019). BERTweet was designed specifically for English Twitter text (Nguyen et al., 2020), and HateBERT was adapted for abusive-language detection (Caselli et al., 2021).

Recent shared tasks also show the importance of transformer-based methods. SemEval-2023 Task 10, Explainable Detection of Online Sexism (EDOS), introduced a recent dataset and taxonomy for abusive-language detection in social media (Kirk et al., 2023). Many systems in this task used fine-tuned transformer architectures such as BERT, RoBERTa, DistilBERT, HateBERT, and XLM-T, often combined with task-adaptive pretraining, data augmentation, or ensemble methods (Segura-Bedmar, 2023; Roy and Shrivastava, 2023; Rallabandi et al., 2023; Mahmoudi, 2023).

Recent work also emphasizes multilingual and cross-lingual offensive-language detection. Jiang and Zubiaga (2024) review datasets, transfer-learning approaches, and challenges in cross-lingual offensive-language detection. He et al. (2024) evaluate large language models on multilingual offensive-language detection and discuss translation, prompt language, and dataset bias.

Based on this literature, this project treats TF-IDF models as classical baselines and includes a fine-tuned transformer classifier as the main recent method.

---

## 4. Dataset

The project uses the **Offensive Language Identification Dataset (OLID)** from OffensEval / SemEval 2019 Task 6. The dataset contains English tweets annotated according to a hierarchical taxonomy. This project focuses on **Level A**, the binary offensive-language identification task.

| Label | Meaning |
|---|---|
| `NOT` | Not offensive |
| `OFF` | Offensive |

For ethical and safety reasons, raw offensive examples are not reproduced in this report. The analysis uses aggregate statistics, metrics, and confusion matrices only.

### 4.1 Dataset Statistics

| Split | Total Examples | NOT | OFF | Average Text Length |
|---|---:|---:|---:|---:|
| Training set | 13,240 | 8,840 | 4,400 | 125.91 characters |
| Official test set | 860 | 620 | 240 | 146.16 characters |

The test set is imbalanced: the `NOT` class contains 620 examples and the `OFF` class contains 240 examples. Because of this imbalance, accuracy alone is not sufficient. Macro F1 and offensive-class F1 are especially important.

---

## 5. Methodology

The experimental pipeline follows these steps:

1. Load the OLID training set and official test files.
2. Prepare the official Level A test set by joining test tweets with their labels.
3. Apply preprocessing appropriate to the model family.
4. Train classical TF-IDF baselines.
5. Train transformer-based models.
6. Evaluate all models using the same official test set.
7. Save metrics, confusion matrices, comparison tables, and report-ready plots.

### 5.1 Classical Baseline Models

The classical models use TF-IDF features with unigram and bigram terms. These models are not presented as recent state-of-the-art methods. They are included only as transparent and reproducible baselines.

| Model | Representation | Purpose |
|---|---|---|
| Logistic Regression | TF-IDF | Strong linear text-classification baseline |
| Linear SVM | TF-IDF | Margin-based classifier for sparse high-dimensional text features |
| Complement Naive Bayes | TF-IDF | Probabilistic baseline suitable for imbalanced text data |

For these models, the preprocessing includes lowercasing, URL normalization/removal, user mention normalization, hashtag-symbol removal, removal of non-text symbols, and whitespace normalization.

### 5.2 Transformer-Embedding Model

The first transformer-based method uses `sentence-transformers/all-MiniLM-L6-v2` to create contextual sentence embeddings. These embeddings are then used as dense features for a Logistic Regression classifier. This method is lighter than full fine-tuning and provides a bridge between classical classifiers and transformer representations.

Implementation file:

```text
src/train_transformer_embeddings.py
```

### 5.3 Fine-Tuned Transformer Model

The main modern method is a fine-tuned transformer classifier. The model uses the pre-trained `sentence-transformers/all-MiniLM-L6-v2` checkpoint and adds a sequence-classification head. It is then fine-tuned directly on the OLID training set for the binary `NOT` / `OFF` classification task.

Implementation file:

```text
src/train_transformer_finetune.py
```

This method is aligned with the recent BERT-family fine-tuning approach used in current offensive-language and abusive-language detection research. The model was trained for one epoch with learning rate `2e-5`, batch size `32`, maximum sequence length `128`, and Apple Silicon MPS acceleration.

---

## 6. Evaluation Setup

All models are evaluated on the same official OLID Level A test set. The following metrics are reported:

| Metric | Reason for Use |
|---|---|
| Accuracy | Overall proportion of correct predictions |
| Precision OFF | How many predicted offensive posts are truly offensive |
| Recall OFF | How many real offensive posts are detected |
| F1 OFF | Balance between precision and recall for the offensive class |
| Macro F1 | Average F1 across both classes; important for imbalanced data |
| Confusion Matrix | Detailed view of correct and incorrect predictions |

The main evaluation metric is **Macro F1**, because the dataset is imbalanced. The offensive-class F1-score is also important because the main practical goal is detecting offensive content.

---

## 7. Experimental Results

### 7.1 Overall Model Comparison

| Rank | Model | Accuracy | Precision OFF | Recall OFF | F1 OFF | Macro F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Fine-tuned Transformer | **0.8430** | **0.7692** | 0.6250 | **0.6897** | **0.7923** |
| 2 | Transformer Embeddings + Logistic Regression | 0.7779 | 0.5939 | **0.6458** | 0.6188 | 0.7310 |
| 3 | TF-IDF + Logistic Regression | 0.7791 | 0.6033 | 0.6083 | 0.6058 | 0.7262 |
| 4 | TF-IDF + Complement Naive Bayes | 0.7988 | 0.7134 | 0.4667 | 0.5642 | 0.7167 |
| 5 | TF-IDF + Linear SVM | 0.7744 | 0.6009 | 0.5708 | 0.5855 | 0.7153 |

The fine-tuned transformer is the best overall model. It achieves the highest accuracy, highest offensive-class precision, highest offensive-class F1-score, and highest Macro F1-score.

### 7.2 Best Classical Baseline

Among the TF-IDF models, Logistic Regression is the strongest baseline:

| Model | Accuracy | F1 OFF | Macro F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.7791 | 0.6058 | 0.7262 |

This confirms that classical methods can provide a reasonable baseline, but they are clearly weaker than the fine-tuned transformer.

### 7.3 Fine-Tuned Transformer Confusion Matrix

The fine-tuned transformer produced the following confusion matrix:

| True Label | Predicted NOT | Predicted OFF |
|---|---:|---:|
| NOT | 575 | 45 |
| OFF | 90 | 150 |

Interpretation:

- The model correctly classified **575 out of 620** not-offensive examples.
- The model correctly detected **150 out of 240** offensive examples.
- There were **45 false positives**, where not-offensive texts were incorrectly predicted as offensive.
- There were **90 false negatives**, where offensive texts were missed.

The fine-tuned transformer provides the strongest overall balance. It reduces false positives compared with the transformer-embedding model while maintaining much stronger performance than the TF-IDF baselines.

### 7.4 Report Assets

The project generates report-ready visual assets:

```text
reports/assets/model_comparison_accuracy.png
reports/assets/model_comparison_macro_f1.png
reports/assets/model_comparison_offensive_f1.png
reports/assets/finetuned_minilm_transformer_confusion_matrix.png
```

The final comparison table is stored in:

```text
reports/model_comparison_official_test.csv
```

---

## 8. Discussion

The results support the main expectation from recent NLP research: transformer-based models outperform classical sparse-feature baselines. The fine-tuned transformer improves Macro F1 from `0.7262` for the best TF-IDF baseline to `0.7923`. It also improves offensive-class F1 from `0.6058` to `0.6897`.

The transformer-embedding model also slightly improves over the best TF-IDF baseline in Macro F1 and offensive-class F1, showing that contextual transformer representations are useful even without full fine-tuning. However, full fine-tuning gives the best result because the transformer encoder and classification head are adapted directly to the OLID offensive-language detection task.

The classical models remain useful for comparison. They are faster, easier to interpret, and simpler to reproduce. However, they rely on surface-level lexical features and cannot capture deeper context, implicit offensiveness, sarcasm, or nuanced social-media language as effectively as transformer-based models.

---

## 9. Ethical Considerations

Offensive-language detection is a sensitive task. Automated models can make mistakes and should not be used as the only decision-making mechanism. False positives may unfairly flag harmless speech. False negatives may allow harmful content to remain undetected.

This project follows several ethical precautions:

- Raw offensive examples are not displayed in the report.
- The system is presented as support for human moderation, not as a replacement for human judgment.
- Evaluation includes class-specific metrics, not only accuracy.
- Limitations and risks are explicitly discussed.

---

## 10. Limitations

The project has several limitations:

1. The dataset is limited to English tweets from OLID.
2. The model is evaluated only on Level A binary classification.
3. The fine-tuned transformer uses `all-MiniLM-L6-v2`, not a tweet-specific model such as BERTweet or a hate-speech-specific model such as HateBERT.
4. The experiment does not include detailed error analysis of false positives and false negatives.
5. The project does not evaluate multilingual transfer or Hebrew offensive-language detection.

---

## 11. Future Work

Future improvements could include:

- Fine-tuning BERTweet, HateBERT, RoBERTa, DistilBERT, or XLM-R on the same OLID split.
- Testing recent datasets such as EDOS for explainable abusive-language detection.
- Adding error analysis for false positives and false negatives.
- Testing explainability methods for model decisions.
- Extending the project to multilingual or cross-lingual offensive-language detection.
- Comparing transformer models with large language model prompting approaches.

---

## 12. Conclusion

This project implements a complete offensive-language detection pipeline and revises the methodology to match recent scientific work. TF-IDF models are kept only as classical baselines. A transformer-embedding model and a fine-tuned transformer classifier are added for modern comparison.

The best overall model is the **Fine-tuned Transformer**, which achieves:

| Metric | Score |
|---|---:|
| Accuracy | 0.8430 |
| Precision OFF | 0.7692 |
| Recall OFF | 0.6250 |
| F1 OFF | 0.6897 |
| Macro F1 | 0.7923 |

The best classical baseline is **TF-IDF + Logistic Regression**, with Macro F1 `0.7262`. Therefore, the fine-tuned transformer substantially improves the project and directly addresses the requirement to include recent scientific methods and references.

Overall, the revised work is no longer only a classical-machine-learning project. It is a comparative offensive-language detection study grounded in recent literature and supported by reproducible experiments.

---

## References

1. Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., & Kumar, R. (2019a). *SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval).* Proceedings of SemEval 2019.
2. Zampieri, M., Rosenthal, S., Nakov, P., & Potthast, M. (2019b). *Predicting the Type and Target of Offensive Posts in Social Media.* NAACL-HLT.
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.* NAACL-HLT.
4. Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP-IJCNLP.
5. Nguyen, D. Q., Vu, T., & Nguyen, A. T. (2020). *BERTweet: A pre-trained language model for English Tweets.* EMNLP System Demonstrations.
6. Caselli, T., Basile, V., Mitrovic, J., & Granitzer, M. (2021). *HateBERT: Retraining BERT for Abusive Language Detection in English.* WOAH.
7. Kirk, H. R., Yin, W., Vidgen, B., & Röttger, P. (2023). *SemEval-2023 Task 10: Explainable Detection of Online Sexism.* arXiv:2303.04222. https://arxiv.org/abs/2303.04222
8. Segura-Bedmar, I. (2023). *HULAT at SemEval-2023 Task 10: Data augmentation for pre-trained transformers applied to the detection of sexism in social media.* arXiv:2302.12840. https://arxiv.org/abs/2302.12840
9. Roy, D., & Shrivastava, M. (2023). *Attention at SemEval-2023 Task 10: Explainable Detection of Online Sexism (EDOS).* arXiv:2304.04610. https://arxiv.org/abs/2304.04610
10. Rallabandi, S., Singhal, S., & Seth, P. (2023). *SSS at SemEval-2023 Task 10: Explainable Detection of Online Sexism using Majority Voted Fine-Tuned Transformers.* arXiv:2304.03518. https://arxiv.org/abs/2304.03518
11. Mahmoudi, H. (2023). *IUST_NLP at SemEval-2023 Task 10: Explainable Detecting Sexism with Transformers and Task-adaptive Pretraining.* arXiv:2305.06892. https://arxiv.org/abs/2305.06892
12. Jiang, A., & Zubiaga, A. (2024). *Cross-lingual Offensive Language Detection: A Systematic Review of Datasets, Transfer Approaches and Challenges.* arXiv:2401.09244. https://arxiv.org/abs/2401.09244
13. He, J., Wang, L., Wang, J., Liu, Z., Na, H., Wang, Z., Wang, W., & Chen, Q. (2024). *Guardians of Discourse: Evaluating LLMs on Multilingual Offensive Language Detection.* arXiv:2410.15623. https://arxiv.org/abs/2410.15623
