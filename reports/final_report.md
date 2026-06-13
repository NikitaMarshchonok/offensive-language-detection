# Offensive Language Detection in Social Media Texts: Classical Baselines and a Transformer-Based Approach

**Student:** Nikita Marshchonok  
**Degree:** M.Sc. Software Engineering  
**Course:** Advanced Techniques in Information Retrieval  
**Project type:** Practical programming project  

---

## Abstract

This project investigates automatic offensive language detection in short social-media texts. The task is binary classification: identifying whether a post is offensive (`OFF`) or not offensive (`NOT`). In the revised version, TF-IDF based machine-learning models are treated only as classical baselines, while the project adds a recent fine-tuned transformer-based method. The implemented pipeline includes dataset preparation, preprocessing, baseline training, transformer fine-tuning, evaluation, and report-ready result generation.

---

## 1. Introduction

Social-media platforms contain large amounts of user-generated text. Some posts include offensive, abusive, or harmful language, and manual moderation is expensive and difficult to scale. Automatic offensive-language detection can support human moderation by flagging potentially harmful content for review.

Earlier text-classification systems often relied on sparse representations such as bag-of-words or TF-IDF combined with classifiers such as Logistic Regression, Support Vector Machines, or Naive Bayes. These methods are still useful as reproducible baselines, but recent NLP research has moved toward transformer-based models such as BERT, RoBERTa, BERTweet, HateBERT, XLM-R, and large language models. This project therefore compares classical baselines with a modern transformer-based approach.

---

## 2. Research Question

**How effective are classical TF-IDF machine-learning baselines for offensive language detection, and how can a recent transformer-based approach be integrated into the same experimental pipeline for comparison?**

---

## 3. Related Work and Scientific Background

Offensive-language detection became a widely studied benchmark task through OffensEval and the Offensive Language Identification Dataset (OLID). Zampieri et al. (2019a, 2019b) introduced a hierarchical annotation scheme for offensive language in tweets, including Level A binary offensive-language identification, which is the task used in this project.

Recent work shows that transformer-based models are now the dominant approach for abusive, offensive, and hateful language detection. The SemEval-2023 EDOS shared task introduced a recent dataset and taxonomy for explainable detection of online sexism, and its overview reports strong performance from transformer-based systems (Kirk et al., 2023). Several participating systems fine-tuned BERT-family models such as BERT, RoBERTa, DistilBERT, XLNet, HateBERT, and XLM-T, often improving performance through domain adaptation, task-adaptive pretraining, data augmentation, or model ensembling (Segura-Bedmar, 2023; Roy and Shrivastava, 2023; Rallabandi et al., 2023; Mahmoudi, 2023).

The importance of multilingual and cross-lingual offensive-language detection has also increased. Jiang and Zubiaga (2024) provide a systematic review of cross-lingual offensive-language detection, emphasizing transfer learning and multilingual resources. He et al. (2024) evaluate large language models for multilingual offensive-language detection and discuss the role of prompt language, translation, and dataset bias.

Based on this literature, the present project keeps TF-IDF models as classical baselines and adds a transformer-based method as the recent approach. This aligns the practical implementation with current scientific work rather than presenting older methods as the main state of the art.

---

## 4. Dataset

The project uses the **Offensive Language Identification Dataset (OLID)** from OffensEval / SemEval 2019 Task 6. OLID contains English tweets annotated according to a hierarchical three-level taxonomy. This project focuses on **Sub-task A**, the binary classification task:

| Label | Meaning |
|---|---|
| `NOT` | Not offensive |
| `OFF` | Offensive |

The official training set contains 13,240 examples. The official Level A test set used in this project contains 860 examples: 620 `NOT` and 240 `OFF`. For ethical and safety reasons, the report does not reproduce raw offensive examples.

---

## 5. Methodology

The project implements two groups of methods.

### 5.1 Classical Baselines

The classical baseline pipeline uses text cleaning, TF-IDF vectorization with unigram and bigram features, and supervised classifiers. These models are included because they are efficient, interpretable, and historically strong for short-text classification. In this revised project, they are explicitly treated as baselines rather than as the main state-of-the-art contribution.

| Model | Representation | Role |
|---|---|---|
| Logistic Regression | TF-IDF | Strong linear baseline |
| Linear SVM | TF-IDF | Margin-based sparse-text baseline |
| Complement Naive Bayes | TF-IDF | Probabilistic baseline for imbalanced text data |

### 5.2 Transformer-Based Approach

The main modern approach is implemented in `src/train_transformer_finetune.py`. It fine-tunes a pre-trained transformer encoder, `sentence-transformers/all-MiniLM-L6-v2`, as a sequence-classification model directly on the OLID training set. This is the closest implemented method to the recent BERT-family fine-tuning approach commonly used in current offensive-language detection research.

The project also includes a lighter transformer-embedding variant in `src/train_transformer_embeddings.py`. That script uses the same pre-trained transformer encoder to create contextual sentence embeddings and trains Logistic Regression on top of them. It is useful as a faster transformer baseline, while the fine-tuned transformer is the main recent method.

The older `src/train_transformer.py` script is kept as an optional Hugging Face `Trainer`-based path for DistilBERT-style models, but the final verified transformer result in this report comes from the custom fine-tuning script.

---

## 6. Preprocessing

For the TF-IDF baselines, the following preprocessing steps are applied:

1. Lowercasing
2. URL normalization/removal
3. User mention normalization
4. Hashtag sign removal
5. Removal of non-text symbols
6. Extra whitespace normalization

For transformer-based models, preprocessing is intentionally minimal because transformer tokenizers are designed to preserve subword and contextual information.

---

## 7. Evaluation Methodology

The models are evaluated using accuracy, precision for the offensive class, recall for the offensive class, F1-score for the offensive class, Macro F1-score, and confusion matrices. The main metric is **Macro F1-score**, because offensive-language datasets are imbalanced and accuracy alone may hide poor performance on the minority offensive class.

---

## 8. Experimental Results

The verified local experiment was run on the official OLID Level A test set. The results are shown below.

| Model | Accuracy | Precision OFF | Recall OFF | F1 OFF | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Fine-tuned Transformer | 0.8430 | 0.7692 | 0.6250 | 0.6897 | 0.7923 |
| Transformer Embeddings + Logistic Regression | 0.7779 | 0.5939 | 0.6458 | 0.6188 | 0.7310 |
| TF-IDF + Logistic Regression | 0.7791 | 0.6033 | 0.6083 | 0.6058 | 0.7262 |
| TF-IDF + Complement Naive Bayes | 0.7988 | 0.7134 | 0.4667 | 0.5642 | 0.7167 |
| TF-IDF + Linear SVM | 0.7744 | 0.6009 | 0.5708 | 0.5855 | 0.7153 |

The fine-tuned transformer achieved the best overall performance across the most important metrics: accuracy, offensive-class F1-score, and Macro F1-score. Among the classical baselines, **TF-IDF + Logistic Regression** is the strongest baseline.

The fine-tuned transformer experiment can be executed with:

```bash
python -m src.train_transformer_finetune \
  --train_path data/raw/olid-training-v1.0.tsv \
  --test_path data/processed/olid-test-levela-labeled.tsv \
  --text_col tweet \
  --label_col subtask_a \
  --output_dir outputs_final
```

The fine-tuned transformer confusion matrix shows 150 correctly detected offensive examples out of 240, with only 45 false positives for the `NOT` class. This model provides the strongest balance between detecting offensive content and avoiding excessive false positives.

---

## 9. Discussion

The results show that classical TF-IDF models provide a reasonable baseline for offensive-language detection, but they remain limited. Sparse TF-IDF features do not model deep context, implicit offensiveness, sarcasm, pragmatic meaning, or domain-specific usage. These limitations are exactly why recent research has moved toward transformer-based approaches.

The revised project therefore positions the classical models correctly: they are not presented as recent state of the art, but as transparent baselines. The fine-tuned transformer gives the project a modern method aligned with recent scientific papers and substantially improves Macro F1-score and offensive-class F1-score over the strongest TF-IDF baseline.

The confusion matrix for the fine-tuned transformer shows that it correctly detects 150 out of 240 offensive examples and reduces false positives compared with the transformer-embedding model. Offensive-class recall remains a key challenge, but the fine-tuned transformer is clearly stronger than the classical baselines overall.

---

## 10. Ethical Considerations

Offensive-language detection is a sensitive task. Automated systems can make mistakes and should not be used as the only decision-making mechanism. False positives may unfairly flag normal speech, while false negatives may miss harmful content. The system should support human moderation rather than fully replace it.

The project avoids displaying raw offensive examples and focuses on aggregate metrics, model behavior, and responsible evaluation.

---

## 11. Limitations and Future Work

The main limitation of the transformer experiment is that it fine-tunes `all-MiniLM-L6-v2` rather than a tweet-specific model such as BERTweet or a hate-speech-specific model such as HateBERT. Future work should fine-tune BERTweet, HateBERT, RoBERTa, or XLM-R and compare them on the same OLID test set.

Additional future improvements include fine-tuning DistilBERT, BERTweet, HateBERT, or RoBERTa on OLID; testing cross-lingual models such as XLM-R; using recent datasets such as EDOS; adding error analysis; testing explainability methods; and extending the system to multilingual settings, for example Hebrew offensive-language detection.

---

## 12. Conclusion

This project implements an offensive-language detection pipeline and revises the methodology to align with recent scientific work. The TF-IDF models are now clearly presented as classical baselines, while the project includes and evaluates a fine-tuned transformer classifier. The best overall model is the Fine-tuned Transformer, with a Macro F1-score of 0.7923 and offensive-class F1-score of 0.6897 on the official OLID Level A test set. The best classical baseline is TF-IDF + Logistic Regression, with a Macro F1-score of 0.7262 and offensive-class F1-score of 0.6058.

The revised project therefore addresses the main feedback: it is now explicitly based on recent offensive-language detection research and includes a modern transformer-based method in addition to the older baseline models.

---

## References

1. Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., & Kumar, R. (2019a). *SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval).* Proceedings of SemEval 2019.
2. Zampieri, M., et al. (2019b). *Predicting the Type and Target of Offensive Posts in Social Media.* NAACL-HLT.
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
