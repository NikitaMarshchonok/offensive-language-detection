# Обнаружение оскорбительной речи в текстах социальных сетей

## Сравнительное исследование классических TF-IDF baseline-моделей и дообученных transformer-моделей

**Студент:** Nikita Marshchonok  
**Степень:** M.Sc. Software Engineering  
**Датасет:** OLID / OffensEval 2019, Level A  

---

## Аннотация

Этот проект посвящен автоматическому обнаружению оскорбительной речи (`offensive language`) в коротких текстах из социальных сетей. Задача формулируется как бинарная классификация: определить, является ли сообщение оскорбительным (`OFF`) или не оскорбительным (`NOT`). В пересмотренной версии проект явно опирается на современные научные работы в области offensive language detection, abusive language detection и hate speech detection. Классические TF-IDF модели используются только как воспроизводимые baseline-методы, а основной современный подход основан на дообученной transformer-модели.

В проекте используется датасет OLID из OffensEval / SemEval 2019. Все модели оцениваются на официальном тестовом наборе Level A. Сравниваются пять моделей: три классические TF-IDF baseline-модели, модель на transformer embeddings и дообученный transformer classifier. Лучший результат показывает дообученная transformer-модель: **Accuracy = 0.8430**, **Macro F1 = 0.7923**, **F1 для offensive class = 0.6897**. Это заметно выше результата лучшей классической baseline-модели, TF-IDF + Logistic Regression, у которой **Macro F1 = 0.7262**.

**Ключевые слова:** offensive language detection, hate speech detection, text classification, OLID, TF-IDF, transformer, fine-tuning, BERT, social media.

---

## 1. Введение

Социальные сети содержат огромное количество пользовательских текстов. Часть сообщений может включать оскорбительную, агрессивную или вредную лексику. Ручная модерация таких сообщений дорогая, эмоционально сложная и плохо масштабируется. Автоматическое обнаружение оскорбительной речи может помогать модераторам, выделяя потенциально проблемные сообщения для дальнейшей проверки человеком.

Ранние системы текстовой классификации часто использовали разреженные лексические представления, например bag-of-words или TF-IDF, вместе с такими классификаторами, как Logistic Regression, Support Vector Machines или Naive Bayes. Эти методы полезны, потому что они простые, быстрые, воспроизводимые и относительно интерпретируемые. Однако они уже не считаются самым современным подходом для offensive language detection. Современные исследования в NLP в основном используют transformer-based модели: BERT, RoBERTa, BERTweet, HateBERT, XLM-R и большие языковые модели.

Главная цель пересмотренной версии проекта — сравнить старые классические baseline-модели с современным transformer-based подходом и связать проект с релевантной научной литературой.

---

## 2. Исследовательский вопрос

В проекте рассматривается следующий исследовательский вопрос:

> Насколько эффективны классические TF-IDF baseline-модели по сравнению с transformer-based моделями для обнаружения оскорбительной речи в текстах социальных сетей?

Практическая цель проекта — построить воспроизводимый экспериментальный pipeline, который включает подготовку датасета, preprocessing, обучение моделей, оценку качества, визуализацию результатов и научный отчет.

---

## 3. Научная основа и связанные работы

Задача offensive language detection стала широко изучаться благодаря датасету Offensive Language Identification Dataset (OLID) и соревнованию OffensEval / SemEval 2019 Task 6. Zampieri et al. (2019a, 2019b) предложили иерархическую схему аннотации оскорбительной речи в твитах. Level A в OLID — это бинарная задача, где нужно отличить offensive (`OFF`) сообщения от not offensive (`NOT`) сообщений. Именно эта задача используется в проекте.

Transformer-based модели сейчас являются одним из основных направлений в NLP text classification. BERT предложил глубокое двунаправленное transformer-предобучение и показал сильные результаты на задачах понимания языка (Devlin et al., 2019). Sentence-BERT адаптировал BERT-подобные модели для получения sentence embeddings (Reimers and Gurevych, 2019). BERTweet был разработан специально для английских Twitter-текстов (Nguyen et al., 2020), а HateBERT был адаптирован для abusive language detection (Caselli et al., 2021).

Современные shared tasks также показывают важность transformer-based методов. SemEval-2023 Task 10, Explainable Detection of Online Sexism (EDOS), предложил современный датасет и таксономию для abusive language detection в социальных сетях (Kirk et al., 2023). Многие системы в этой задаче использовали fine-tuned transformer architectures, включая BERT, RoBERTa, DistilBERT, HateBERT и XLM-T, часто вместе с task-adaptive pretraining, data augmentation или ensemble methods (Segura-Bedmar, 2023; Roy and Shrivastava, 2023; Rallabandi et al., 2023; Mahmoudi, 2023).

Современные работы также уделяют внимание multilingual и cross-lingual offensive language detection. Jiang and Zubiaga (2024) рассматривают датасеты, transfer-learning подходы и сложности cross-lingual offensive language detection. He et al. (2024) оценивают большие языковые модели на multilingual offensive language detection и обсуждают влияние перевода, языка prompt и dataset bias.

На основе этой литературы в проекте TF-IDF модели рассматриваются только как классические baseline-модели, а главным современным методом является fine-tuned transformer classifier.

---

## 4. Датасет

Проект использует **Offensive Language Identification Dataset (OLID)** из OffensEval / SemEval 2019 Task 6. Датасет содержит английские твиты, размеченные по иерархической таксономии. В проекте используется **Level A** — бинарная задача offensive-language identification.

| Метка | Значение |
|---|---|
| `NOT` | Не оскорбительный текст |
| `OFF` | Оскорбительный текст |

По этическим причинам в отчете не приводятся реальные offensive-примеры из датасета. Анализ строится только на агрегированных статистиках, метриках и confusion matrix.

### 4.1 Статистика датасета

| Split | Всего примеров | NOT | OFF | Средняя длина текста |
|---|---:|---:|---:|---:|
| Training set | 13,240 | 8,840 | 4,400 | 125.91 символов |
| Official test set | 860 | 620 | 240 | 146.16 символов |

Тестовый набор несбалансирован: класс `NOT` содержит 620 примеров, а класс `OFF` — 240 примеров. Из-за этого одной accuracy недостаточно. Важны Macro F1 и F1-score для offensive class.

---

## 5. Методология

Экспериментальный pipeline состоит из следующих шагов:

1. Загрузка OLID training set и official test files.
2. Подготовка official Level A test set путем объединения test tweets и labels.
3. Применение preprocessing в зависимости от типа модели.
4. Обучение классических TF-IDF baseline-моделей.
5. Обучение transformer-based моделей.
6. Оценка всех моделей на одном и том же official test set.
7. Сохранение метрик, confusion matrices, сравнительных таблиц и графиков для отчета.

### 5.1 Классические baseline-модели

Классические модели используют TF-IDF features с unigram и bigram признаками. Эти модели не представлены как современные state-of-the-art методы. Они используются только как прозрачные и воспроизводимые baseline-модели.

| Модель | Представление текста | Назначение |
|---|---|---|
| Logistic Regression | TF-IDF | Сильный линейный baseline для text classification |
| Linear SVM | TF-IDF | Margin-based classifier для разреженных высокоразмерных текстовых признаков |
| Complement Naive Bayes | TF-IDF | Вероятностный baseline, подходящий для несбалансированных текстовых данных |

Для этих моделей preprocessing включает lowercasing, нормализацию/удаление URL, нормализацию user mentions, удаление символа hashtag, удаление нетекстовых символов и нормализацию пробелов.

### 5.2 Модель на transformer embeddings

Первый transformer-based метод использует `sentence-transformers/all-MiniLM-L6-v2` для получения contextual sentence embeddings. Затем эти embeddings используются как dense features для Logistic Regression classifier. Этот метод легче, чем full fine-tuning, и является промежуточным вариантом между классическими классификаторами и transformer representations.

Файл реализации:

```text
src/train_transformer_embeddings.py
```

### 5.3 Fine-tuned transformer model

Главный современный метод — это fine-tuned transformer classifier. Модель использует pre-trained checkpoint `sentence-transformers/all-MiniLM-L6-v2`, к которому добавляется sequence-classification head. После этого модель дообучается напрямую на OLID training set для бинарной классификации `NOT` / `OFF`.

Файл реализации:

```text
src/train_transformer_finetune.py
```

Этот метод соответствует современному BERT-family fine-tuning подходу, который часто используется в исследованиях offensive language detection и abusive language detection. Модель обучалась 1 epoch с learning rate `2e-5`, batch size `32`, maximum sequence length `128` и Apple Silicon MPS acceleration.

---

## 6. Оценка качества

Все модели оцениваются на одном и том же official OLID Level A test set. Используются следующие метрики:

| Метрика | Зачем используется |
|---|---|
| Accuracy | Общая доля правильных предсказаний |
| Precision OFF | Сколько predicted offensive сообщений действительно offensive |
| Recall OFF | Сколько реальных offensive сообщений модель нашла |
| F1 OFF | Баланс precision и recall для offensive class |
| Macro F1 | Средний F1 по обоим классам; важен при class imbalance |
| Confusion Matrix | Детальный разбор правильных и ошибочных предсказаний |

Главная метрика — **Macro F1**, потому что датасет несбалансирован. F1-score для offensive class также важен, потому что практическая цель модели — находить offensive content.

---

## 7. Экспериментальные результаты

### 7.1 Общее сравнение моделей

| Rank | Model | Accuracy | Precision OFF | Recall OFF | F1 OFF | Macro F1 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Fine-tuned Transformer | **0.8430** | **0.7692** | 0.6250 | **0.6897** | **0.7923** |
| 2 | Transformer Embeddings + Logistic Regression | 0.7779 | 0.5939 | **0.6458** | 0.6188 | 0.7310 |
| 3 | TF-IDF + Logistic Regression | 0.7791 | 0.6033 | 0.6083 | 0.6058 | 0.7262 |
| 4 | TF-IDF + Complement Naive Bayes | 0.7988 | 0.7134 | 0.4667 | 0.5642 | 0.7167 |
| 5 | TF-IDF + Linear SVM | 0.7744 | 0.6009 | 0.5708 | 0.5855 | 0.7153 |

Fine-tuned Transformer является лучшей моделью в проекте. Она показывает наибольшую accuracy, наибольший precision для offensive class, наибольший F1 для offensive class и наибольший Macro F1.

### 7.2 Лучшая классическая baseline-модель

Среди TF-IDF моделей лучшим baseline является Logistic Regression:

| Model | Accuracy | F1 OFF | Macro F1 |
|---|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.7791 | 0.6058 | 0.7262 |

Это подтверждает, что классические методы дают разумный baseline, но уступают fine-tuned transformer.

### 7.3 Confusion Matrix для Fine-tuned Transformer

Fine-tuned Transformer дал следующую confusion matrix:

| True Label | Predicted NOT | Predicted OFF |
|---|---:|---:|
| NOT | 575 | 45 |
| OFF | 90 | 150 |

Интерпретация:

- Модель правильно классифицировала **575 из 620** not-offensive примеров.
- Модель правильно нашла **150 из 240** offensive примеров.
- Было **45 false positives**, когда not-offensive тексты были ошибочно предсказаны как offensive.
- Было **90 false negatives**, когда offensive тексты были пропущены.

Fine-tuned Transformer показывает лучший общий баланс. Он уменьшает количество false positives по сравнению с transformer-embedding моделью и значительно превосходит TF-IDF baselines.

### 7.4 Файлы с результатами

Проект генерирует графики и таблицы для отчета:

```text
reports/assets/model_comparison_accuracy.png
reports/assets/model_comparison_macro_f1.png
reports/assets/model_comparison_offensive_f1.png
reports/assets/finetuned_minilm_transformer_confusion_matrix.png
```

Итоговая сравнительная таблица находится здесь:

```text
reports/model_comparison_official_test.csv
```

---

## 8. Обсуждение

Результаты подтверждают основное ожидание из современных NLP-исследований: transformer-based модели превосходят классические sparse-feature baseline-модели. Fine-tuned Transformer улучшает Macro F1 с `0.7262` у лучшего TF-IDF baseline до `0.7923`. Также F1-score для offensive class увеличивается с `0.6058` до `0.6897`.

Модель на transformer embeddings также немного улучшает результат по сравнению с лучшим TF-IDF baseline по Macro F1 и offensive-class F1. Это показывает, что contextual transformer representations полезны даже без полного fine-tuning. Однако полный fine-tuning дает лучший результат, потому что transformer encoder и classification head адаптируются непосредственно к задаче OLID offensive-language detection.

Классические модели остаются полезными для сравнения. Они быстрее, проще для интерпретации и легче воспроизводятся. Однако они опираются на поверхностные лексические признаки и хуже улавливают контекст, скрытую оскорбительность, сарказм и нюансы языка социальных сетей.

---

## 9. Этические аспекты

Offensive-language detection — чувствительная задача. Автоматические модели могут ошибаться и не должны быть единственным механизмом принятия решений. False positives могут несправедливо помечать нормальную речь как offensive. False negatives могут пропускать вредный контент.

В проекте соблюдаются следующие меры:

- Реальные offensive-примеры не показываются в отчете.
- Система рассматривается как инструмент поддержки human moderation, а не как замена человеку.
- Оценка включает class-specific metrics, а не только accuracy.
- Ограничения и риски явно описаны.

---

## 10. Ограничения

У проекта есть несколько ограничений:

1. Датасет ограничен английскими твитами из OLID.
2. Модель оценивается только на Level A binary classification.
3. Fine-tuned transformer использует `all-MiniLM-L6-v2`, а не tweet-specific модель вроде BERTweet и не hate-speech-specific модель вроде HateBERT.
4. В проекте нет подробного error analysis для false positives и false negatives.
5. Проект не оценивает multilingual transfer или Hebrew offensive-language detection.

---

## 11. Будущая работа

Возможные улучшения:

- Fine-tuning BERTweet, HateBERT, RoBERTa, DistilBERT или XLM-R на том же OLID split.
- Проверка современных датасетов, например EDOS, для explainable abusive-language detection.
- Error analysis для false positives и false negatives.
- Использование explainability methods для анализа решений модели.
- Расширение проекта на multilingual или cross-lingual offensive-language detection.
- Сравнение transformer-моделей с prompting-подходами на основе large language models.

---

## 12. Заключение

Проект реализует полный pipeline для offensive-language detection и приводит методологию в соответствие с современными научными работами. TF-IDF модели используются только как классические baseline-модели. Для современного сравнения добавлены transformer-embedding model и fine-tuned transformer classifier.

Лучшая модель — **Fine-tuned Transformer**, которая достигает следующих результатов:

| Метрика | Значение |
|---|---:|
| Accuracy | 0.8430 |
| Precision OFF | 0.7692 |
| Recall OFF | 0.6250 |
| F1 OFF | 0.6897 |
| Macro F1 | 0.7923 |

Лучшая классическая baseline-модель — **TF-IDF + Logistic Regression** с Macro F1 `0.7262`. Следовательно, fine-tuned transformer существенно улучшает проект и напрямую отвечает требованию использовать современные научные методы и references.

Итог: пересмотренная работа больше не является проектом только на классических machine-learning методах. Это сравнительное исследование offensive-language detection, основанное на современной литературе и подтвержденное воспроизводимыми экспериментами.

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
