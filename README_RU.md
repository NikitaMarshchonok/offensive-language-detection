# Обнаружение оскорбительной речи в текстах социальных сетей

## Краткое описание проекта

Этот проект посвящен автоматическому обнаружению оскорбительной речи (`offensive language`) в коротких текстах из социальных сетей. Задача решается как бинарная классификация:

- `NOT` — текст не является оскорбительным;
- `OFF` — текст является оскорбительным.

Проект был пересмотрен так, чтобы старые методы машинного обучения использовались только как **базовые модели**, а основным современным подходом стал **fine-tuned transformer classifier**.

---

## Главная идея

В проекте сравниваются:

1. Классические TF-IDF модели как baseline.
2. Transformer embeddings + Logistic Regression.
3. Fine-tuned Transformer как основной современный метод.

Главный вывод:

> Fine-tuned Transformer показывает лучший результат и превосходит классические TF-IDF baseline-модели.

---

## Научная основа

Проект опирается на современные работы по offensive language detection, hate speech detection и abusive language detection.

В отчете `reports/final_report.md` есть раздел:

```text
Scientific Background and Related Work
```

В нем описаны:

- OLID / OffensEval dataset;
- BERT;
- Sentence-BERT;
- BERTweet;
- HateBERT;
- SemEval-2023 EDOS;
- современные работы 2023-2024 по offensive language detection.

---

## Датасет

Используется датасет **OLID** из OffensEval / SemEval 2019.

В проекте используется Level A — бинарная классификация:

| Метка | Значение |
|---|---|
| `NOT` | Не оскорбительный текст |
| `OFF` | Оскорбительный текст |

Статистика:

| Набор данных | Количество | NOT | OFF |
|---|---:|---:|---:|
| Train | 13,240 | 8,840 | 4,400 |
| Test | 860 | 620 | 240 |

Так как классы несбалансированы, важны не только `accuracy`, но и:

- `Macro F1`;
- `F1` для класса `OFF`;
- `Precision OFF`;
- `Recall OFF`.

---

## Реализованные модели

### 1. Классические baseline-модели

Эти модели оставлены только как базовые методы для сравнения:

| Модель | Представление текста |
|---|---|
| Logistic Regression | TF-IDF |
| Linear SVM | TF-IDF |
| Complement Naive Bayes | TF-IDF |

Они быстрые и воспроизводимые, но считаются устаревшими по сравнению с transformer-based подходами.

### 2. Transformer Embeddings

Файл:

```text
src/train_transformer_embeddings.py
```

Используется модель:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Она создает contextual sentence embeddings, после чего сверху обучается Logistic Regression.

### 3. Fine-tuned Transformer

Файл:

```text
src/train_transformer_finetune.py
```

Это главный современный метод проекта.

Используется pre-trained transformer encoder:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Затем он дообучается на OLID training set как sequence-classification model.

---

## Финальные результаты

Файл с таблицей результатов:

```text
reports/model_comparison_official_test.csv
```

Основная таблица:

| Модель | Accuracy | Precision OFF | Recall OFF | F1 OFF | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Fine-tuned Transformer | **0.8430** | **0.7692** | 0.6250 | **0.6897** | **0.7923** |
| Transformer Embeddings + Logistic Regression | 0.7779 | 0.5939 | **0.6458** | 0.6188 | 0.7310 |
| TF-IDF + Logistic Regression | 0.7791 | 0.6033 | 0.6083 | 0.6058 | 0.7262 |
| TF-IDF + Complement Naive Bayes | 0.7988 | 0.7134 | 0.4667 | 0.5642 | 0.7167 |
| TF-IDF + Linear SVM | 0.7744 | 0.6009 | 0.5708 | 0.5855 | 0.7153 |

Лучший результат:

```text
Fine-tuned Transformer
Accuracy = 0.8430
Macro F1 = 0.7923
F1 OFF = 0.6897
```

Лучший классический baseline:

```text
TF-IDF + Logistic Regression
Macro F1 = 0.7262
F1 OFF = 0.6058
```

Вывод:

> Fine-tuned Transformer значительно улучшает результат по сравнению с лучшей TF-IDF baseline-моделью.

---

## Как запустить проект

Перейти в папку проекта:

```bash
cd /Users/nikitamarshchonok/Downloads/offensive-language-detection
```

Активировать окружение:

```bash
source .venv/bin/activate
```

Запустить только классические baseline-модели:

```bash
python -m src.run_final_experiment
```

Запустить transformer embeddings:

```bash
python -m src.run_final_experiment --include_transformer
```

Запустить fine-tuned transformer:

```bash
python -m src.run_final_experiment --include_finetuning
```

Запустить все модели:

```bash
python -m src.run_final_experiment --include_transformer --include_finetuning
```

Сгенерировать графики и таблицы для отчета:

```bash
python -m src.plot_final_results
```

---

## Что показывать на защите

Открыть эти файлы:

```text
README_RU.md
reports/final_report.md
reports/model_comparison_official_test.csv
src/run_final_experiment.py
src/train_transformer_finetune.py
```

Главный отчет:

```text
reports/final_report.md
```

Графики:

```text
reports/assets/
```

---

## Что говорить на защите

Кратко:

> Проект решает задачу offensive language detection в текстах социальных сетей. Сначала были реализованы классические TF-IDF модели, но после пересмотра они оставлены только как baseline. Для современного сравнения я добавил transformer-based подходы, включая fine-tuned transformer classifier. Лучший результат показал fine-tuned transformer с Macro F1 = 0.7923, что выше лучшего TF-IDF baseline с Macro F1 = 0.7262.

Если спросят, почему TF-IDF остался:

> TF-IDF модели оставлены только как baseline, чтобы сравнить старые классические методы с современными transformer-based моделями.

Если спросят, почему важен Macro F1:

> Датасет несбалансирован: класса `NOT` больше, чем `OFF`. Поэтому accuracy может быть недостаточной метрикой, и важно смотреть на Macro F1 и F1 для offensive class.

Если спросят, что главное в результате:

> Fine-tuned transformer оказался лучшей моделью и подтвердил, что современный transformer-based подход лучше classical TF-IDF baseline для этой задачи.

---

## Основной вывод

Проект теперь является сравнительным исследованием:

- старые TF-IDF методы используются как baseline;
- добавлены современные transformer-based методы;
- fine-tuned transformer показывает лучший результат;
- отчет содержит scientific background, references, results и discussion.

Итог:

> Проект соответствует требованиям пересмотра: он основан на научных работах, содержит описание методов с references и включает современный transformer-based подход.
