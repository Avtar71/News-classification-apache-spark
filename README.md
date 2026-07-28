# 📰 News Classification using Apache Spark MLlib

A scalable Natural Language Processing (NLP) pipeline built using Apache Spark MLlib to classify news articles into four categories:

- 🌍 World
- ⚽ Sports
- 💼 Business
- 🔬 Sci/Tech

The project demonstrates distributed data processing, feature engineering, machine learning model comparison, and hyperparameter tuning on the AG News dataset.

---

## Dataset

- Dataset: AG News Corpus
- Training Samples: 120,000
- Testing Samples: 7,600
- Categories: 4

---

## Tech Stack

- Python
- Apache Spark (PySpark)
- Spark MLlib
- TF-IDF
- HashingTF
- Logistic Regression
- Naive Bayes
- Random Forest
- Cross Validation

---

## Workflow

Data Loading

↓

Text Cleaning

↓

Tokenization

↓

Stopword Removal

↓

HashingTF

↓

TF-IDF

↓

Model Training

↓

Evaluation

↓

Prediction

---

## Results

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 88.89% | 88.87% |
| Naive Bayes | **89.71%** | **89.69%** |
| Random Forest | 79.86% | 79.62% |
| Tuned Logistic Regression | **90.47%** | **90.44%** |

---

## Repository Structure

```text
src/
data/
outputs/
images/
presentation/
```

---

## Future Improvements

- BERT
- DistilBERT
- Spark Streaming
- MLflow
- Docker
