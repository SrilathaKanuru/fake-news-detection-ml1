# Fake News Detection System

## Overview

This project is an AI-based Fake News Detection System built using Machine Learning and Natural Language Processing (NLP).

The system preprocesses news text, converts it into numerical features using TF-IDF Vectorization, and predicts whether the news is REAL or FAKE using machine learning algorithms.

---

## Features

- Data preprocessing
- Text cleaning
- Stopword removal
- Stemming
- TF-IDF vectorization
- Fake news prediction
- Multiple machine learning models
- Interactive Streamlit web app

---

## Machine Learning Algorithms Used

- Logistic Regression
- K-Nearest Neighbour (KNN)
- Decision Tree
- Random Forest

---

## Technologies Used

- Python
- Pandas
- Scikit-learn
- NLTK
- Streamlit

---

## Project Structure

```text
fake-news-detection-ml/
│
├── data/
├── preprocess.py
├── train_model.py
├── predict.py
├── app.py
├── requirements.txt
└── README.md
```

---

## How It Works

1. Load dataset
2. Preprocess text data
3. Remove stopwords and punctuation
4. Convert text into vectors using TF-IDF
5. Train machine learning models
6. Predict fake or real news

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Future Improvements

- Deep learning integration
- Live news verification
- API integration
- Confidence score prediction
- Real-time news analysis

---

## Author

Srilatha
