import pandas as pd
import string
import nltk
import joblib

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# download stopwords
nltk.download('stopwords')

# load dataset
data = pd.read_csv('cleaned_news.csv')

# preprocessing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# cleaning function
def clean_text(text):

    text = text.lower()

    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# prepare data
X = data['cleaned_text']
y = data['label']

# vectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# model
model = LogisticRegression()

model.fit(X, y)

# user input
news = input("Enter news text: ")

# preprocess input
cleaned_news = clean_text(news)

# vectorize input
news_vector = vectorizer.transform([cleaned_news])

# prediction
prediction = model.predict(news_vector)

print("\nPrediction:")

if prediction[0] == 1:
    print("REAL NEWS")
else:
    print("FAKE NEWS")
