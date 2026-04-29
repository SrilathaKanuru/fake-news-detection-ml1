import pandas as pd
import string
import nltk
import streamlit as st

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# download stopwords
nltk.download('stopwords')

# page title
st.title("Fake News Detection System")

st.write("Enter a news article below to check if it is Fake or Real.")

# load dataset
data = pd.read_csv('cleaned_news.csv')

# preprocessing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# clean text function
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

# vectorization
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

# train model
model = LogisticRegression()

model.fit(X, y)

# user input
user_input = st.text_area("Paste News Content")

# predict button
if st.button("Check News"):

    cleaned_input = clean_text(user_input)

    vector_input = vectorizer.transform([cleaned_input])

    prediction = model.predict(vector_input)

    if prediction[0] == 1:
        st.success("This looks like REAL news.")
    else:
        st.error("This looks like FAKE news.")
