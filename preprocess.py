import pandas as pd
import string
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# download stopwords
nltk.download('stopwords')

# load dataset
data = pd.read_csv('news.csv')

# initialize stemmer
stemmer = PorterStemmer()

# stopwords
stop_words = set(stopwords.words('english'))

# text cleaning function
def clean_text(text):
    
    # convert to lowercase
    text = text.lower()
    
    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # split words
    words = text.split()
    
    # remove stopwords and stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]
    
    return " ".join(words)

# apply preprocessing
data['cleaned_text'] = data['text'].apply(clean_text)

# save cleaned data
data.to_csv('cleaned_news.csv', index=False)

print("Data preprocessing completed!")
