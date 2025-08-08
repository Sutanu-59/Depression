# preprocessing.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords only once
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|[^a-zA-Z]", " ", text)
    return ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])
