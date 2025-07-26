import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|@\S+|[^a-zA-Z]", " ", text)
    text = ' '.join([PorterStemmer().stem(word) 
                     for word in text.split() 
                     if word not in stopwords.words('english')])
    return text
