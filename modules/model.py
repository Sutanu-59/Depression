import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def train_and_save_model(data_path, model_path):
    df = pd.read_csv("C:/Users/User/Desktop/Depression Analysis/depression_data.csv")
    
    from preprocessing import clean_text
    df['cleaned'] = df['text'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned'])
    y = df['label']

    model = LogisticRegression()
    model.fit(X, y)

    joblib.dump(model, model_path)
    joblib.dump(tfidf, 'models/vectorizer.pkl')
