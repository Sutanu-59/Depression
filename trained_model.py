import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

from modules.preprocessing import clean_text

# Load dataset
df = pd.read_csv('depression_data.csv')

# Clean the text
df['cleaned'] = df['text'].apply(clean_text)

# TF-IDF vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned'])
y = df['label']

# Train the model
model = LogisticRegression()
model.fit(X, y)

# Ensure 'models' folder exists
os.makedirs('models', exist_ok=True)

# Save the model and vectorizer
joblib.dump(model, 'models/model.pkl')
joblib.dump(tfidf, 'models/vectorizer.pkl')

print("✅ Model and vectorizer saved successfully.")
