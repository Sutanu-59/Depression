import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
import joblib

from modules.preprocessing import clean_text

def run_model():
    # Load dataset
    df = pd.read_csv('depression_analysis_dataset_100000.csv')

    # Clean the text
    df['cleaned'] = df['text'].apply(clean_text)

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(max_features=1000, min_df=1, max_df=0.9)
    X = tfidf.fit_transform(df['cleaned'])
    y = df['label']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

    # Train the model
    model = LogisticRegression(C=0.1, max_iter=1000)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_percent = round(accuracy * 100, 2)

    # ✅ Print in terminal
    print(f"✅ Model Accuracy: {accuracy_percent}%")

    # Save accuracy to a text file
    # with open('models/accuracy.txt', 'w') as f:
    #     f.write(str(round(accuracy * 100, 2)))  # Save as percentage

    # Ensure 'models' folder exists
    os.makedirs('models', exist_ok=True)

    # Save the model and vectorizer
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(tfidf, 'models/vectorizer.pkl')

    print("✅ Model and vectorizer saved successfully.")
