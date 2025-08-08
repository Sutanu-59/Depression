# predict.py
import joblib
from modules.preprocessing import clean_text

# Load model and vectorizer once
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict(text):
    if not isinstance(text, str) or not text.strip():
        return "Invalid input", 0.0

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])

    prob = model.predict_proba(vector)[0][1]  # Probability of "Depressed"
    label = "Depressed" if prob >= 0.5 else "Not Depressed"

    return label, round(prob * 100, 2)
