import joblib
from modules.preprocessing import clean_text

model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    result = model.predict(vector)
    return "Depressed" if result[0] == 1 else "Not Depressed"
