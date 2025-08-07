import joblib
from modules.preprocessing import clean_text

# Load model and vectorizer
model = joblib.load('models/model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict(text):
    if not isinstance(text, str) or not text.strip():
        return "Invalid input", 0.0

    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    
    # Get probability of depression (class 1)
    prob = model.predict_proba(vector)[0][1]  # probability of class 1 ("Depressed")
    label = "Depressed" if prob >= 0.5 else "Not Depressed"
    
    return label, round(prob * 100, 2)  # returning percentage


# def predict(text):
#     cleaned = clean_text(text)
#     vector = vectorizer.transform([cleaned])
#     prediction = model.predict(vector)[0]
#     confidence = np.max(model.predict_proba(vector)) * 100
#     label = "Depressed" if prediction == 1 else "Not Depressed"
#     return label, round(confidence, 2)
