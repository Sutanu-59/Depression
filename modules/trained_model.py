import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from preprocessing import clean_text

# Step 1: Load or clean dataset
CLEANED_FILE = "cleaned_dataset.csv"
RAW_FILE = "depression_analysis_dataset_100000.csv"

if os.path.exists(CLEANED_FILE):
    df = pd.read_csv(CLEANED_FILE)
    print("✅ Loaded cleaned dataset.")
else:
    df = pd.read_csv(RAW_FILE)
    df['cleaned'] = df['text'].apply(clean_text)
    df.to_csv(CLEANED_FILE, index=False)
    print("✅ Cleaned and saved dataset.")

# Step 2: Show class balance
print("\n📊 Class Distribution:")
print(df['label'].value_counts())

# Drop duplicates and shuffle
df.drop_duplicates(subset='cleaned', inplace=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 3: TF-IDF
tfidf = TfidfVectorizer(max_features=200, min_df=5, max_df=0.8)
X = tfidf.fit_transform(df['cleaned'])
y = df['label']

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)

# Step 5: Train model
model = LogisticRegression(C=0.1, max_iter=300)
model.fit(X_train, y_train)

# Step 6: Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {round(accuracy * 100, 2)}%")

# Step 7: Classification Report
print("\n📄 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 8: Confusion Matrix
print("🧾 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 9: Save model only if not overfitting
if accuracy >= 0.98:
    print("\n⚠️ Warning: Accuracy too high — possible overfitting. Model not saved.")
else:
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(tfidf, 'models/vectorizer.pkl')
    print("\n💾 Model and vectorizer saved.")
