# app.py
import streamlit as st
from modules.predict import predict

# run_model()

st.set_page_config(page_title="Depression Detector", layout="centered")
st.title("🧠 Depression Detection AI App")
st.markdown("This tool uses AI to detect if your input shows signs of depression.")

# Load model accuracy from file
accuracy = None
try:
    with open("models/accuracy.txt", "r") as f:
        accuracy = f.read().strip()
except:
    accuracy = "Not available"

# st.markdown(f"📊 **Model Accuracy**: `{accuracy}%`")

# Input area
user_input = st.text_area("Enter a sentence or paragraph:", height=150)

if st.button("Analyze"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        label, confidence = predict(user_input)

        if label == "Depressed":
            st.error(f"**Prediction**: {label}")
            st.info(f"**Confidence**: {confidence}% chance of being Depressed")
        else:
            st.success(f"**Prediction**: {label}")
            st.info(f"**Confidence**: {confidence}% chance of being Not Depressed")


st.markdown("---")
st.markdown("🔁 You can retrain the model by running `trained_model.py` separately.")
