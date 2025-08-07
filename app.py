import streamlit as st
from modules.trained_model import run_model
from modules.predict import predict

# creating the trained model
run_model()

st.set_page_config(page_title="Depression Detector", layout="centered")
st.title("🧠 Depression Detection AI App")
st.markdown("This tool uses AI to detect if your input shows signs of depression.")

user_input = st.text_area("Enter a sentence or paragraph:")

# Load accuracy
# accuracy = None
# try:
#     with open("models/accuracy.txt", "r") as f:
#         accuracy = f.read().strip()
# except:
#     accuracy = "Not available"

# st.markdown(f"**📊 Model Accuracy**: {accuracy}%")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text before analyzing.")
    else:
        label, percent = predict(user_input)
        if label=="Depressed":
            st.success(f"**Prediction**: {label}")
            st.info(f"**Confidence**: {percent}% chance of being Depressed")
        else:
            st.success(f"**Prediction**: {label}")

st.markdown("---")

