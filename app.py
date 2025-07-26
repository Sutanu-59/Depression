import streamlit as st
from modules.predict import predict

st.set_page_config(page_title="Depression Detector", layout="centered")
st.title("Depression Detection AI App")

user_input = st.text_area("Enter a sentence or paragraph:")

if st.button("Analyze"):
    prediction = predict(user_input)
    st.success(f"The input text is: **{prediction}**")
