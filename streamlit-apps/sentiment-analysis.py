import streamlit as st
from transformers import pipeline
analysis=pipeline('sentiment-analysis')
st.title("sentiment Analaysis web app")
user_input=st.text_area("Enter text to analyze:")
if st.button("Analyze"):
    if user_input:
        result=analysis(user_input)[0]
        label=result['label']
        score=result['score']
        st.write(f'sentiment: **{label}**')
        st.write(f"Confidence:{round(score,2)}")
    else:
        st.warning("Please enter some text to analyze")