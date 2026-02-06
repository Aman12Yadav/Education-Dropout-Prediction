import streamlit as st
import numpy as np
from model import train_model

st.set_page_config(page_title="Student Status Prediction", layout="centered")

st.title("🎓 Student Status Prediction System")
st.write("Predict whether a student will Dropout or Graduate")

model, features, encoder = train_model()

inputs = []

for feature in features:
    value = st.number_input(f"{feature}", value=0.0)
    inputs.append(value)

if st.button("Predict Status"):
    prediction = model.predict([inputs])[0]
    result = encoder.inverse_transform([prediction])[0]

    if result == "Graduate":
        st.success(f"Prediction: {result} 🎉")
    else:
        st.error(f"Prediction: {result} ⚠️")
