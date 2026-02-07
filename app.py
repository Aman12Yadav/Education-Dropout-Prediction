import streamlit as st
import numpy as np
from model import train_model

st.set_page_config(page_title="Student Status Prediction", layout="centered")

st.title("🎓 Student Status Prediction System")
st.write("Predict whether a student will Dropout or Graduate")

@st.cache_resource
def load_model():
    return train_model()

model, features, encoder = load_model()
st.write("Model expects features in this order:", features)

if "age" not in st.session_state:
    st.session_state.age = 20
    st.session_state.gender = "Male"
    st.session_state.attendance = 75
    st.session_state.cgpa = 7.0
    st.session_state.parental_support = "Medium"

if st.button("🎯 Try Sample Student"):
    st.session_state.age = 19
    st.session_state.gender = "Male"
    st.session_state.attendance = 45
    st.session_state.cgpa = 4.5
    st.session_state.parental_support = "Low"

st.subheader("Enter Student Details")

with st.expander("👤 Student Information", expanded=True):
    age = st.slider("Age", 15, 30, key="age")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")

with st.expander("📚 Academic Performance"):
    attendance = st.slider("Attendance (%)", 0, 100, key="attendance")
    cgpa = st.slider("CGPA", 0.0, 10.0, key="cgpa")

with st.expander("🏠 Family Support"):
    parental_support = st.selectbox(
        "Parental Support Level",
        ["Low", "Medium", "High"],
        key="parental_support"
    )


gender_val = 1 if gender == "Male" else 0

support_map = {"Low": 0, "Medium": 1, "High": 2}
parental_support_val = support_map[parental_support]

# Create full feature vector with default values
input_dict = {feature: 0 for feature in features}

# Map UI values to correct feature names
input_dict["Age"] = age
input_dict["Gender"] = gender_val
input_dict["Attendance"] = attendance
input_dict["CGPA"] = cgpa
input_dict["ParentalSupport"] = parental_support_val

# Convert to ordered list
inputs = [input_dict[feature] for feature in features]

if st.button("🔍 Predict Student Status"):
    prediction = model.predict([inputs])[0]
    result = encoder.inverse_transform([prediction])[0]

    if result == "Graduate":
        st.success("✅ Low Risk of Dropout")
        st.markdown("""
        **Prediction:** Graduate 🎉  
        This student is likely to successfully complete their studies.
        """)
    else:
        st.error("⚠️ High Risk of Dropout")
        st.markdown("""
        **Prediction:** Dropout  
        Early academic or emotional intervention is recommended.
        """)
