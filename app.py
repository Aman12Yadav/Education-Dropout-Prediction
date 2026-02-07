import streamlit as st
import numpy as np
from model import train_model

st.set_page_config(page_title="Student Status Prediction", layout="centered")

st.title("🎓 Student Status Prediction System")
st.write("Predict whether a student will Dropout or Graduate")

@st.cache_resource
def load_model():
    return train_model()

model, features, encoder, feature_means = load_model()
#st.write("Model expects features in this order:", features)

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
# Start with mean values (safe defaults)
input_dict = feature_means.copy()


# Map UI values to correct feature names
input_dict["Age"] = age
input_dict["Gender"] = gender_val
input_dict["Attendance"] = attendance
input_dict["CGPA"] = cgpa
input_dict["ParentalSupport"] = parental_support_val

# ---- Heuristic adjustments for academic risk ----

# Attendance impact
if attendance < 40:
    for col in features:
        if (
            "attendance" in col.lower()
            or "curricular" in col.lower()
            or "units" in col.lower()
        ):
            input_dict[col] *= 0.3

# CGPA impact
if cgpa < 4:
    for col in features:
        if "grade" in col.lower():
            input_dict[col] *= 0.3

# Parental support impact
if parental_support_val == 0:  # Low support
    for col in features:
        if "support" in col.lower() or "scholar" in col.lower():
            input_dict[col] *= 0.5


# Convert to ordered list
inputs = [input_dict[feature] for feature in features]

if st.button("🔍 Predict Student Status"):

    # Get probabilities
    proba = model.predict_proba([inputs])[0]

    dropout_index = encoder.transform(["Dropout"])[0]
    dropout_prob = proba[dropout_index]

    # Decision logic
    if dropout_prob >= 0.6:
        st.error(f"⚠️ High Risk of Dropout ({dropout_prob*100:.1f}% probability)")
        st.markdown("""
        **Why this result?**
        - Very low attendance and/or CGPA
        - Weak academic engagement indicators
        - Low parental support increases risk
        """)
    else:
        st.success(f"✅ Low Risk of Dropout ({(1-dropout_prob)*100:.1f}% probability)")
        st.markdown("""
        **Why this result?**
        - Stable academic performance
        - Adequate attendance
        - Support factors reduce risk
        """)

