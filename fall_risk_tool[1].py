import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from joblib import dump, load
import os

# --- Train the model once ---
model_filename = 'fall_risk_model.joblib'

if not os.path.exists(model_filename):
    # Sample dataset
    df = pd.DataFrame({
        'age': [82, 75, 90, 65, 78, 45, 32],
        'prior_falls': [1, 0, 2, 1, 3, 0, 1],
        'mobility_score': [3, 4, 2, 5, 1, 4, 2],
        'medications': [5, 2, 7, 3, 6, 1, 3],
        'fall_risk': [1, 0, 1, 0, 1, 0, 1]
    })

    X = df[['age', 'prior_falls', 'mobility_score', 'medications']]
    y = df['fall_risk']

    model = RandomForestClassifier()
    model.fit(X, y)
    dump(model, model_filename)
else:
    model = load(model_filename)

# --- Streamlit App ---
st.set_page_config(page_title="Fall Risk Predictor", layout="centered")
st.title("🩺 Fall Risk Prediction Tool")

st.markdown("""
This AI-powered tool helps caregivers and healthcare professionals estimate a patient's fall risk.
Simply fill in the details below and get a quick prediction.
""")

# Input fields
age = st.slider("Age", 30, 100, 65)
prior_falls = st.number_input("Number of prior falls", 0, 10, value=1)
mobility = st.slider("Mobility score (1 = low, 5 = high)", 1, 5, 3)
medications = st.number_input("Number of medications", 0, 20, value=4)

if st.button("Predict Fall Risk"):
    input_data = pd.DataFrame([[age, prior_falls, mobility, medications]],
                               columns=['age', 'prior_falls', 'mobility_score', 'medications'])
    result = model.predict(input_data)[0]
    risk = "🔴 HIGH" if result == 1 else "🟢 LOW"
    st.success(f"**Predicted Fall Risk:** {risk}")
