import streamlit as st
import numpy as np
import joblib

st.title("Rock vs Mine Classifier")

model = joblib.load("rock_vs_mine_model.pkl")

# User input
st.write("Enter 60 features separated by commas:")
user_input = st.text_area("Input", placeholder="e.g. 0.02,0.03,0.05,... (60 values)")

if st.button("Predict"):
    try:
        values = np.array([float(i) for i in user_input.split(",")])
        if len(values) != 60:
            st.error("You must enter exactly 60 values.")
        else:
            prediction = model.predict([values])
            st.success(f"Prediction: {prediction[0]}")
    except:
        st.error("Invalid input. Please enter numeric values only.")
