import streamlit as st
import numpy as np
import joblib

st.title("Rock vs Mine Classifier")

# Load the trained model
try:
    model = joblib.load("rock_vs_mine_model.pkl")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Button to generate random input
if st.button("Generate Random Input (60 values between 0 and 0.1)"):
    random_values = np.round(np.random.uniform(0, 0.1, size=60), 4)  # Fixed parenthesis
    st.session_state.default_input = ",".join(map(str, random_values))
    
# Initialize default input if not exists
if 'default_input' not in st.session_state:
    st.session_state.default_input = ""

# Text area to display or allow custom editing
user_input = st.text_area(
    "Enter 60 features (comma-separated):",
    value=st.session_state.default_input
)

if st.button("Predict"):
    if not user_input.strip():
        st.error("Please enter some values before predicting.")
    else:
        try:
            # Convert input to a list of floats
            values = np.array([float(i.strip()) for i in user_input.split(",")]).reshape(1, -1)
            
            # Check if there are exactly 60 values
            if values.shape[1] != 60:
                st.error(f"You must enter exactly 60 values. You entered {values.shape[1]}.")
            else:
                # Predict using the model
                prediction = model.predict(values)
                result = "Mine" if prediction[0] == "M" else "Rock"
                st.success(f"Prediction: {result}")
                
        except ValueError as e:
            st.error(f"Invalid input. Please enter numeric values only. Error: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")