import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model

# --------------------
# LOAD MODELS
# --------------------
model_choice = st.sidebar.selectbox("Select Model", ["Random Forest", "DNN"])

if model_choice == "Random Forest":
    with open("random_forest_model.pkl", "rb") as f:
        model = pickle.load(f)
else:
    model = load_model("dnn_model.keras")

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load symptom features (from your original dataset)
df = pd.read_csv("Final_Augmented_dataset_Diseases_and_Symptoms.csv")
df.columns = df.columns.str.strip()
if 'diseases' in df.columns:
    df.rename(columns={'diseases': 'Disease'}, inplace=True)

symptoms = df.drop(columns='Disease').columns.tolist()

# --------------------
# UI INPUT
# --------------------
st.title("🧠 Disease Prediction from Symptoms")
st.markdown("### Select symptoms you are experiencing:")

selected_symptoms = st.multiselect("Choose symptoms:", symptoms)

# Create input vector
input_vector = np.zeros(len(symptoms))
for symptom in selected_symptoms:
    idx = symptoms.index(symptom)
    input_vector[idx] = 1

input_df = pd.DataFrame([input_vector], columns=symptoms)

# --------------------
# PREDICTION
# --------------------
if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        if model_choice == "DNN":
            pred_probs = model.predict(input_df)
            pred_idx = np.argmax(pred_probs, axis=1)[0]
        else:
            pred_idx = model.predict(input_df)[0]

        predicted_disease = le.inverse_transform([pred_idx])[0]

        st.success(f"🩺 **Predicted Disease:** {predicted_disease}")

        # Optional: Precautions using AI lookup
        st.markdown("### Precautions (via Wikipedia)")
        search_link = f"https://en.wikipedia.org/wiki/{predicted_disease.replace(' ', '_')}"
        st.markdown(f"🔍 [Click here to read more about {predicted_disease}]({search_link})")

# --------------------
# Footer with Accuracy
# --------------------
st.markdown("---")
st.markdown("Developed by **Vivekanand Ojha**")
st.markdown("📊 **Model Accuracy:**")
st.markdown("- 🎯 DNN Accuracy: **81.89%**")
st.markdown("- 🌲 Random Forest Accuracy: **78.10%**")
