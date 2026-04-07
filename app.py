import streamlit as st
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- LOAD MODEL (LIGHTWEIGHT) ----------------
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- LOAD DATASETS ----------------
@st.cache_data
def load_data():
    symptom_df = pd.read_csv("datasets/Symptom2Disease.csv")
    desc_df = pd.read_csv("datasets/symptom_Description.csv")
    prec_df = pd.read_csv("datasets/symptom_precaution.csv")
    sev_df = pd.read_csv("datasets/Symptom-severity.csv")
    return symptom_df, desc_df, prec_df, sev_df

symptom_df, desc_df, prec_df, sev_df = load_data()

disease_names = symptom_df["label"].unique()

# ---------------- PREDICTION FUNCTION ----------------
def predict(symptoms):
    inputs = tokenizer(symptoms, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    disease = disease_names[predicted_class]

    description = desc_df[desc_df["Disease"] == disease]["Description"].values
    description = description[0] if len(description) > 0 else "No description available"

    precautions = prec_df[prec_df["Disease"] == disease].iloc[:, 1:].values
    precautions = precautions[0] if len(precautions) > 0 else []

    severity = "Unknown"
    symptoms_list = symptoms.lower().split()
    scores = []

    for s in symptoms_list:
        match = sev_df[sev_df.iloc[:, 0].str.lower() == s]
        if len(match) > 0:
            scores.append(match.iloc[0, 1])

    if len(scores) > 0:
        avg = sum(scores) / len(scores)
        if avg >= 4:
            severity = "High"
        elif avg >= 2:
            severity = "Medium"
        else:
            severity = "Low"

    return disease, description, precautions, severity


# ---------------- UI ----------------
st.set_page_config(page_title="AI Medical Triage Chatbot", page_icon="🩺")

st.title("🩺 AI Medical Triage Chatbot")

user_input = st.text_input("Enter symptoms")

if st.button("Diagnose"):
    if user_input.strip() == "":
        st.warning("Please enter symptoms")
    else:
        disease, description, precautions, severity = predict(user_input)

        st.subheader("🦠 Predicted Disease")
        st.write(disease)

        st.subheader("📖 Description")
        st.write(description)

        st.subheader("💊 Precautions")
        for p in precautions:
            st.write("✔", p)

        st.subheader("⚠ Severity Level")
        st.write(severity)

        if severity == "High":
            st.error("🚑 Visit hospital immediately")
        else:
            st.warning("🩹 Monitor symptoms")