import streamlit as st
import pandas as pd
import torch
import os
import zipfile
import gdown
from transformers import BertTokenizer, BertForSequenceClassification

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Medical Triage Chatbot", page_icon="🩺")

# ---------------- LOAD MODEL (SAFE + CACHED) ----------------
@st.cache_resource
def load_model():
    model_path = "model"

    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1GkSEN0S88XgeG7ObWeFEPFT27ZhExoC6"
        output = "model.zip"

        try:
            # Download model
            gdown.download(url, output, quiet=False, fuzzy=True)

            # Extract model
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall(model_path)

            # Remove zip after extraction
            os.remove(output)

        except Exception as e:
            st.error("❌ Model download failed. Check Google Drive permissions.")
            st.stop()

    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.eval()
    except Exception:
        st.error("❌ Model files are corrupted or missing.")
        st.stop()

    return tokenizer, model

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    try:
        symptom_df = pd.read_csv("datasets/Symptom2Disease.csv")
        desc_df = pd.read_csv("datasets/symptom_Description.csv")
        prec_df = pd.read_csv("datasets/symptom_precaution.csv")
        sev_df = pd.read_csv("datasets/Symptom-severity.csv")
    except Exception:
        st.error("❌ Dataset files missing. Check datasets folder.")
        st.stop()

    return symptom_df, desc_df, prec_df, sev_df

symptom_df, desc_df, prec_df, sev_df = load_data()
disease_names = symptom_df["label"].unique()

# ---------------- PREDICTION ----------------
def predict(symptoms):
    tokenizer, model = load_model()

    inputs = tokenizer(symptoms, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    disease = disease_names[predicted_class]

    description = desc_df[desc_df["Disease"] == disease]["Description"].values
    description = description[0] if len(description) > 0 else "No description available"

    precautions = prec_df[prec_df["Disease"] == disease].iloc[:, 1:].values
    precautions = precautions[0] if len(precautions) > 0 else []

    # -------- Severity Calculation --------
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
st.title("🩺 AI Medical Triage Chatbot")

st.write("Enter symptoms and get AI-based disease prediction with precautions.")

user_input = st.text_input("🧾 Enter symptoms (e.g., fever headache cough)")

if st.button("🔍 Diagnose"):
    if user_input.strip() == "":
        st.warning("⚠ Please enter symptoms")
    else:
        with st.spinner("Analyzing symptoms... ⏳"):
            disease, description, precautions, severity = predict(user_input)

        st.subheader("🦠 Predicted Disease")
        st.success(disease)

        st.subheader("📖 Description")
        st.info(description)

        st.subheader("💊 Precautions")
        for p in precautions:
            st.write("✔", p)

        st.subheader("⚠ Severity Level")
        if severity == "High":
            st.error("🚑 High Severity - Visit hospital immediately")
        elif severity == "Medium":
            st.warning("⚠ Medium Severity - Consult a doctor")
        else:
            st.success("✅ Low Severity - Monitor symptoms")
