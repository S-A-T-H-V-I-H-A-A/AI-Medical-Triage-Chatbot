import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load datasets
symptom_df = pd.read_csv("C:\\Users\\jeiya\\OneDrive\\Desktop\\ai-medical-triage-chatbot\\datasets\\Symptom2Disease.csv")
desc_df = pd.read_csv("C:\\Users\\jeiya\\OneDrive\\Desktop\\ai-medical-triage-chatbot\\datasets\\symptom_Description.csv")
prec_df = pd.read_csv("C:\\Users\\jeiya\\OneDrive\\Desktop\\ai-medical-triage-chatbot\\datasets\\symptom_precaution.csv")
sev_df = pd.read_csv("C:\\Users\\jeiya\\OneDrive\\Desktop\\ai-medical-triage-chatbot\\datasets\\Symptom-severity.csv")

disease_names = symptom_df["label"].unique()

model_path = "results/checkpoint-360"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(model_path)

def predict(symptoms):

    inputs = tokenizer(symptoms, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    disease = disease_names[predicted_class]

    description = desc_df[desc_df["Disease"] == disease]["Description"].values
    description = description[0] if len(description) > 0 else "No description available"

    precautions = prec_df[prec_df["Disease"] == disease].iloc[:,1:].values
    precautions = precautions[0] if len(precautions) > 0 else []

    severity = "Unknown"

    symptoms_list = symptoms.lower().split()
    scores = []

    for s in symptoms_list:
        match = sev_df[sev_df.iloc[:,0].str.lower() == s]
        if len(match) > 0:
            scores.append(match.iloc[0,1])

    if len(scores) > 0:
        avg = sum(scores) / len(scores)

        if avg >= 4:
            severity = "High"
        elif avg >= 2:
            severity = "Medium"
        else:
            severity = "Low"

    return disease, description, precautions, severity


# ------------------- Streamlit UI -------------------
import streamlit as st

st.set_page_config(page_title="AI Medical Triage Chatbot", page_icon="🩺", layout="centered")

# --------- CUSTOM CSS ---------
st.markdown("""
<style>

html, body, [class*="css"]  {
    background-color: #ffeaea;
}

/* Title */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #b30000;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #333333;
    margin-bottom: 25px;
}

/* Input box */
.stTextInput>div>div>input {
    border-radius: 10px;
    border: 2px solid #b30000;
    padding: 12px;
}

/* Button */
.stButton>button {
    background-color: #b30000;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #800000;
    transform: scale(1.03);
}

/* Result cards */
.card {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.15);
    animation: fadeIn 0.6s ease-in;
}

/* Animation */
@keyframes fadeIn {
    from {opacity:0; transform:translateY(10px);}
    to {opacity:1; transform:translateY(0);}
}

/* Icon size */
.icon {
    font-size: 26px;
}

</style>
""", unsafe_allow_html=True)

# --------- HEADER ---------
st.markdown('<div class="title">🩺 AI Medical Triage Chatbot</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle">Enter symptoms and the AI will predict possible disease and precautions.</div>',
    unsafe_allow_html=True
)

# --------- INPUT ---------
user_input = st.text_input("🧾 Enter symptoms")

if st.button("🔍 Diagnose"):

    disease, description, precautions, severity = predict(user_input)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🦠 Predicted Disease")
    st.write(disease)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📖 Description")
    st.write(description)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💊 Precautions")

    for p in precautions:
        st.write("✔", p)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ⚠ Severity Level")
    st.write(severity)

    if severity == "High":
        st.error("🚑 Visit hospital immediately")
    else:
        st.warning("🩹 Monitor symptoms and consult doctor if needed")

    st.markdown("</div>", unsafe_allow_html=True)