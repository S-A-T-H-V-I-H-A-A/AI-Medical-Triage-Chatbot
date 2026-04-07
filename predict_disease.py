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

    # Get description
    description = desc_df[desc_df["Disease"].str.lower() == disease.lower()]["Description"].values
    description = description[0] if len(description) > 0 else "No description available"

    # Get precautions
    precautions = prec_df[prec_df["Disease"].str.lower() == disease.lower()].iloc[:,1:].values
    precautions = precautions[0] if len(precautions) > 0 else []

    # Calculate severity using symptom weights
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


while True:

    text = input("\nEnter symptoms (type 'exit' to stop): ")

    if text.lower() == "exit":
        break

    disease, description, precautions, severity = predict(text)

    print("\nPredicted Disease:", disease)
    print("\nDescription:", description)

    print("\nPrecautions:")
    for p in precautions:
        print("-", p)

    print("\nSeverity Level:", severity)

    if severity == "High":
        print("Recommendation: Visit hospital immediately")
    else:
        print("Recommendation: Monitor symptoms and consult doctor if needed")