import streamlit as st
import requests
from io import BytesIO
import torch
from transformers import AutoTokenizer
from custom_model import CustomCamembertForSequenceClassification


# URL of the model on GitHub
MODEL_URL = "https://github.com/JohannG3/DS_ML/blob/main/camembert_model_full.pth?raw=true"

@st.cache(allow_output_mutation=True)
def load_model():
    # Download the model
    response = requests.get(MODEL_URL)
    model_path = BytesIO(response.content)
    # Load the model
    model = CamembertForSequenceClassification.from_pretrained(model_path)
    return model

model = load_model()
tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")

def predict_difficulty(sentence):
    tokenized = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(**tokenized)
    prediction = torch.argmax(logits.logits, dim=1)
    return prediction.item()

# Streamlit interface
st.title("Prédiction de la difficulté de la langue française")
sentence = st.text_input("Entrez une phrase en français:")
if sentence:
    prediction = predict_difficulty(sentence)
    st.write(f"Le niveau de difficulté prédit est : {prediction}")
