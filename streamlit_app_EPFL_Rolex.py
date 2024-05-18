import streamlit as st
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import requests
from io import BytesIO
from nltk.corpus import wordnet
import nltk

nltk.download('wordnet')

# URL du modèle
MODEL_URL = "https://github.com/JohannG3/DS_ML/raw/main/camembert_model_full.pth"

# Charger le modèle et le tokenizer
@st.cache(allow_output_mutation=True)
def load_model():
    response = requests.get(MODEL_URL)
    model_file = BytesIO(response.content)
    model = CamembertForSequenceClassification.from_pretrained(model_file)
    tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
    return model, tokenizer

model, tokenizer = load_model()

def predict_difficulty(text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.softmax(logits, dim=-1)
    difficulty = torch.argmax(probabilities).item()
    return difficulty

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word, lang='fra'):
        for lemma in syn.lemmas('fra'):
            synonyms.add(lemma.name())
    return list(synonyms)

st.title('Prédiction de Difficulté de Texte en Français')

user_input = st.text_input("Entrez une phrase en français:")

if user_input:
    difficulty = predict_difficulty(user_input)
    st.write(f'Niveau de difficulté estimé : {difficulty}')
    
    words = user_input.split()
    chosen_word = st.selectbox("Choisissez un mot pour trouver des synonymes :", words)
    synonyms = get_synonyms(chosen_word)
    st.write("Synonymes :", synonyms)

    new_sentence = st.text_input("Entrez une nouvelle phrase pour augmenter le niveau de difficulté:")
