import streamlit as st
import requests
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
from nltk.corpus import wordnet
import nltk
from io import BytesIO

nltk.download('wordnet')

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

# Initialize the tokenizer
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')

model = load_model()

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
