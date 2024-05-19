import streamlit as st
import requests
from io import BytesIO
import torch
import torch.nn as nn
from transformers import AutoTokenizer, CamembertForSequenceClassification

class CustomCamembertForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.1):
        super(CustomCamembertForSequenceClassification, self).__init__()
        self.camembert = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.camembert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.camembert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else (None, logits)

# URL of the model on GitHub
MODEL_URL = "https://github.com/JohannG3/DS_ML/blob/main/camembert_model_full.pth?raw=true"

@st.cache(allow_output_mutation=True)
def load_model():
    # Download the model
    response = requests.get(MODEL_URL)
    model_path = BytesIO(response.content)
    # Create an instance of the model
    model = CustomCamembertForSequenceClassification('almanach/camembert-base', num_labels=your_num_labels, dropout_prob=0.1)
    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

model = load_model()
tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")

def predict_difficulty(sentence):
    tokenized = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        logits = model(input_ids=tokenized['input_ids'], attention_mask=tokenized['attention_mask'])
    prediction = torch.argmax(logits[1], dim=1)
    return prediction.item()

# Streamlit interface
st.title("Prédiction de la difficulté de la langue française")
sentence = st.text_input("Entrez une phrase en français:")
if sentence:
    prediction = predict_difficulty(sentence)
    st.write(f"Le niveau de difficulté prédit est : {prediction}")

