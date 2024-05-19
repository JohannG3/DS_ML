import streamlit as st
import requests
from io import BytesIO
import torch
from torch import nn
from transformers import AutoTokenizer, CamembertConfig

# Custom model class
class CustomCamembertForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels, dropout_prob=0.1):
        super(CustomCamembertForSequenceClassification, self).__init__()
        self.camembert = CamembertForSequenceClassification(CamembertConfig.from_pretrained(config, num_labels=num_labels))
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.camembert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.camembert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # First token [CLS] for classification
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
    model_path.seek(0)  # Rewind the BytesIO object
    # Load the model
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

# Load and configure the model and tokenizer
model = load_model()
tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")

# Define label dictionary
id2label = {0: 'Easy', 1: 'Medium', 2: 'Hard'}  # Update with your actual labels

def predict_difficulty(sentence):
    tokenized = tokenizer(sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**tokenized)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
    prediction = torch.argmax(logits, dim=1)
    return id2label[prediction.item()]

# Streamlit interface
st.title("Prédiction de la difficulté de la langue française")
sentence = st.text_input("Entrez une phrase en français:")
if sentence:
    prediction = predict_difficulty(sentence)
    st.write(f"Le niveau de difficulté prédit est : {prediction}")
