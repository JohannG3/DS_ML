import torch
import torch.nn as nn
from transformers import CamembertForSequenceClassification

class CustomCamembertForSequenceClassification(nn.Module):
    def __init__(self, model_name, num_labels, dropout_prob=0.1):
        super(CustomCamembertForSequenceClassification, self).__init__()
        self.camembert = CamembertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.camembert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.camembert.roberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]  # Utilisation du premier token [CLS] pour classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.classifier.out_features), labels.view(-1))

        return (loss, logits) if loss is not None else (None, logits)
