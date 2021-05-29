import torch
import torch.nn as nn
import config
from transformers import BertForSequenceClassification


class BertOneStageModel(nn.Module):
    """
    BERT-based 1-stage model
    """

    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout()
        self.fc = nn.Linear(768, config.maven_class_numbers, bias=True)

    def forward(self, *args):
        rep = self.bert(*args)
        rep = self.drop(rep)
        logits = self.fc(rep)
        return logits


class BertTwoStageModel(nn.Module):
    """
    BERT-based 2-stage model
    """

    def __init__(self):
        super().__init__()
        self.bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout()
        self.fc1 = nn.Linear(768, 2, bias=True)
        self.fc2 = nn.Linear(768, config.maven_class_numbers, bias=True)

    def forward(self, *args):
        rep = self.bert(*args)
        rep = self.drop(rep)
        logits_tf = self.fc1(rep)  # used for predicting T/F
        logits_cl = self.fc2(rep)  # used for predicting type
        return logits_tf, logits_cl
