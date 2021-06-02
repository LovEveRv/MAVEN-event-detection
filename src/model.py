import torch
import torch.nn as nn
import config
from transformers import BertModel


class BertOneStageModel(nn.Module):
    """
    BERT-based 1-stage model
    """

    def __init__(self, pretrained_path='bert-base-uncased', dropout=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(768, config.maven_class_numbers, bias=True)

    def forward(self, input_ids):
        rep = self.bert(input_ids=input_ids)
        rep = rep[1]
        rep = self.drop(rep)
        logits = self.fc(rep)
        return logits


class BertTwoStageModel(nn.Module):
    """
    BERT-based 2-stage model
    """

    def __init__(self, pretrained_path='bert-base-uncased', dropout=0.5):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 2, bias=True)
        self.fc2 = nn.Linear(768, config.maven_class_numbers, bias=True)

    def forward(self, input_ids):
        rep = self.bert(input_ids=input_ids)
        rep = rep[1]
        rep = self.drop(rep)
        logits_tf = self.fc1(rep)  # used for predicting T/F
        logits_cl = self.fc2(rep)  # used for predicting type
        return logits_tf, logits_cl


if __name__ == '__main__':
    """
    For test only
    """
    model = BertOneStageModel()
    # model = BertTwoStageModel()
    print('ok')
