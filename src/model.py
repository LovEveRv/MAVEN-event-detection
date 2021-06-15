import torch
import torch.nn as nn
import config
from transformers import BertModel


# class BertOneStageModel(nn.Module):
#     """
#     BERT-based 1-stage model
#     """

#     def __init__(self, pretrained_path='bert-base-uncased', dropout=0.5):
#         super().__init__()
#         self.bert = BertModel.from_pretrained(pretrained_path)
#         self.drop = nn.Dropout(dropout)
#         self.fc = nn.Linear(768, config.maven_class_numbers, bias=True)

#     def forward(self, input_ids):
#         rep = self.bert(input_ids=input_ids)
#         rep = rep[1]
#         rep = self.drop(rep)
#         logits = self.fc(rep)
#         return logits


class BertTwoStageModel(nn.Module):
    """
    BERT-based 2-stage model
    """

    def __init__(self, pretrained_path='bert-base-uncased', dropout=0.5):
        super().__init__()
        self.hidden_size = 768
        self.bert = BertModel.from_pretrained(pretrained_path)
        self.drop = nn.Dropout(dropout)
        self.maxpooling = nn.MaxPool1d(config.sentence_max_length)
        self.fc1 = nn.Linear(2 * self.hidden_size, 2, bias=True)
        self.fc2 = nn.Linear(2 * self.hidden_size, config.maven_class_numbers, bias=True)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, l_mask=None, r_mask=None):
        batch_size = input_ids.size(0)
        rep = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        conved = rep[0]
        conved = conved.transpose(1, 2)
        conved = conved.transpose(0, 1)
        L = (conved * l_mask).transpose(0, 1)
        R = (conved * r_mask).transpose(0, 1)
        L = L + torch.ones_like(L)
        R = R + torch.ones_like(R)
        l_pooled = self.maxpooling(L).contiguous().view(batch_size, self.hidden_size)
        r_pooled = self.maxpooling(R).contiguous().view(batch_size, self.hidden_size)
        pooled = torch.cat((l_pooled, r_pooled), 1)
        pooled = pooled - torch.ones_like(pooled)
        pooled = self.drop(pooled)
        
        logits_tf = self.fc1(pooled)  # used for predicting T/F
        logits_cl = self.fc2(pooled)  # used for predicting type
        
        return logits_tf, logits_cl


if __name__ == '__main__':
    """
    For test only
    """
    model = BertOneStageModel()
    # model = BertTwoStageModel()
    print('ok')
