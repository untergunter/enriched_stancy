import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from transformers import BertConfig,BertModel
import torch.nn.functional as F

class DoubleLoss(nn.Module):

    def __init__(self,device):
        super(DoubleLoss, self).__init__()
        self.device = device

        bert_config = BertConfig.from_pretrained('bert-base-uncased',
                                                 output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              config=bert_config)
        self.stance = nn.Linear(769, 2)
        self.cosine = nn.CosineSimilarity()
        self.dropout = nn.Dropout(0.1)
        self.similarity_cosine_loss = CosineEmbeddingLoss()
        self.stance_loss_func = CrossEntropyLoss()

    def forward(self, both_ids, both_mask, claim_ids, claim_mask,labels = None):

        both_hs = self.bert(both_ids, attention_mask=both_mask).pooler_output
        both_hs = self.dropout(both_hs)

        claim_hs = self.bert(claim_ids, attention_mask=claim_mask).pooler_output

        cos_sim = self.cosine(both_hs, claim_hs).unsqueeze(1)
        combined = torch.cat([both_hs, cos_sim], dim=1)
        probabilities = self.stance(combined)

        if labels is not None:

            # first loss

            stance_loss = \
                self.stance_loss_func(probabilities.view(-1,2),
                                 labels.view(-1))

            # second loss
            similarity_labels = torch.ones(labels.shape,device=self.device)
            similarity_labels[ labels == 0 ] = -1

            loss_claim = self.similarity_cosine_loss(both_hs,
                                                 claim_hs,
                                                 similarity_labels.float())

            loss = stance_loss + loss_claim

            return loss

        return combined, probabilities

    def predict(self,both_ids, both_mask, claim_ids, claim_mask):
        combined, probabilities = self.forward(both_ids, both_mask, claim_ids, claim_mask)
        _, predicted = torch.max(probabilities, 1)
        return predicted


class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.fc_1 = nn.Linear(68, 68)
        self.hidden = nn.Linear(68, 34)
        self.out = nn.Linear(68, 2)
        self.hidden_layer = self.init_hidden()
        self.activation = torch.tanh

    def init_hidden(self):
        return torch.zeros(34)

    def reset_hidden_layer(self):
        self.hidden_layer = self.init_hidden()

    def forward(self, x):
        x = torch.cat([self.hidden_layer.flatten(), x.flatten()])
        x = self.activation(self.fc_1(x))
        self.hidden_layer = self.activation(self.hidden(x))
        out = self.out(x)
        return out

class RnnClaimPerspective(nn.Module):
    def __init__(self):
        super(RnnClaimPerspective, self).__init__()
        self.fc_1 = nn.Linear(70, 70)
        self.hidden = nn.Linear(70, 35)
        self.out = nn.Linear(70, 2)
        self.hidden_layer = self.init_hidden()
        self.activation = torch.tanh

    def init_hidden(self):
        return torch.zeros(35)

    def reset_hidden_layer(self):
        self.hidden_layer = self.init_hidden()

    def forward(self, x):
        x = torch.cat([self.hidden_layer.flatten(), x.flatten()])
        x = self.activation(self.fc_1(x))
        self.hidden_layer = self.activation(self.hidden(x))
        out = self.out(x)
        return out


class DoubleLossFrozenBert(nn.Module):

    def __init__(self,device):
        super(DoubleLossFrozenBert, self).__init__()
        self.device = device

        bert_config = BertConfig.from_pretrained('bert-base-uncased',
                                                 output_hidden_states=True)
        self.bert = BertModel.from_pretrained('bert-base-uncased',
                                              config=bert_config)
        for param in self.bert.parameters():
            param.requires_grad = False
        self.fc_1 = nn.Linear(768, 768)
        self.fc_2 = nn.Linear(768, 768)
        self.stance = nn.Linear(768*2 +1, 2)
        self.cosine = nn.CosineSimilarity()
        self.dropout = nn.Dropout(0.1)
        self.similarity_cosine_loss = CosineEmbeddingLoss()
        self.stance_loss_func = CrossEntropyLoss()

    def forward(self, both_ids, both_mask, claim_ids, claim_mask,labels = None):

        both_hs = self.bert(both_ids, attention_mask=both_mask).pooler_output
        both_hs = F.relu(self.fc_1(both_hs))
        both_hs = F.relu(self.fc_2(both_hs))
        both_hs = self.dropout(both_hs)

        claim_hs = self.bert(claim_ids,
                             attention_mask=claim_mask).pooler_output
        claim_hs = F.relu(self.fc_1(claim_hs))
        claim_hs = F.relu(self.fc_2(claim_hs))
        claim_hs = self.dropout(claim_hs)

        cos_sim = self.cosine(both_hs, claim_hs).unsqueeze(1)
        combined = torch.cat([both_hs,claim_hs, cos_sim], dim=1)

        probabilities = self.stance(combined)

        if labels is not None:

            # first loss

            stance_loss = \
                self.stance_loss_func(probabilities.view(-1,2),
                                 labels.view(-1))

            # second loss
            similarity_labels = torch.ones(labels.shape,device=self.device)
            similarity_labels[ labels == 0 ] = -1

            loss_claim = self.similarity_cosine_loss(both_hs,
                                                 claim_hs,
                                                 similarity_labels.float())

            loss = stance_loss + loss_claim

            return loss

        return combined, probabilities

    def predict(self,both_ids, both_mask, claim_ids, claim_mask):
        combined, probabilities = self.forward(both_ids, both_mask, claim_ids, claim_mask)
        _, predicted = torch.max(probabilities, 1)
        return predicted

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = DoubleLossFrozenBert(device)