import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from transformers import BertConfig,BertModel

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


if __name__ == '__main__':
    pass