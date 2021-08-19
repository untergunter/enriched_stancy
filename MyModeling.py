import torch
import torch.nn as nn
from transformers import BertForSequenceClassification


class BertForSequenceClassificationDualLoss(nn.Module):
    """ this is based on the same class from modeling.py """
    def __init__(self,hidden_dropout_prob,hidden_size):
        self.bert = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False
            )
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size + 1, 2) # +1 for the cos distance
        self.cosine = nn.CosineSimilarity()
        self.alpha = 0.5

    def forward(self, combined_ids,combined_attention,claim_ids,claim_attention, labels=None, sim_labels=None):

        combined_out = self.bert(combined_ids,
                                token_type_ids=None,
                                attention_mask=combined_attention,
                                output_all_encoded_layers=False,
                                return_dict=True,
                                output_hidden_states=True
                                )

        claim_out = self.bert(  claim_ids,
                                token_type_ids=None,
                                attention_mask=claim_attention,
                                output_all_encoded_layers=False,
                                return_dict=True,
                                output_hidden_states=True
                                )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, sim_labels=None):

        sen1_attention_mask = (1 - token_type_ids) * attention_mask

        _, pooled_output_combined = self.bert(input_ids, token_type_ids, attention_mask,
                                              output_all_encoded_layers=False)
        pooled_output_combined = self.dropout(pooled_output_combined)

        _, pooled_output_sen1 = self.bert(input_ids, token_type_ids, sen1_attention_mask,
                                          output_all_encoded_layers=False)

        cos_sim = self.cosine(pooled_output_combined, pooled_output_sen1).unsqueeze(1)

        combined = torch.cat([pooled_output_combined, cos_sim], dim=1)
        logits = self.classifier(combined)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss_bert = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            loss_cosine = CosineEmbeddingLoss()
            loss_intent = loss_cosine(pooled_output_combined, pooled_output_sen1, sim_labels.float())

            loss = loss_bert + loss_intent

            return loss
        else:
            return logits