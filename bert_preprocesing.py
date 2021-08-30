from transformers import BertTokenizer
from prep import get_train_dev_test
import pandas as pd
import numpy as np
from sklearn import preprocessing
import torch
from torch.utils.data import TensorDataset,DataLoader,RandomSampler

from transformers import pipeline
sentiment_analysis = pipeline("sentiment-analysis")

def make_tokenizer():
    tknzr = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_list_of_strings(list_of_strings):
        ids, attentions = [], []
        for input_string in list_of_strings:
            encoded = tknzr.encode_plus(input_string,
                                        add_special_tokens=False,
                                        truncation=True,
                                        padding='max_length',
                                        max_length=100,

                                        return_attention_mask=True,
                                        return_tensors='pt'
                                        )
            id_tensor = encoded['input_ids']
            attention_tensor = encoded['attention_mask']
            ids.append(id_tensor)
            attentions.append(attention_tensor)
        ids = torch.cat(ids, dim=0)
        attentions = torch.cat(attentions, dim=0)
        return ids, attentions

    return tokenize_list_of_strings

def make_2_kinds_data_set(raw_data,batch_size:int=24):


    claim = '[CLS] ' + raw_data['text'].str.strip() + ' [SEP]'
    perspective = raw_data['perspective'].str.strip() + ' [SEP]'
    together = claim + perspective

    label =[1 if single_label=='supports' else 0 for single_label in raw_data['stance_label_3']]

    preprocessor = make_tokenizer()

    claim_ids,claim_masks = preprocessor(claim)
    together_ids,together_masks = preprocessor(together)
    labels = torch.tensor(label)


    together_only_dataset = TensorDataset(together_ids,
                                          together_masks,
                                          labels)
    together_and_claim_dataset = TensorDataset(together_ids,
                                               together_masks,
                                               claim_ids,
                                               claim_masks,
                                               labels)

    together_only_loader = DataLoader(
        together_only_dataset,
        sampler=RandomSampler(together_only_dataset),
        batch_size=batch_size
    )

    together_and_claim_loader = DataLoader(
        together_and_claim_dataset,
        sampler=RandomSampler(together_and_claim_dataset),
        batch_size=batch_size
    )

    # together_only_loader->claim_ids, claim_masks, labels
    # together_and_claim_loader->together_ids,together_masks,
    # claim_ids,claim_masks,labels

    return together_only_loader,together_and_claim_loader

def sent(text):

    return 0 if sentiment_analysis(text)[0]['label'] == 'NEGATIVE' else 1


def make_2_kinds_data_set_with_sentiment(raw_data,batch_size:int=24, bert_version='bert-base-uncased'):

    claim = '[CLS] ' + raw_data['text'].str.strip() + ' [SEP]'
    perspective = raw_data['perspective'].str.strip() + ' [SEP]'
    together = claim + perspective
    label =[1 if single_label=='supports' else 0 for single_label in raw_data['stance_label_3'] ]

    preprocessor = make_tokenizer()

    claim_ids,claim_masks = preprocessor(claim)
    together_ids,together_masks = preprocessor(together)

    labels = torch.tensor(label)

    sentiment = raw_data['perspective'].apply(sent)
    sentiments = torch.tensor(sentiment.tolist())

    together_only_dataset = TensorDataset(together_ids,
                                          together_masks,
                                          labels, sentiments)


    together_and_claim_dataset = TensorDataset(together_ids,
                                               together_masks,
                                               claim_ids,
                                               claim_masks,
                                               labels, sentiments)

    together_only_loader = DataLoader(
        together_only_dataset,
        sampler=RandomSampler(together_only_dataset),
        batch_size=batch_size
    )

    together_and_claim_loader = DataLoader(
        together_and_claim_dataset,
        sampler=RandomSampler(together_and_claim_dataset),
        batch_size=batch_size
    )

    # together_only_loader->claim_ids, claim_masks, labels
    # together_and_claim_loader->together_ids,together_masks,
    # claim_ids,claim_masks,labels

    return together_only_loader,together_and_claim_loader

if __name__ =='__main__':

    train, dev, test = get_train_dev_test()

    dev_together_only_loader,dev_together_and_claim_loader =\
        make_2_kinds_data_set(dev)
    train_together_only_loader,train_together_and_claim_loader =\
        make_2_kinds_data_set(dev)
    test_together_only_loader,test_together_and_claim_loader =\
        make_2_kinds_data_set(dev)


