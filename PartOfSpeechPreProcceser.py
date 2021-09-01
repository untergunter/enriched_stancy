from transformers import BertTokenizer
from prep import get_train_dev_test
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader,RandomSampler
from nltk import word_tokenize,pos_tag,download
from bert_preprocesing import make_tokenizer
download('averaged_perceptron_tagger')

class PartOfSpeechDataSet:
    tag_to_word = pd.read_csv('part_of_speech_words.csv')
    tag_to_word.index = tag_to_word.Tag
    tag_to_word = tag_to_word.Description.to_dict() # Tag -> Description

    def create_part_of_speech(self,text):
        """ from string of words -> word , POS , word , POS ... word, POS """
        tokens = word_tokenize(text)
        tags_and_pos = pos_tag(tokens)
        all_text = []
        for word,tag in tags_and_pos:
            all_text.append(word)
            if tag in self.tag_to_word:
                tag_name = self.tag_to_word[tag]
                all_text.append(tag_name)
        text_part_of_speech_alternately = ' '.join(
            tag_or_pos for tag_or_pos in all_text)
        return text_part_of_speech_alternately

    def __init__(self):
        pass

    def make_data_sets(self,raw_data,batch_size:int=8):
        claim_with_pos = raw_data['text'].str.strip().apply(
            lambda x:self.create_part_of_speech(x))

        claim = '[CLS] ' + claim_with_pos + ' [SEP]'

        perspective = raw_data['perspective'].str.strip().apply(
            lambda x:self.create_part_of_speech(x))
        perspective = perspective + ' [SEP]'
        together = claim + perspective

        label = [1 if single_label == 'supports' else 0 for single_label in raw_data['stance_label_3']]

        preprocessor = make_tokenizer()

        claim_ids, claim_masks = preprocessor(claim)
        together_ids, together_masks = preprocessor(together)
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
            batch_size=batch_size,
            num_workers=3
        )

        together_and_claim_loader = DataLoader(
            together_and_claim_dataset,
            sampler=RandomSampler(together_and_claim_dataset),
            batch_size=batch_size,
            num_workers=3
        )

        # together_only_loader->claim_ids, claim_masks, labels
        # together_and_claim_loader->together_ids,together_masks,
        # claim_ids,claim_masks,labels

        return together_only_loader, together_and_claim_loader