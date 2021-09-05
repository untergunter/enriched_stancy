import pandas as pd
from bert_preprocessing import get_train_dev_test
from nltk import word_tokenize,pos_tag,download
download('averaged_perceptron_tagger')
from collections import Counter
import torch
import torch.nn as nn
from torch.utils.data import Dataset

def make_part_of_speech_embeddings():
    # based on list from https://www.guru99.com/pos-tagging-chunking-nltk.html

    nltk_pos_types = """CC 	coordinating conjunction
    CD 	cardinal digit
    DT 	determiner
    EX 	existential there
    FW 	foreign word
    IN 	preposition/subordinating conjunction
    JJ 	This NLTK POS Tag is an adjective (large)
    JJR 	adjective, comparative (larger)
    JJS 	adjective, superlative (largest)
    LS 	list market
    MD 	modal (could, will)
    NN 	noun, singular (cat, tree)
    NNS 	noun plural (desks)
    NNP 	proper noun, singular (sarah)
    NNPS 	proper noun, plural (indians or americans)
    PDT 	predeterminer (all, both, half)
    POS 	possessive ending (parent\ 's)
    PRP 	personal pronoun (hers, herself, him,himself)
    PRP$ 	possessive pronoun (her, his, mine, my, our )
    RB 	adverb (occasionally, swiftly)
    RBR 	adverb, comparative (greater)
    RBS 	adverb, superlative (biggest)
    RP 	particle (about)
    TO 	infinite marker (to)
    UH 	interjection (goodbye)
    VB 	verb (ask)
    VBG 	verb gerund (judging)
    VBD 	verb past tense (pleaded)
    VBN 	verb past participle (reunified)
    VBP 	verb, present tense not 3rd person singular(wrap)
    VBZ 	verb, present tense with 3rd person singular (bases)
    WDT 	wh-determiner (that, what)
    WP 	wh- pronoun (who)
    WRB 	wh- adverb (how)"""

    all_nltk_pos = [line.split('\t')[0].strip()
                    for line in nltk_pos_types.split('\n')]

    position_one_hot_encoding = {part_of_speech: i
                                 for i, part_of_speech
                                 in enumerate(all_nltk_pos)}
    return position_one_hot_encoding


class POSOneHotEnc:

    def __init__(self):
        self.pos_positional = make_part_of_speech_embeddings()

    def text_to_one_hot(self, text):
        """ creates a list of one hot encodings based on the part of speech """
        tokens = word_tokenize(text)
        tags_and_pos = pos_tag(tokens)
        parts_of_speech = [tag[1] for tag in tags_and_pos]
        positions_to_be_1 = [(row, self.pos_positional[part])
                             for row, part in enumerate(parts_of_speech)
                             if part in self.pos_positional]
        one_hot = torch.zeros(len(tokens),
                              len(self.pos_positional) +1 # for separator between claim and perspective
                              )
        for position in positions_to_be_1:
            one_hot[position] = 1
        one_hot_as_lines = list(torch.unbind(one_hot))

        return one_hot_as_lines

    def claim_and_perspective_to_one_hot(self,claim,perspective):
        claim_tensors = self.text_to_one_hot(claim)
        separator_tensor = [torch.zeros(1,len(self.pos_positional) +1)]
        perspective_tensors = self.text_to_one_hot(perspective)
        all_together = claim_tensors + separator_tensor + perspective_tensors
        return all_together

class PosRnnDataSet(Dataset):

    def __init__(self,df,perspective_only = True):
        self.tensor_maker = POSOneHotEnc()
        self.lines , self.targets = self.get_perspective_labels(df) if \
        perspective_only else self.get_both_labels(df)

    def get_perspective_labels(self,df):
        pos_tensors = [self.tensor_maker.text_to_one_hot(text)
                       for text in df['perspective']]
        label =[torch.tensor([1]) if single_label=='supports'
                else torch.tensor([0])
                for single_label in df['stance_label_3']]
        return pos_tensors,label

    def get_both_labels(self,df):
        both_tensors = (df.apply(lambda x:
                        self.tensor_maker.claim_and_perspective_to_one_hot(
                            x.text, x.perspective), axis=1)).tolist()
        label = [torch.tensor([1]) if single_label == 'supports'
                 else torch.tensor([0])
                 for single_label in df['stance_label_3']]
        return both_tensors,label

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        return self.lines[index],self.targets[index]


if __name__ == '__main__':
    pass