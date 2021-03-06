import torch
import torch.nn as nn
import pandas as pd
import os

from glob import glob
from datetime import datetime
from transformers import BertTokenizer,AdamW,BertConfig,BertModel
from torch.utils.data import TensorDataset,DataLoader,RandomSampler
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from sklearn.metrics import f1_score,recall_score,precision_score
from tqdm import tqdm
from models import DoubleLossFrozenBert
from torch.optim import Adam
#%%

def test_consistency_model(model, dataloader, device):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            together_ids, together_masks, claim_ids, claim_masks, labels = batch
            together_ids = together_ids.to(device)
            together_masks = together_masks.to(device)
            claim_ids = claim_ids.to(device)
            claim_masks = claim_masks.to(device)
            labels = labels.to(device)

            model_prediction = model.predict(together_ids,
                              together_masks,
                              claim_ids,
                              claim_masks,
                              )
            y_true += [int(label) for label in labels]
            y_pred += [int(label) for label in model_prediction]
    return y_true,y_pred

def find_file(files_list,file_name):
    for file in files_list:
        if file.split(os.sep)[-1] == file_name:
            return file
    return None

def get_paper_train_dev_test():
    all_tsv = glob('./**/*.tsv', recursive=True)
    dev = find_file(all_tsv,'dev.tsv')
    dev = pd.read_csv(dev,
                      sep='\t',
                      names=['index','text','perspective','stance_label_3']) if dev else None
    train = find_file(all_tsv, 'train.tsv')
    train = pd.read_csv(train,
                      sep='\t',
                      names=['index', 'text', 'perspective', 'stance_label_3']) if train else None

    test = find_file(all_tsv, 'test.tsv')
    test = pd.read_csv(test,
                        sep='\t',
                        names=['index', 'text', 'perspective', 'stance_label_3']) if test else None

    return train,dev,test

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

    label =[1 if single_label=='supports' else 0 for single_label in raw_data['stance_label_3'] ]

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
        batch_size=batch_size,
        num_workers=3
    )

    together_and_claim_loader = DataLoader(
        together_and_claim_dataset,
        sampler=RandomSampler(together_and_claim_dataset),
        batch_size=batch_size,
        num_workers = 3
    )

    # together_only_loader->claim_ids, claim_masks, labels
    # together_and_claim_loader->together_ids,together_masks,
    # claim_ids,claim_masks,labels

    return together_only_loader,together_and_claim_loader

def add_to_result_csv(loss,f1,precision,
                      recall,seconds,epoch,hp):
  df = pd.DataFrame({  'epoch':[epoch],
                       'loss':[loss],
                       'f1':[f1],
                       'precision':[precision],
                       'recall':[recall],
                       'seconds':[seconds],
                       'hyper_parameters':[str(hp)]})

  df.to_csv('/home/ido/data/idc/advanced ml/final_project/bert_frozen_hp_opt.csv',
            header=False,mode='a',index=False)

#%%



def train_consistency(batch_size,lr,eps):
    hyper_parameters = {'batch_size':batch_size,
                        'lr':lr,'eps':eps,
                        'dropout':0.1,'optimizer':'Adam'}

    train, dev, test = get_paper_train_dev_test()
    train_together_only_loader, train_together_and_claim_loader = \
        make_2_kinds_data_set(train,batch_size)
    # dev_together_only_loader, dev_together_and_claim_loader = \
    #     make_2_kinds_data_set(dev,batch_size)
    test_together_only_loader, test_together_and_claim_loader = \
        make_2_kinds_data_set(test,batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoubleLossFrozenBert(device).to(device)
    # optimizer = AdamW(model.parameters(),
    #                   lr=lr,
    #                   eps=eps
    #                   )
    optimizer = Adam(model.parameters(),lr = lr)
    for epoch in range(1,11):
      # train
        start_time = datetime.now()
        total_loss = 0
        model.train()
        for batch in tqdm(train_together_and_claim_loader):

            together_ids, together_masks, claim_ids, claim_masks, labels = batch
            together_ids = together_ids.to(device)
            together_masks = together_masks.to(device)
            claim_ids = claim_ids.to(device)
            claim_masks = claim_masks.to(device)
            labels = labels.to(device)

            loss = model(
                        together_ids,
                        together_masks,
                        claim_ids,
                        claim_masks,
                        labels
                        )

            optimizer.zero_grad()
            model.zero_grad()

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # test and save

        y_true, y_pred = test_consistency_model(model, test_together_and_claim_loader, device)
        end_time = datetime.now()
        total_seconds = (end_time-start_time).seconds

        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        print(f'epoch:{epoch},loss:{total_loss},precision:{precision}'
              f',f1:{weighted_f1} ,recall:{recall},{total_seconds} seconds')
        add_to_result_csv(total_loss,weighted_f1,precision,recall,total_seconds,epoch,hyper_parameters)
        #                  loss     ,f1         ,precision,recall ,seconds     ,epoch,hp)
#%%

def tested(df,bs,lr,eps)->bool:

    test_done = \
        len(
            df[(df['hyper_parameters']==
            str({'batch_size':bs,'lr':lr,'eps':eps}))&
               (df['epoch']==10)
            ]) ==1
    return test_done

if __name__ == '__main__':
    searched_already = pd.read_csv('/results/bert_frozen_hp_opt.csv')
    hyper_parameters = {'batch_size':[6],
                        'lr' : [1e-3*i for i in (1,11)],
                        'eps' : [-1],
                        }


    for bs in hyper_parameters['batch_size']:
      for lr in hyper_parameters['lr']:
        for eps in hyper_parameters['eps']:
          if tested(searched_already,bs,lr,eps):
            continue
          else:
            train_consistency(bs,lr,eps)

#%%