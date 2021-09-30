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

#%%

def test_consistency_model(model, dataloader, device):
    y_true = []
    y_pred = []
    # model.eval()
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

  df.to_csv('/home/ido/data/idc/advanced ml/final_project/bert_consistency_hp_opt_1.csv',
            header=False,mode='a',index=False)

#%%

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
        # self.dropout = nn.Dropout(0.1)
        self.similarity_cosine_loss = CosineEmbeddingLoss()
        self.stance_loss_func = CrossEntropyLoss()

    def forward(self, both_ids, both_mask, claim_ids, claim_mask,labels = None):

        both_hs = self.bert(both_ids, attention_mask=both_mask).pooler_output
        # both_hs = self.dropout(both_hs)

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
        _ , probabilities = self.forward(both_ids, both_mask, claim_ids, claim_mask)
        predicted = torch.argmax(probabilities, dim=1)
        return predicted

#%%

def train_consistency(batch_size,lr,eps):
    hyper_parameters = {'batch_size':batch_size,'lr':lr,'eps':eps}

    train, dev, test = get_paper_train_dev_test()
    train_together_only_loader, train_together_and_claim_loader = \
        make_2_kinds_data_set(train,batch_size)
    # dev_together_only_loader, dev_together_and_claim_loader = \
    #     make_2_kinds_data_set(dev,batch_size)
    test_together_only_loader, test_together_and_claim_loader = \
        make_2_kinds_data_set(test,batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoubleLoss(device).to(device)
    optimizer = AdamW(model.parameters(),
                      lr=lr,
                      eps=eps
                      )

    for epoch in range(1,11):
      # train
        start_time = datetime.now()
        total_loss = 0
        #model.train()
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
    searched_already = pd.read_csv('results/bert_consistency_hp_opt_1.csv')
    hyper_parameters = {'batch_size':[4],
                        'lr' : [2e-5*i for i in (0.9,1,1.1)],
                        'eps' : [1e-8*i for i in (0.9,1,1.1)]
                        }


    for bs in hyper_parameters['batch_size']:
      for lr in hyper_parameters['lr']:
        for eps in hyper_parameters['eps']:
          if tested(searched_already,bs,lr,eps):
            continue
          else:
            train_consistency(bs,lr,eps)

#%%