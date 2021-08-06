import pandas as pd

from bert_preprocesing import make_2_kinds_data_set
from prep import get_paper_train_dev_test
from test_model import test_model
from transformers import BertForSequenceClassification,AdamW
import torch
from sklearn.metrics import f1_score,accuracy_score
from datetime import datetime

def train_base():
    train, dev, test = get_paper_train_dev_test()
    train_together_only_loader, train_together_and_claim_loader = \
        make_2_kinds_data_set(train)
    dev_together_only_loader, dev_together_and_claim_loader = \
        make_2_kinds_data_set(dev)
    test_together_only_loader, test_together_and_claim_loader = \
        make_2_kinds_data_set(test)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=3,
        output_attentions=False,
        output_hidden_states=False
    )
    model.to(device)

    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )
    losses = []
    f1s = []
    accuracy_list = []
    seconds = []
    epochs = []

    # add the basic performance
    y_true, y_pred = test_model(model, dev_together_only_loader, device)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    losses.append(-1)
    f1s.append(weighted_f1)
    accuracy_list.append(accuracy)
    seconds.append(-1)
    epochs.append(-1)
    print(f'epoch:{-1}, loss:{-1}, f1:{weighted_f1} ,accuracy:{accuracy} ,seconds:{-1}')

    for epoch in range(10):
        start_time = datetime.now()
        total_loss = 0
        for batch in train_together_only_loader:
            model.zero_grad()
            ids = batch[0].to(device)
            masks = batch[1].to(device)
            labels = batch[2].to(device)
            model_out = model(ids,
                              token_type_ids=None,
                              attention_mask=masks,
                              labels=labels
                              )
            loss = model_out.loss
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        y_true, y_pred = test_model(model,dev_together_only_loader,device)
        end_time = datetime.now()
        total_seconds = (end_time-start_time).seconds

        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        accuracy = accuracy_score(y_true, y_pred)

        seconds.append(total_seconds)
        losses.append(total_loss)
        f1s.append(weighted_f1)
        accuracy_list.append(accuracy)
        epochs.append(epoch)


        print(f'epoch:{epoch}, loss:{total_loss}, f1:{weighted_f1} ,accuracy:{accuracy} ,seconds:{total_seconds}')


    # add the basic performance
    y_true, y_pred = test_model(model, test_together_only_loader, device)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    accuracy = accuracy_score(y_true, y_pred)

    losses.append(-2)
    f1s.append(weighted_f1)
    accuracy_list.append(accuracy)
    seconds.append(-2)
    epochs.append(-2)
    print(f'epoch:{-2}, loss:{-2}, f1:{weighted_f1} ,accuracy:{accuracy} ,seconds:{-2}')

    results_df = pd.DataFrame({"epoch":epochs,'loss':losses,'f1':f1s,'accuracy':accuracy_list,'seconds':total_seconds})
    results_df.to_csv('results/base_bert_results.csv',index=False)

if __name__ == '__main__':
    train_base()