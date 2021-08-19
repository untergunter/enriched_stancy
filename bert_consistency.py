from bert_preprocesing import make_2_kinds_data_set
from prep import get_paper_train_dev_test
from models import DoubleLoss
from test_model import test_consistency_model
import torch
from transformers import AdamW
import torch
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from datetime import datetime
import pandas as pd


def train_consistency():

    train, dev, test = get_paper_train_dev_test()
    train_together_only_loader, train_together_and_claim_loader = \
        make_2_kinds_data_set(train,12)
    dev_together_only_loader, dev_together_and_claim_loader = \
        make_2_kinds_data_set(dev,12)
    test_together_only_loader, test_together_and_claim_loader = \
        make_2_kinds_data_set(test,12)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoubleLoss(device).to(device)
    optimizer = AdamW(model.parameters(),
                      lr=2e-5,
                      eps=1e-8
                      )
    losses = []
    f1s = []
    accuracy_list = []
    recalls = []
    seconds = []
    epochs = []

    # add the basic performance
    y_true, y_pred = test_consistency_model(model, dev_together_and_claim_loader, device)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    losses.append(-1)
    f1s.append(weighted_f1)
    accuracy_list.append(precision)
    recalls.append(recall)
    seconds.append(-1)
    epochs.append(-1)
    print(f'epoch:{-1}, loss:{-1}, f1:{weighted_f1} ,precision:{precision} ,recall:{recall} ,seconds:{-1}')

    for epoch in range(8):
        start_time = datetime.now()
        total_loss = 0
        for batch in train_together_and_claim_loader:
            model.zero_grad()
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
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        y_true, y_pred = test_consistency_model(model, dev_together_and_claim_loader, device)
        end_time = datetime.now()
        total_seconds = (end_time-start_time).seconds

        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        seconds.append(total_seconds)
        losses.append(total_loss)
        f1s.append(weighted_f1)
        recalls.append(recall)
        accuracy_list.append(precision)
        epochs.append(epoch)


        print(f'epoch:{epoch}, loss:{total_loss}, f1:{weighted_f1} ,precision:{precision} ,recall:{recall} ,seconds:{total_seconds}')


    # add the test set performance
    y_true, y_pred = test_consistency_model(model, test_together_and_claim_loader, device)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    losses.append(-2)
    f1s.append(weighted_f1)
    accuracy_list.append(precision)
    recalls.append(recall)
    seconds.append(-2)
    epochs.append(-2)
    print(f'epoch:{-2}, loss:{-2}, f1:{weighted_f1} ,precision:{precision}  ,recall:{recall} ,seconds:{-2}')

    results_df = pd.DataFrame({"epoch":epochs,
                               'loss':losses,
                               'f1':f1s,
                               'precision':accuracy_list,
                               'recall':recalls,
                               'seconds':total_seconds})
    results_df.to_csv('results/bert_consistency_results.csv',index=False)

if __name__ == '__main__':
    train_consistency()