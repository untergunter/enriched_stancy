from bert_preprocessing import make_2_kinds_data_set, make_2_kinds_data_set_with_sentiment
from prep import get_paper_train_dev_test
from models import DoubleLoss, DoubleLossSentiment, SingleLossSentiment
from test_model import test_consistency_model, test_consistency_sentiment_model, test_basic_sentiment_model
from transformers import AdamW
import torch
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from datetime import datetime
import pandas as pd
import numpy as np
import os


def train_consistency_with_sentiment(learning_rate, epsilon, epochs_count, batch_size, dropout, bert_version='bert-base-uncased'):

    train, dev, test = get_paper_train_dev_test()
    train_together_only_loader, train_together_and_claim_loader = \
        make_2_kinds_data_set_with_sentiment(train,batch_size, bert_version)
    dev_together_only_loader, dev_together_and_claim_loader = \
        make_2_kinds_data_set_with_sentiment(dev,batch_size, bert_version)
    test_together_only_loader, test_together_and_claim_loader = \
        make_2_kinds_data_set_with_sentiment(test,batch_size, bert_version)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DoubleLossSentiment(device, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon
                      )
    losses = []
    f1s = []
    accuracy_list = []
    recalls = []
    seconds = []
    epochs = []

    # add the basic performance
    y_true, y_pred = test_consistency_sentiment_model(model, dev_together_and_claim_loader, device)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
    precision = precision_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))

    losses.append(-1)
    f1s.append(weighted_f1)
    accuracy_list.append(precision)
    recalls.append(recall)
    seconds.append(-1)
    epochs.append(-1)
    print(f'epoch:{-1}, loss:{-1}, f1:{weighted_f1} ,precision:{precision} ,recall:{recall} ,seconds:{-1}')

    for i in range(epochs_count):
        start_time = datetime.now()
        total_loss = 0
        for batch in train_together_and_claim_loader:
            model.zero_grad()
            together_ids, together_masks, claim_ids, claim_masks, labels, sentiment_labels = batch

            together_ids = together_ids.to(device)
            together_masks = together_masks.to(device)
            claim_ids = claim_ids.to(device)
            claim_masks = claim_masks.to(device)
            labels = labels.to(device)
            sentiments = sentiment_labels.to(device)

            loss = model(
                        together_ids,
                        together_masks,
                        claim_ids,
                        claim_masks,
                        labels,
                        sentiments
                        )
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        y_true, y_pred = test_consistency_sentiment_model(model, dev_together_and_claim_loader, device)
        end_time = datetime.now()
        total_seconds = (end_time-start_time).seconds

        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        seconds.append(total_seconds)
        losses.append(total_loss)
        f1s.append(weighted_f1)
        recalls.append(recall)
        accuracy_list.append(precision)
        epochs.append(i)


        print(f'epoch:{i}, loss:{total_loss}, f1:{weighted_f1} ,precision:{precision} ,recall:{recall} ,seconds:{total_seconds}')


    # add the test set performance
    y_true, y_pred = test_consistency_sentiment_model(model, test_together_and_claim_loader, device)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

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
    results_df.to_csv(f'results/bert_consistency_results_{learning_rate}_{epsilon}_{epochs}_{batch_size}_{dropout}_{bert_version}.csv',index=False)

def train_sentiment(learning_rate, epsilon, epochs_count, batch_size, dropout, bert_version='bert-base-uncased'):

    train, dev, test = get_paper_train_dev_test()
    train_together_only_loader, train_together_and_claim_loader = \
        make_2_kinds_data_set_with_sentiment(train,batch_size, bert_version)
    dev_together_only_loader, dev_together_and_claim_loader = \
        make_2_kinds_data_set_with_sentiment(dev,batch_size, bert_version)
    test_together_only_loader, test_together_and_claim_loader = \
        make_2_kinds_data_set_with_sentiment(test,batch_size, bert_version)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleLossSentiment(device, dropout).to(device)
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      eps=epsilon
                      )
    losses = []
    f1s = []
    accuracy_list = []
    recalls = []
    seconds = []
    epochs = []

    # add the basic performance
    y_true, y_pred = test_basic_sentiment_model(model, dev_together_only_loader, device)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
    precision = precision_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))
    recall = recall_score(y_true, y_pred, average='weighted', labels=np.unique(y_pred))

    losses.append(-1)
    f1s.append(weighted_f1)
    accuracy_list.append(precision)
    recalls.append(recall)
    seconds.append(-1)
    epochs.append(-1)
    print(f'epoch:{-1}, loss:{-1}, f1:{weighted_f1} ,precision:{precision} ,recall:{recall} ,seconds:{-1}')

    for i in range(epochs_count):
        start_time = datetime.now()
        total_loss = 0
        for batch in train_together_only_loader:
            model.zero_grad()
            together_ids, together_masks, labels, sentiment_labels = batch

            together_ids = together_ids.to(device)
            together_masks = together_masks.to(device)
            labels = labels.to(device)
            sentiments = sentiment_labels.to(device)

            loss = model(
                        together_ids,
                        together_masks,
                        labels,
                        sentiments
                        )
            loss.backward()
            total_loss += loss.item()
            optimizer.step()

        y_true, y_pred = test_basic_sentiment_model(model, dev_together_only_loader, device)
        end_time = datetime.now()
        total_seconds = (end_time-start_time).seconds

        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

        seconds.append(total_seconds)
        losses.append(total_loss)
        f1s.append(weighted_f1)
        recalls.append(recall)
        accuracy_list.append(precision)
        epochs.append(i)


        print(f'epoch:{i}, loss:{total_loss}, f1:{weighted_f1} ,precision:{precision} ,recall:{recall} ,seconds:{total_seconds}')


    # add the test set performance
    y_true, y_pred = test_basic_sentiment_model(model, test_together_only_loader, device)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

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
    results_df.to_csv(f'results/bert_consistency_results_{learning_rate}_{epsilon}_{epochs}_{batch_size}_{dropout}_{bert_version}.csv',index=False)


if __name__ == '__main__':

    bert_versions = ['bert-large-uncased', 'bert-base-uncased']
    learning_rates = [0.0000035, 0.000005, 0.000065]
    epsilons = [1e-8, 1e-8 * 10]
    epochss = [8]
    batch_sizes = [12, 16]
    dropouts = [0.1, 0.2, 0.3, 0.4]
    errors = list()
    repeat = True
    while repeat:
        errors = list()
        for learning_rate in learning_rates:
            for epsilon in epsilons:
                for epoch in epochss:
                    for batch_size in batch_sizes:
                        for dropout in dropouts:


                            epochs_s = [-1]
                            for i in range(epoch):
                                epochs_s.append(epoch)
                            epochs_s.append(-2)
                            file_name = f'results/bert_consistency_results_{learning_rate}_{epsilon}_{epochs_s}_{batch_size}_{dropout}.csv'

                            if os.path.exists(file_name):
                                print(f'{file_name}')
                                continue
                            print('Results for:', learning_rate, epsilon, epoch, batch_size, dropout)
                            try:
                                train_consistency_with_sentiment(learning_rate, epsilon, epoch, batch_size, dropout, 'bert-base-uncased')
                            except Exception as e:
                                errors.append(e)
                                print(e)

                            if learning_rate == learning_rates[-1] and epsilon == epsilons[-1] and epoch == epochss[-1] and  batch_size  == batch_sizes[-1] and dropout == dropouts[-1]:
                                if len(errors) == 0:
                                    repeat = False
