from prep import get_paper_train_dev_test
from RnnDataSet import PosRnnDataSet
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,RandomSampler
from models import Rnn,RnnClaimPerspective
from torch.optim import Adam
from sklearn.metrics import classification_report
from tqdm import tqdm

if __name__ == '__main__':
    train,dev,_ = get_paper_train_dev_test()

    train_set = PosRnnDataSet(train,perspective_only=False)
    train_loader = DataLoader(train_set,sampler=RandomSampler(train_set))

    dev_set = PosRnnDataSet(dev)
    dev_loader = DataLoader(dev_set)
    model = RnnClaimPerspective()

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    for epoch in range(10):

        for embeddings, label in tqdm(train_loader):
            model.reset_hidden_layer()
            for embedding in embeddings:
                out = model(embedding)
            out = out.reshape(1,2)
            label = label.flatten()

            optimizer.zero_grad()
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            predictions,true_labels = [],[]
            for embeddings, label in dev_loader:
                model.reset_hidden_layer()
                for embedding in embeddings:
                    out = model(embedding)
                out = out.reshape(1, 2)
                label = label.flatten()
                predicted = torch.argmax(out,dim=1)
                predictions.append(predicted.item())
                true_labels.append(label.item())

            print(classification_report(true_labels,predictions))