import torch

def test_basic_model(model, dataloader, device):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in dataloader:
            ids = batch[0].to(device)
            masks = batch[1].to(device)
            labels = batch[2].to(device)
            model_out = model(ids,
                              token_type_ids=None,
                              attention_mask=masks,
                              labels=labels
                              )
            class_predicted = torch.argmax(model_out['logits'], dim=1)
            y_true += [int(label) for label in labels]
            y_pred += [int(label) for label in class_predicted]
    return y_true,y_pred


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




def test_consistency_sentiment_model(model, dataloader, device):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            together_ids, together_masks, claim_ids, claim_masks, labels, sentiment_labels = batch
            together_ids = together_ids.to(device)
            together_masks = together_masks.to(device)
            claim_ids = claim_ids.to(device)
            claim_masks = claim_masks.to(device)
            labels = labels.to(device)
            sentiments = sentiment_labels.to(device)


            model_prediction = model.predict(together_ids,
                              together_masks,
                              claim_ids,
                              claim_masks,
                              sentiment_labels=sentiments
                              )
            y_true += [int(label) for label in labels]
            y_pred += [int(label) for label in model_prediction]
    return y_true,y_pred


def test_basic_sentiment_model(model, dataloader, device):
    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            ids, masks, labels, sentiment_labels = batch
            ids = ids.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            sentiments = sentiment_labels.to(device)

            model_prediction = model.predict(ids,
                              masks,
                              sentiment_labels=sentiments
                              )
            y_true += [int(label) for label in labels]
            y_pred += [int(label) for label in model_prediction]

    return y_true, y_pred