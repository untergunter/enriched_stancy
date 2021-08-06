import torch

def test_model(model,dataloader,device):
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