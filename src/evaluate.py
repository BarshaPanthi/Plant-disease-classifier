
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def evaluate(model, loader, device):
    """
    Runs model on a dataloader and returns predictions and true labels.
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def print_metrics(preds, labels):
    """
    Prints accuracy, precision, recall and F1 score.
    """
    print(f"Accuracy  : {accuracy_score(labels, preds)*100:.2f}%")
    print(f"Precision : {precision_score(labels, preds)*100:.2f}%")
    print(f"Recall    : {recall_score(labels, preds)*100:.2f}%")
    print(f"F1 Score  : {f1_score(labels, preds)*100:.2f}%")
