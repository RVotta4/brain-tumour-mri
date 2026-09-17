"""Scoring a model on a set of scans, without learning from them."""

import torch

from src.metrics import accuracy


@torch.no_grad()
def evaluate_model(model, loader, loss_fn):
    """Run the model over every batch and collect its answers.

    Returns average loss, accuracy, and per-scan lists of true labels,
    predictions and confidences (the probability the model gave to the
    class it chose).
    """
    model.eval()
    total_loss = 0.0
    y_true, y_pred, confidences = [], [], []
    for images, labels in loader:
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * len(labels)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)
        y_true += labels.tolist()
        y_pred += predicted.tolist()
        confidences += confidence.tolist()
    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confidences,
    }
