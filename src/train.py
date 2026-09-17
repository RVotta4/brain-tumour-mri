"""Train a model and record the experiment."""

import random

import numpy as np
import torch


def set_seed(seed):
    """Fix every source of randomness so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(labels, n_classes):
    """Loss weights that make rare tumour types count as much as common ones.

    weight = total / (n_classes * count). A perfectly balanced dataset gets
    1.0 for every class; a class half as common gets double weight.
    """
    counts = np.bincount(np.asarray(labels), minlength=n_classes)
    if (counts == 0).any():
        raise ValueError(f"every class needs at least one example, got counts {counts.tolist()}")
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float32)


def train_one_epoch(model, loader, loss_fn, optimizer):
    """One full pass over the training data. For every batch:

    1. guess:   the model scores each tumour type
    2. measure: the loss says how wrong those scores were
    3. blame:   backward() works out how each weight contributed to the error
    4. adjust:  the optimiser nudges every weight to reduce the error
    """
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return {"loss": total_loss / seen, "accuracy": correct / seen}
