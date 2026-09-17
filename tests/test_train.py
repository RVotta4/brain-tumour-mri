import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate_model
from src.models import SmallCNN
from src.train import class_weights, set_seed, train_one_epoch


def tiny_learnable_batch():
    """8 images, 32x32: each class has a bright square in its own row band."""
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    images = torch.rand(8, 1, 32, 32) * 0.1
    for i, label in enumerate(labels.tolist()):
        top = 2 + 10 * label
        images[i, 0, top:top + 8, 12:20] = 1.0
    return images, labels


def test_class_weights_balance_rare_classes():
    weights = class_weights([0, 0, 0, 1], n_classes=2)

    assert weights.tolist() == pytest.approx([4 / 6, 2.0])


def test_class_weights_reject_a_missing_class():
    with pytest.raises(ValueError, match="at least one"):
        class_weights([0, 0, 1], n_classes=3)


def test_model_can_memorise_a_tiny_batch():
    set_seed(0)
    images, labels = tiny_learnable_batch()
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    model = SmallCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    first = train_one_epoch(model, loader, loss_fn, optimizer)["loss"]
    for _ in range(59):
        last = train_one_epoch(model, loader, loss_fn, optimizer)["loss"]

    assert last < first * 0.5


def test_evaluate_model_reports_predictions_and_confidences():
    set_seed(0)
    images, labels = tiny_learnable_batch()
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    result = evaluate_model(SmallCNN(), loader, nn.CrossEntropyLoss())

    assert result["y_true"] == labels.tolist()
    assert len(result["y_pred"]) == 8
    assert all(0.0 <= c <= 1.0 for c in result["confidences"])
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["loss"] > 0
