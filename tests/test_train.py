import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate_model
from src.models import SmallCNN
from src.train import best_loss_epoch, class_weights, set_seed, train_one_epoch


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


import json

import numpy as np

from src.data import save_splits, split_patients
from src.train import run_experiment


def write_tiny_dataset(folder):
    """12 patients (4 per class), 3 slices each, 32x32 images with a class-specific square."""
    rng = np.random.default_rng(0)
    images, labels, patient_ids = [], [], []
    for label in range(3):
        for patient in range(4):
            for _ in range(3):
                image = (rng.random((32, 32)) * 30).astype(np.uint8)
                top = 2 + 10 * label
                image[top:top + 8, 12:20] = 255
                images.append(image)
                labels.append(label)
                patient_ids.append(f"{label}{patient}")
    dataset_path = folder / "tiny.npz"
    np.savez_compressed(
        dataset_path,
        images=np.stack(images),
        masks=np.zeros((len(images), 32, 32), dtype=np.uint8),
        labels=np.array(labels, dtype=np.int64),
        patient_ids=np.array(patient_ids),
    )
    splits_path = folder / "splits.csv"
    save_splits(split_patients(patient_ids, labels, seed=42), splits_path)
    return dataset_path, splits_path


def test_run_experiment_writes_a_complete_record(tmp_path):
    dataset_path, splits_path = write_tiny_dataset(tmp_path)

    metrics = run_experiment(
        name="tiny", model_name="small_cnn", epochs=2, batch_size=4, lr=1e-3, seed=42,
        dataset_path=dataset_path, splits_path=splits_path, experiments_dir=tmp_path / "experiments",
    )

    folder = tmp_path / "experiments" / "tiny"
    for filename in ["config.json", "history.csv", "metrics.json", "model.pt", "curves.png", "confusion_matrix.png"]:
        assert (folder / filename).exists(), filename
    assert metrics["split"] == "validation"
    assert metrics["n_scans"] == 9  # 1 validation patient per class x 3 slices
    assert set(metrics["per_class"]) == {"meningioma", "glioma", "pituitary"}
    assert json.loads((folder / "config.json").read_text())["epochs"] == 2
    assert len((folder / "history.csv").read_text().strip().splitlines()) == 3  # header + 2 epochs


def test_best_loss_epoch_picks_the_lowest_validation_loss():
    history = [
        {"epoch": 1, "val_loss": 1.2},
        {"epoch": 2, "val_loss": 0.6},
        {"epoch": 3, "val_loss": 0.9},
    ]

    assert best_loss_epoch(history) == 2


def test_best_loss_epoch_keeps_the_earliest_on_a_tie():
    history = [
        {"epoch": 1, "val_loss": 0.5},
        {"epoch": 2, "val_loss": 0.5},
    ]

    assert best_loss_epoch(history) == 1
