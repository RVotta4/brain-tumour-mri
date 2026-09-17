"""Train a model and record the experiment.

Usage:
    .\\.venv\\Scripts\\python.exe -m src.train --name 01-small-cnn --model small_cnn --epochs 15
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import CLASS_NAMES, DATASET_PATH, SPLITS_PATH, MRIDataset, indices_for_split, load_dataset, load_splits
from src.evaluate import evaluate_model
from src.metrics import confusion_matrix, precision_recall
from src.models import build_model
from src.plots import plot_confusion_matrix, plot_history

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"


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


def write_history(history, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def run_experiment(name, model_name, epochs, batch_size, lr, seed, subset=0,
                   dataset_path=DATASET_PATH, splits_path=SPLITS_PATH, experiments_dir=EXPERIMENTS_DIR):
    """Train on the training patients, pick the best epoch on validation, save the record.

    subset: if above 0, use only this many training and validation scans
            (for a quick check that everything runs).
    """
    set_seed(seed)
    out_dir = Path(experiments_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"name": name, "model": model_name, "epochs": epochs, "batch_size": batch_size,
              "lr": lr, "seed": seed, "subset": subset}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    data = load_dataset(dataset_path)
    assignment = load_splits(splits_path)
    train_idx = indices_for_split(data["patient_ids"], assignment, "train")
    val_idx = indices_for_split(data["patient_ids"], assignment, "validation")
    if subset:
        rng = np.random.default_rng(seed)
        train_idx = np.sort(rng.permutation(train_idx)[:subset])
        val_idx = np.sort(rng.permutation(val_idx)[:subset])

    train_loader = DataLoader(
        MRIDataset(data["images"], data["labels"], train_idx),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(MRIDataset(data["images"], data["labels"], val_idx), batch_size=batch_size)

    model = build_model(model_name, n_classes=len(CLASS_NAMES))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(data["labels"][train_idx], len(CLASS_NAMES)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training {model_name} on {len(train_idx)} scans, validating on {len(val_idx)}")
    history, best_accuracy, best_epoch = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        train_scores = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_scores = evaluate_model(model, val_loader, loss_fn)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_scores["loss"], 4),
            "train_accuracy": round(train_scores["accuracy"], 4),
            "val_loss": round(val_scores["loss"], 4),
            "val_accuracy": round(val_scores["accuracy"], 4),
        })
        write_history(history, out_dir / "history.csv")

        improved = val_scores["accuracy"] > best_accuracy
        if improved:
            best_accuracy, best_epoch = val_scores["accuracy"], epoch
            torch.save(model.state_dict(), out_dir / "model.pt")
        print(f"epoch {epoch:>2}/{epochs}  "
              f"train loss {train_scores['loss']:.3f} acc {train_scores['accuracy']:.1%}  |  "
              f"val loss {val_scores['loss']:.3f} acc {val_scores['accuracy']:.1%}  "
              f"({time.time() - started:.0f}s){'  <- best so far' if improved else ''}")

    model.load_state_dict(torch.load(out_dir / "model.pt", weights_only=True))
    final = evaluate_model(model, val_loader, loss_fn)
    matrix = confusion_matrix(final["y_true"], final["y_pred"], len(CLASS_NAMES))
    per_class = precision_recall(matrix)
    metrics = {
        "split": "validation",
        "best_epoch": best_epoch,
        "n_scans": len(final["y_true"]),
        "accuracy": round(final["accuracy"], 4),
        "loss": round(final["loss"], 4),
        "per_class": {name: {k: round(v, 4) for k, v in per_class[i].items()}
                      for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": matrix,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_history(history, out_dir / "curves.png")
    plot_confusion_matrix(matrix, CLASS_NAMES, out_dir / "confusion_matrix.png")

    print(f"\nBest epoch {best_epoch}: validation accuracy {final['accuracy']:.1%}")
    for class_name, scores in metrics["per_class"].items():
        print(f"  {class_name:<11} precision {scores['precision']:.1%}  recall {scores['recall']:.1%}")
    print(f"Saved to {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a brain tumour classifier.")
    parser.add_argument("--name", required=True, help="experiment folder name, e.g. 01-small-cnn")
    parser.add_argument("--model", default="small_cnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate: how big each nudge is")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=0, help="use only this many scans (quick check)")
    args = parser.parse_args()
    run_experiment(args.name, args.model, args.epochs, args.batch_size, args.lr, args.seed, args.subset)


if __name__ == "__main__":
    main()
