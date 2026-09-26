"""Score a saved experiment's model on the validation or test patients.

Usage:
    .\\.venv\\Scripts\\python.exe -m src.score --name 03-resnet18-finetuned --split validation
    .\\.venv\\Scripts\\python.exe -m src.score --name 03-resnet18-finetuned --split test

The test set is the final exam. Every decision so far was made on
validation, so only the test patients give an honest score, and only the
first time: re-scoring after adjusting anything would turn them into a
second validation set. So a test score is allowed once in the whole
project, and this module refuses a second one.
"""

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import CLASS_NAMES, DATASET_PATH, SPLITS_PATH, MRIDataset, indices_for_split, load_dataset, load_splits
from src.evaluate import evaluate_model
from src.metrics import confusion_matrix, precision_recall
from src.models import build_model
from src.plots import plot_confusion_matrix
from src.train import EXPERIMENTS_DIR, class_weights

TEST_RESULTS = "test_metrics.json"


def check_test_set_unused(experiments_dir):
    """Refuse if any experiment has already been scored on the test set."""
    used = sorted(Path(experiments_dir).glob(f"*/{TEST_RESULTS}"))
    if used:
        raise RuntimeError(
            f"the test set has already been used ({used[0]}). It is scored once per project; "
            "a second score would no longer be an honest estimate."
        )


def score_experiment(name, split, batch_size=32, dataset_path=DATASET_PATH, splits_path=SPLITS_PATH,
                     experiments_dir=EXPERIMENTS_DIR):
    """Reload an experiment's saved model and score it on one split.

    split="validation": prints and returns the scores, writes nothing. Used
        to confirm the saved model reloads and matches its metrics.json.
    split="test": allowed once per project. Also writes test_metrics.json
        and test_confusion_matrix.png into the experiment's folder.
    """
    if split not in ("validation", "test"):
        raise ValueError(f"split must be 'validation' or 'test', got {split!r}")
    if split == "test":
        check_test_set_unused(experiments_dir)

    folder = Path(experiments_dir) / name
    config = json.loads((folder / "config.json").read_text())
    # No pretrained download needed: model.pt replaces every weight.
    model = build_model(config["model"], n_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(folder / "model.pt", weights_only=True))

    data = load_dataset(dataset_path)
    assignment = load_splits(splits_path)
    train_idx = indices_for_split(data["patient_ids"], assignment, "train")
    split_idx = indices_for_split(data["patient_ids"], assignment, split)
    loader = DataLoader(MRIDataset(data["images"], data["labels"], split_idx), batch_size=batch_size)
    # Same class-weighted loss as training, so the loss figure is comparable.
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(data["labels"][train_idx], len(CLASS_NAMES)))

    result = evaluate_model(model, loader, loss_fn)
    matrix = confusion_matrix(result["y_true"], result["y_pred"], len(CLASS_NAMES))
    per_class = precision_recall(matrix)
    metrics = {
        "split": split,
        "experiment": name,
        "n_scans": len(result["y_true"]),
        "accuracy": round(result["accuracy"], 4),
        "loss": round(result["loss"], 4),
        "per_class": {class_name: {k: round(v, 4) for k, v in per_class[i].items()}
                      for i, class_name in enumerate(CLASS_NAMES)},
        "confusion_matrix": matrix,
    }
    if split == "test":
        (folder / TEST_RESULTS).write_text(json.dumps(metrics, indent=2))
        plot_confusion_matrix(matrix, CLASS_NAMES, folder / "test_confusion_matrix.png")

    print(f"{name} on {split}: accuracy {metrics['accuracy']:.1%} over {metrics['n_scans']} scans")
    for class_name, scores in metrics["per_class"].items():
        print(f"  {class_name:<11} precision {scores['precision']:.1%}  recall {scores['recall']:.1%}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Score a saved experiment on validation or test.")
    parser.add_argument("--name", required=True, help="experiment folder name, e.g. 03-resnet18-finetuned")
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    args = parser.parse_args()
    score_experiment(args.name, args.split)


if __name__ == "__main__":
    main()
