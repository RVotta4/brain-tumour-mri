# Stage 3: Augmentation and the Test-Set Score â€” Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add training-only data augmentation, run experiment 4, choose the final model by a rule fixed in advance, and score that model exactly once on the untouched test patients.

**Architecture:** `src/data.py` gains `train_transform()` (torchvision `transforms.v2`) and an optional `transform` on `MRIDataset`. `src/train.py` gains an `--augment` flag that applies it to the training data only and saves a preview grid drawn by a new `src/plots.py` function. A new module, `src/score.py`, reloads a saved experiment and scores it on validation or test, with a guard that refuses a second test-set score anywhere in the project.

**Tech Stack:** Python 3.13 (Windows, `py` launcher), PyTorch 2.14 CPU, torchvision 0.29, NumPy, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-26-stage-3-augmentation-and-test-design.md`. Overall design: `docs/superpowers/specs/2026-09-17-brain-tumour-mri-design.md`.

---

## Ground rules for whoever executes this

- **Platform:** Windows, PowerShell. All commands run from `C:\Users\Robbi\brain-tumour-mri`.
- **Always call the venv's Python directly:** `.\.venv\Scripts\python.exe`. Bare `python` on this machine opens the Microsoft Store.
- **Robbi is a beginner and learning.** Each task starts with a **Concept** line â€” explain that concept in plain English before writing the code.
- **Code tasks (1â€“5) run straight through without check-ins.** The two marked PAUSES (Tasks 8 and 10) stop for Robbi.
- **Robbi runs the training and scoring commands** (Tasks 6, 7 and 9). Don't run them on Robbi's behalf. If the laptop sleeps mid-run, training pauses and resumes; it does not need restarting.
- **Commit messages** end with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (passed as a second `-m`).
- **Verified facts as of 2026-09-26:** `torch 2.14.0+cpu` and `torchvision 0.29.0+cpu` installed; torchvision `v2` transforms work on (1, 224, 224) float tensors; 40 tests passing; `main` holds the spec commit `241fd3e` (not yet pushed). ResNet-18 epochs take about 3.5 minutes.
- **The final-model rule is fixed** (spec section 2): experiment 4 is the final model only if its `metrics.json` `accuracy` is **0.95 or higher**; otherwise experiment 3 is. Do not revisit it after seeing results.

## File map

| File | Change |
|---|---|
| `src/data.py` | Add `train_transform()` and `clip_to_unit_range`; `MRIDataset` takes optional `transform` |
| `src/plots.py` | Add `plot_augmentation` |
| `src/train.py` | `augment` argument and `--augment` flag; preview after training |
| `src/score.py` | New: `check_test_set_unused`, `score_experiment`, command line |
| `tests/test_data.py` | Transform and dataset tests |
| `tests/test_plots.py` | Preview grid test |
| `tests/test_train.py` | Augmented run test |
| `tests/test_score.py` | New: guard and scoring tests |
| `README.md`, `docs/learning-notes.md` | Experiment 4, final test results, notes 14â€“16 |

---

### Task 1: Branch

- [ ] **Step 1: Create the branch**

```powershell
git checkout -b feat/stage-3-augmentation
```
Expected: `Switched to a new branch 'feat/stage-3-augmentation'`.

---

### Task 2: The augmentation transform

**Concept:** data augmentation. Experiment 3 reached 100% training accuracy by epoch 5 â€” it memorised the 2,100 training scans. Augmentation shows a slightly different version of each scan every time it is seen (mirrored, tilted a few degrees, a little brighter or darker), so memorising exact pixels stops working and the model has to learn what the tumour looks like. Every change must be something a real scanner could produce, which is why there are no upside-down flips.

**Files:**
- Modify: `src/data.py`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:

```python
from src.data import train_transform


def test_train_transform_keeps_shape_and_range():
    torch.manual_seed(0)
    transform = train_transform()
    image = torch.rand(1, 224, 224)

    for _ in range(20):
        out = transform(image)
        assert out.shape == (1, 224, 224)
        assert out.min() >= 0.0 and out.max() <= 1.0


def test_train_transform_changes_the_image():
    torch.manual_seed(0)
    image = torch.rand(1, 64, 64)

    out = train_transform()(image)

    assert not torch.equal(out, image)


def test_dataset_applies_its_transform():
    images = np.full((1, 8, 8), 255, dtype=np.uint8)
    labels = np.array([1], dtype=np.int64)
    dataset = MRIDataset(images, labels, indices=np.array([0]), transform=lambda x: x * 0.5)

    image, label = dataset[0]

    assert torch.allclose(image, torch.full((1, 8, 8), 0.5))
    assert label == 1
```

- [ ] **Step 2: Run them to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_data.py -v
```
Expected: `ImportError: cannot import name 'train_transform' from 'src.data'`.

- [ ] **Step 3: Implement**

In `src/data.py`, add to the imports after `from torch.utils.data import Dataset`:

```python
from torchvision.transforms import v2
```

Add these two functions directly before `class MRIDataset`:

```python
def clip_to_unit_range(image):
    """Keep every pixel within [0, 1] after brightness and contrast changes."""
    return image.clamp(0.0, 1.0)


def train_transform():
    """Random, realistic changes applied to training scans only.

    Each time a scan is served it is changed slightly, so the model never
    sees exactly the same image twice and cannot simply memorise pixels:
      - mirrored left-right half the time (a mirrored brain is still a
        realistic brain; upside-down never comes out of a scanner, so no
        vertical flips);
      - rotated by up to 10 degrees either way, like a small head tilt,
        with the empty corners filled black to match the scan background;
      - brightness and contrast changed by up to 10%, like a different
        scanner or setting.
    """
    return v2.Compose([
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomRotation(degrees=10),
        v2.ColorJitter(brightness=0.1, contrast=0.1),
        v2.Lambda(clip_to_unit_range),
    ])
```

Replace the whole `MRIDataset` class with:

```python
class MRIDataset(Dataset):
    """Serves (image tensor, label) pairs for the slices at the given indices.

    Images come out as float32 in [0, 1] with shape (1, height, width):
    one channel because MRI slices are greyscale. If a transform is given
    (training only), it is applied to each image as it is served.
    """

    def __init__(self, images, labels, indices, transform=None):
        self.images = images
        self.labels = labels
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        i = self.indices[position]
        image = torch.from_numpy(self.images[i].astype(np.float32) / 255.0).unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        return image, int(self.labels[i])
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_data.py -v
```
Expected: all pass, including the three new ones and the existing `test_dataset_returns_scaled_single_channel_tensor_and_label` (unchanged behaviour without a transform).

- [ ] **Step 5: Commit**

```powershell
git add src/data.py tests/test_data.py
git commit -m "feat: training-only augmentation transform" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: The augmentation preview

**Concept:** check the augmentation by eye. A transform that is too strong (scans rotated to nonsense, brightness blown out) would teach the model the wrong thing, and no number reveals that as quickly as looking at the pictures.

**Files:**
- Modify: `src/plots.py`
- Test: `tests/test_plots.py`

- [ ] **Step 1: Write the failing test**

Change the import line at the top of `tests/test_plots.py` to:

```python
from src.plots import plot_augmentation, plot_confusion_matrix, plot_history
```

Append:

```python
def test_plot_augmentation_writes_png(tmp_path):
    import numpy as np

    images = [np.random.default_rng(i).random((16, 16)) for i in range(8)]
    path = tmp_path / "augmentation.png"

    plot_augmentation(images, path)

    assert path.read_bytes()[:4] == b"\x89PNG"
```

- [ ] **Step 2: Run it to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_plots.py -v
```
Expected: `ImportError: cannot import name 'plot_augmentation' from 'src.plots'`.

- [ ] **Step 3: Implement**

Append to `src/plots.py`:

```python
def plot_augmentation(images, path):
    """One scan and its augmented versions in a 2x4 grid.

    images[0] is the original; the rest are the same scan after the random
    training changes. Used to check by eye that augmentation looks realistic.
    """
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for index, (ax, image) in enumerate(zip(axes.flat, images)):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title("original" if index == 0 else f"augmented {index}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run the tests to verify they pass**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_plots.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/plots.py tests/test_plots.py
git commit -m "feat: augmentation preview grid" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The --augment flag

**Concept:** augment training, never validation. Validation is the practice exam; if its scans were randomly altered, the score would change from run to run for reasons that have nothing to do with the model, and comparisons with experiments 1â€“3 would break.

**Files:**
- Modify: `src/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_train.py`:

```python
def test_run_experiment_with_augmentation_records_it_and_saves_a_preview(tmp_path):
    dataset_path, splits_path = write_tiny_dataset(tmp_path)

    run_experiment(
        name="aug", model_name="small_cnn", epochs=1, batch_size=4, lr=1e-3, seed=42, augment=True,
        dataset_path=dataset_path, splits_path=splits_path, experiments_dir=tmp_path / "experiments",
    )

    folder = tmp_path / "experiments" / "aug"
    assert json.loads((folder / "config.json").read_text())["augment"] is True
    assert (folder / "augmentation.png").exists()
```

- [ ] **Step 2: Run it to verify it fails**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_train.py::test_run_experiment_with_augmentation_records_it_and_saves_a_preview -v
```
Expected: FAIL with `TypeError: run_experiment() got an unexpected keyword argument 'augment'`.

- [ ] **Step 3: Implement**

In `src/train.py`, change the `src.data` import line to:

```python
from src.data import (CLASS_NAMES, DATASET_PATH, SPLITS_PATH, MRIDataset, indices_for_split, load_dataset,
                      load_splits, train_transform)
```

and the `src.plots` import line to:

```python
from src.plots import plot_augmentation, plot_confusion_matrix, plot_history
```

Change the `run_experiment` signature and docstring to:

```python
def run_experiment(name, model_name, epochs, batch_size, lr, seed, subset=0, augment=False,
                   dataset_path=DATASET_PATH, splits_path=SPLITS_PATH, experiments_dir=EXPERIMENTS_DIR):
    """Train on the training patients, pick the best epoch on validation, save the record.

    subset:  if above 0, use only this many training and validation scans
             (for a quick check that everything runs).
    augment: if True, training scans get random realistic changes each time
             they are served. Validation scans are never changed.
    """
```

Change the `config = {...}` line to:

```python
    config = {"name": name, "model": model_name, "epochs": epochs, "batch_size": batch_size,
              "lr": lr, "seed": seed, "subset": subset, "augment": augment}
```

Replace the `train_loader = DataLoader(...)` statement with:

```python
    train_loader = DataLoader(
        MRIDataset(data["images"], data["labels"], train_idx, transform=train_transform() if augment else None),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
```

Directly after the line `plot_confusion_matrix(matrix, CLASS_NAMES, out_dir / "confusion_matrix.png")`, add:

```python
    if augment:
        # Drawn after training, so its random draws cannot change the run itself.
        original = MRIDataset(data["images"], data["labels"], train_idx)[0][0]
        transform = train_transform()
        versions = [original] + [transform(original) for _ in range(7)]
        plot_augmentation([version[0].numpy() for version in versions], out_dir / "augmentation.png")
```

In `main()`, add after the `--subset` argument:

```python
    parser.add_argument("--augment", action="store_true", help="randomly flip, rotate and adjust training scans")
```

and replace the `run_experiment(...)` call with:

```python
    run_experiment(args.name, args.model, args.epochs, args.batch_size, args.lr, args.seed,
                   subset=args.subset, augment=args.augment)
```

Update the module docstring's usage example to show both forms:

```python
"""Train a model and record the experiment.

Usage:
    .\\.venv\\Scripts\\python.exe -m src.train --name 01-small-cnn --model small_cnn --epochs 15
    .\\.venv\\Scripts\\python.exe -m src.train --name 04-resnet18-augment --model resnet18 --lr 1e-4 --augment
"""
```

- [ ] **Step 4: Run the full suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: 45 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/train.py tests/test_train.py
git commit -m "feat: --augment flag for training-only augmentation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Scoring a saved model, with the one-shot test guard

**Concept:** the test set is the final exam. Every decision so far (which epoch, which learning rate, which model) was made by looking at validation scores, so validation scores are now slightly flattering. The test patients have never influenced anything, which is the only reason their score is honest â€” and it stays honest only if it is used once. Peeking, adjusting and re-scoring would quietly turn it into a second validation set. The guard makes that impossible by accident.

**Files:**
- Create: `src/score.py`
- Test: `tests/test_score.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_score.py`:

```python
import json

import pytest

from src.score import TEST_RESULTS, check_test_set_unused, score_experiment
from src.train import run_experiment
from tests.test_train import write_tiny_dataset


def trained_tiny_experiment(tmp_path):
    dataset_path, splits_path = write_tiny_dataset(tmp_path)
    experiments_dir = tmp_path / "experiments"
    metrics = run_experiment(
        name="tiny", model_name="small_cnn", epochs=1, batch_size=4, lr=1e-3, seed=42,
        dataset_path=dataset_path, splits_path=splits_path, experiments_dir=experiments_dir,
    )
    paths = {"dataset_path": dataset_path, "splits_path": splits_path, "experiments_dir": experiments_dir}
    return metrics, paths


def test_guard_allows_an_unused_test_set(tmp_path):
    (tmp_path / "01-anything").mkdir()

    check_test_set_unused(tmp_path)


def test_guard_refuses_once_any_experiment_has_test_results(tmp_path):
    (tmp_path / "01-anything").mkdir()
    (tmp_path / "01-anything" / TEST_RESULTS).write_text("{}")

    with pytest.raises(RuntimeError, match="already been used"):
        check_test_set_unused(tmp_path)


def test_scoring_on_test_writes_results_and_then_locks(tmp_path):
    _, paths = trained_tiny_experiment(tmp_path)

    metrics = score_experiment("tiny", "test", **paths)

    folder = paths["experiments_dir"] / "tiny"
    assert metrics["split"] == "test"
    assert metrics["n_scans"] == 9  # 1 test patient per class x 3 slices
    assert json.loads((folder / TEST_RESULTS).read_text())["split"] == "test"
    assert (folder / "test_confusion_matrix.png").exists()
    with pytest.raises(RuntimeError, match="already been used"):
        score_experiment("tiny", "test", **paths)


def test_rescoring_validation_reproduces_training_result_and_writes_nothing(tmp_path):
    trained, paths = trained_tiny_experiment(tmp_path)

    rescored = score_experiment("tiny", "validation", **paths)

    assert rescored["accuracy"] == trained["accuracy"]
    assert not (paths["experiments_dir"] / "tiny" / TEST_RESULTS).exists()


def test_unknown_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="split"):
        score_experiment("tiny", "train", experiments_dir=tmp_path)
```

- [ ] **Step 2: Run them to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_score.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.score'`.

- [ ] **Step 3: Implement**

Create `src/score.py`:

```python
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
```

- [ ] **Step 4: Run the full suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: 50 passed (51 after the final-review fixes). The suite must not create `test_metrics.json` anywhere under the real `experiments/` folder; check with:

```powershell
Get-ChildItem experiments -Recurse -Filter test_metrics.json
```
Expected: no output.

- [ ] **Step 5: Commit**

```powershell
git add src/score.py tests/test_score.py
git commit -m "feat: score a saved model, with a one-shot test-set guard" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Smoke run and preview check (Robbi runs this)

**Concept:** look before spending an hour. A 1-epoch run on 64 scans proves the augmented pipeline works and produces the preview grid to inspect.

- [ ] **Step 1: Run it**

```powershell
.\.venv\Scripts\python.exe -m src.train --name smoke --model resnet18 --epochs 1 --subset 64 --lr 1e-4 --augment
```
Expected: one epoch line, per-class scores (meaningless at this size), and `Saved to ...experiments\smoke`. `experiments/smoke/` is gitignored.

- [ ] **Step 2: Inspect the preview with Robbi**

Open `experiments/smoke/augmentation.png` (use the Read tool to view it, and have Robbi open it too). Check: every version is a plausible scan, rotations are small, nothing is upside down, brightness changes are subtle. If anything looks unrealistic, stop and discuss before Task 7 â€” changing augmentation strength would be a spec change.

---

### Task 7: Run experiment 4 (Robbi runs this)

**Concept:** one change only. Same model, same learning rate, same 15 epochs and seed as experiment 3; the only difference is `--augment`. Expect training accuracy to rise more slowly than in experiment 3 â€” that is augmentation working, not failing.

- [ ] **Step 1: Run it**

```powershell
.\.venv\Scripts\python.exe -m src.train --name 04-resnet18-augment --model resnet18 --epochs 15 --lr 1e-4 --augment
```
Expected: 15 epoch lines, about an hour in total with the laptop awake.

- [ ] **Step 2: Commit the experiment record**

```powershell
git add experiments/04-resnet18-augment
git commit -m "exp: 04 ResNet-18 fine-tuned with augmentation" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected: `config.json`, `history.csv`, `metrics.json`, `curves.png`, `confusion_matrix.png` and `augmentation.png` committed; `model.pt` excluded by `.gitignore`.

---

### Task 8: â¸ PAUSE 1 â€” review experiment 4 and apply the rule

- [ ] **Step 1: Build the comparison**

From `experiments/03-resnet18-finetuned/` and `experiments/04-resnet18-augment/` (`metrics.json`, `history.csv`), tabulate: validation accuracy, per-class precision and recall, best-accuracy epoch, best-loss epoch, spread (max minus min validation accuracy), and final-epoch training accuracy (the overfitting signal).

- [ ] **Step 2: Apply the rule and discuss with Robbi**

State the rule's outcome mechanically first: experiment 4's `accuracy` is â‰¥ 0.95 â†’ final model is `04-resnet18-augment`; otherwise â†’ `03-resnet18-finetuned`. Then cover with Robbi, using both `curves.png` files:
- Did augmentation reduce overfitting (training accuracy, gap between training and validation loss)?
- Did it change meningioma precision (experiment 3's weak spot, 0.82)?
- Anything surprising?

Do not continue until Robbi agrees on the final model. If augmentation helped overfitting but missed the 2-point bar, report both facts plainly â€” the rule still decides.

---

### Task 9: Confirm the reload, then score the test set once (Robbi runs this)

**Concept:** check the tool before the one-shot measurement. Re-scoring validation must reproduce the `metrics.json` accuracy exactly; if it doesn't, the saved model isn't loading correctly and the test score would be wasted.

In the commands below, `<final>` is the folder name agreed in Task 8 (`03-resnet18-finetuned` or `04-resnet18-augment`).

- [ ] **Step 1: Re-score validation**

```powershell
.\.venv\Scripts\python.exe -m src.score --name <final> --split validation
```
Expected: the accuracy printed equals that experiment's `metrics.json` accuracy (93.0% for experiment 3). If it differs, stop and investigate with superpowers:systematic-debugging before going further.

- [ ] **Step 2: Score the test set â€” once**

```powershell
.\.venv\Scripts\python.exe -m src.score --name <final> --split test
```
Expected: accuracy and per-class scores over roughly 450 test scans; `test_metrics.json` and `test_confusion_matrix.png` written into `experiments/<final>/`.

- [ ] **Step 3: Commit the test result immediately**

```powershell
git add experiments/<final>/test_metrics.json experiments/<final>/test_confusion_matrix.png
git commit -m "exp: final model scored once on the test set" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: â¸ PAUSE 2 â€” review the test result with Robbi

- [ ] **Step 1: Compare test with validation**

Put the final model's validation and test scores side by side (accuracy, per-class precision and recall) and show `test_confusion_matrix.png`.

- [ ] **Step 2: Discuss**

- How far is the test score from validation, and why a drop is expected (validation was used to choose; test was not).
- Which tumour type is weakest on test, and does it match validation?
- The number that goes in the README headline.

If test is well below validation, report it plainly â€” that gap is the finding. Do not re-score, retrain or adjust anything; the guard would refuse anyway.

---

### Task 11: Report, notes and publish

**Concept:** state the rule, then the result. Showing that the selection rule was written before the result existed is what makes the headline number believable.

**Files:**
- Modify: `README.md`, `docs/learning-notes.md`

- [ ] **Step 1: Update `README.md`**

Fill every angle-bracket value from the experiment files and the Task 8 and 10 discussions. Do not round differently or invent figures.

1. Status line:

```markdown
> **Status:** in progress. Experiments 1â€“4 complete and the final model scored once on the test set; heatmaps and a live demo are next.
```

2. Add a row to the experiments table:

```markdown
| 4 | ResNet-18 fine-tuned + augmentation | 1e-4 | <accuracy> | <meningioma recall> | <spread> points |
```

and change the table's intro sentence "Everything else (data, split, 15 epochs, batch size 32, Adam, seed, class weights, no augmentation) is identical across all three runs." to "Everything else (data, split, 15 epochs, batch size 32, Adam, seed, class weights) is identical across all four runs; only experiment 4 uses augmentation." Replace the sentence "Experiment 3 is the model carried into the next stage." with "<one sentence naming the final model and why, per the rule>".

3. After the Experiment 3 subsection, add:

```markdown
### Experiment 4: adding augmentation

Identical to experiment 3, except each training scan is randomly changed every time it is used: mirrored left-right half the time, rotated up to 10Â°, and brightness and contrast shifted up to 10%. Validation scans are never changed. No vertical flips, because an upside-down brain never comes out of a scanner.

![Augmentation examples](experiments/04-resnet18-augment/augmentation.png)

![Training curves](experiments/04-resnet18-augment/curves.png)

**Validation results** (best epoch <n>, 457 scans):

| Tumour type | Precision | Recall |
|---|---|---|
| Glioma | <p> | <r> |
| Meningioma | <p> | <r> |
| Pituitary | <p> | <r> |

Overall validation accuracy: <accuracy>.

![Confusion matrix](experiments/04-resnet18-augment/confusion_matrix.png)

**What this shows.** <2â€“4 sentences agreed in Task 8: effect on overfitting, on meningioma precision, and anything surprising.>
```

4. After the experiments sections and before `## Limitations`, add:

```markdown
## Final test results

**The rule, written before experiment 4 ran:** experiment 4 would become the final model only if it beat experiment 3's 93.0% validation accuracy by at least 2 points (â‰¥ 95.0%); otherwise experiment 3 would. A smaller margin is within epoch-to-epoch noise, and a tie goes to the simpler model. Experiment 4 scored <accuracy>, so the final model is **<final model>**.

That model was then scored **once** on the 34 test patients (<n> scans) that had played no part in any decision. The code refuses a second test score.

| Tumour type | Precision | Recall |
|---|---|---|
| Glioma | <p> | <r> |
| Meningioma | <p> | <r> |
| Pituitary | <p> | <r> |

**Test accuracy: <accuracy>** (validation: <validation accuracy>).

![Test confusion matrix](experiments/<final>/test_confusion_matrix.png)

<2â€“4 sentences agreed in Task 10: the gap from validation and why, the weakest type.>
```

5. In `## How to run`, add after the experiment 3 line:

```
    .\.venv\Scripts\python.exe -m src.train --name 04-resnet18-augment --model resnet18 --epochs 15 --lr 1e-4 --augment
    .\.venv\Scripts\python.exe -m src.score --name <final> --split test
```

- [ ] **Step 2: Append three sections to `docs/learning-notes.md`**

Fill the angle-bracket figures from the results; keep the rest as written.

```markdown
## 14. Data augmentation

A model that has seen the same 2,100 scans fifteen times can simply memorise them â€” experiment 3 reached 100% training accuracy by epoch 5. Augmentation changes each scan slightly every time it is served: mirrored, tilted a few degrees, a little brighter or darker. The model never sees exactly the same image twice, so memorising pixels stops paying off and it has to learn what a tumour looks like.

The changes must be ones a real scanner could produce. A mirrored brain is still a realistic brain, a slight tilt is a patient's head position, and brightness varies between scanners. An upside-down brain never comes out of an MRI machine, so there are no vertical flips â€” teaching the model to handle impossible images wastes its capacity.

Augmentation is applied to training only. Validation and test scans stay untouched, otherwise their scores would change from run to run for reasons unrelated to the model.

<one or two sentences on what experiment 4 actually showed>

> **Say:** "Augmentation shows the model a slightly different version of each scan every epoch â€” mirrored, tilted, brightness shifted â€” only changes a real scanner could produce. It fights memorisation, and it's only ever applied to training data."

## 15. Deciding the rule before seeing the result

Experiment 3 scored 93.0% on validation. If experiment 4 came in at 93.5%, is it better? Validation accuracy moved by up to 9 points from one epoch to the next in experiment 3, so half a point is noise.

The danger is choosing the rule after seeing the numbers â€” "higher accuracy wins" when that suits, "lower loss wins" when that suits. Each choice feels reasonable, and together they quietly pick whatever looks best. So the rule was written into the spec before experiment 4 ran: augmentation had to win by at least 2 points (95.0%), otherwise the simpler model stayed. In science this is called pre-registration.

> **Say:** "I wrote the model-selection rule down before running the final experiment â€” it had to win by two points, or the simpler model stayed. Deciding the rule after seeing results lets you pick whatever looks best without meaning to."

## 16. Why the test set is used once

Every decision in this project â€” which epoch to keep, which learning rate, which model â€” was made by looking at validation scores. That makes validation a little flattering: the choices were tuned to it. The 34 test patients influenced nothing, which is the only reason their score is an honest estimate of performance on new patients.

That honesty survives exactly one look. Score the test set, adjust something, score again, and the test set has become a second validation set. So `src/score.py` refuses to score the test set if any experiment already has test results.

Final model on test: <test accuracy>, against <validation accuracy> on validation. <one sentence on the gap>

> **Say:** "The test set was touched once, at the very end, and the code refuses a second look. Validation was used for every decision, so it's slightly optimistic; the test score is the honest number â€” <test accuracy>."
```

- [ ] **Step 3: Check nothing was left unfilled**

```powershell
Select-String -Path README.md, docs\learning-notes.md -Pattern '<[a-z]'
```
Expected: no matches.

- [ ] **Step 4: Run the full suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: 50 passed (51 after the final-review fixes).

- [ ] **Step 5: Commit**

```powershell
git add README.md docs/learning-notes.md
git commit -m "docs: stage 3 results, final test score and notes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Robbi reviews the wording, then merge and push**

Ask Robbi to read the new README sections and the **Say:** lines. Only after they approve:

```powershell
git checkout main
git merge feat/stage-3-augmentation --ff-only
git push
git branch -d feat/stage-3-augmentation
```
Expected: the push reports `main -> main`.

---

## Done when

- `.\.venv\Scripts\python.exe -m pytest` passes in full (51 tests).
- `experiments/04-resnet18-augment/` holds config, history, metrics and three charts, committed without `model.pt`.
- Exactly one `test_metrics.json` exists in the project, committed, for the final model chosen by the pre-set rule.
- Robbi has reviewed both pauses and the wording; README and notes are merged to `main` and pushed.
- **Next:** Stage 4 plan â€” Grad-CAM, heatmap-in-mask score and mistake gallery on the final model.
