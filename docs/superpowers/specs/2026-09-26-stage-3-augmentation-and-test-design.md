# Stage 3: Augmentation and the Test-Set Score — Design

**Date:** 2026-09-26
**Status:** Approved in brainstorming, awaiting written-spec review
**Overall design:** `2026-09-17-brain-tumour-mri-design.md` (sections 5.3 and 10)

## 1. Purpose

Experiment 3 (pretrained ResNet-18, learning rate 1e-4) reached 93.0% validation accuracy but memorised its training set: training accuracy hit 100% by epoch 5 and validation loss rose after epoch 3. Stage 3 answers two questions:

1. **Does augmentation reduce that overfitting?** (experiment 4)
2. **How good is the final model on patients it has never been judged against?** (one score on the test set)

## 2. Final-model rule (fixed before experiment 4 runs)

> Experiment 4 becomes the final model only if its validation accuracy is **95.0% or higher** (experiment 3's 93.0% plus 2 points). Otherwise experiment 3 is the final model.

- "Validation accuracy" means the `accuracy` field of `metrics.json`, the same number shown in the README's experiments table.
- A 2-point margin is about 9 of 457 scans. Experiment 3's validation accuracy moved between 84% and 93% from epoch to epoch, so a smaller win is treated as noise, and a tie goes to the simpler model.
- Experiments 1 and 2 (73.7%, 88.0%) are already beaten and are not candidates.
- The rule is written down now so it cannot be chosen after seeing the result.

## 3. Augmentation

**Approach:** torchvision's `transforms.v2`, applied inside `MRIDataset`. This was chosen over hand-written NumPy transforms (more error-prone) and batch-level augmentation in the training loop (GPU-oriented, mixes data concerns into training). Verified: torchvision 0.29.0 applies these transforms to 1-channel float tensors of shape (1, 224, 224).

`train_transform()` in `src/data.py` returns, applied in this order:

| Transform | Setting | Why |
|---|---|---|
| Horizontal flip | probability 0.5 | A mirrored brain is still anatomically realistic |
| Rotation | uniformly within ±10° | Small head tilts in the scanner; empty corners are filled with 0 (black), matching the scan background |
| Brightness and contrast | each within ±10% | MRI intensity varies between scanners and settings |
| Clip | to [0, 1] | Keeps the input range every model expects |

**No vertical flips.** An upside-down brain never comes out of a real scanner.

`MRIDataset(images, labels, indices, transform=None)`: with `transform=None` it returns exactly what it returns today, so experiments 1–3 remain reproducible. The transform is applied to the tensor after scaling to [0, 1].

**Reproducibility:** transforms draw from PyTorch's global random generator, which `set_seed` fixes. Data loading stays single-process, so a run repeats exactly.

## 4. Training changes (`src/train.py`)

- New flag `--augment` (default off), passed to `run_experiment` as `augment=False` and recorded in `config.json`.
- Only the training `MRIDataset` receives `train_transform()`. Validation always sees unaltered scans.
- When `augment` is on, the run saves `augmentation.png` in the experiment folder: one training scan and 7 augmented versions of it, in a 2×4 grid. The drawing function lives in `src/plots.py` alongside the existing plots. The preview is drawn after training finishes, so the random draws it uses cannot change the training run.

**Experiment 4:**

```
.\.venv\Scripts\python.exe -m src.train --name 04-resnet18-augment --model resnet18 --epochs 15 --lr 1e-4 --augment
```

Everything except `--augment` matches experiment 3. The expected runtime is about an hour; experiment 3 took 51 minutes.

## 5. Test-set scoring (`src/score.py`)

A new module rather than an addition to `src/evaluate.py`: `train.py` already imports `evaluate.py`, and scoring needs `class_weights` from `train.py`, so putting it in `evaluate.py` would create a circular import.

New command-line entry point:

```
.\.venv\Scripts\python.exe -m src.score --name <experiment> --split test
```

- It reads the experiment's `config.json` for the model name, builds that model with `pretrained=False` (no download, since `model.pt` overwrites every weight) and loads `model.pt`.
- It scores every scan from the test patients, using the same class-weighted loss as training (weights from the training split) so the loss figure is comparable.
- It writes `test_metrics.json` (the same fields as `metrics.json`, with `"split": "test"`) and `test_confusion_matrix.png` into that experiment's folder, and prints accuracy and per-class precision and recall.
- **One-shot guard:** before loading anything, it refuses with a clear message if any folder under `experiments/` already contains `test_metrics.json`. The "use the test set once" rule is enforced by the code, not just by intention. There is no override flag.
- `--split validation` is also accepted and is not guarded. It prints the validation scores and writes no files. It is run on the final model just before the test score, to confirm the saved model reloads and reproduces its `metrics.json` accuracy, so the one-shot test score cannot be wasted on a loading bug.

## 6. Tests (pytest, synthetic data only)

- `train_transform()` keeps shape (1, 224, 224) and every value within [0, 1].
- `MRIDataset` without a transform returns the same tensor as before; with a transform, the transform is applied.
- The guard refuses when any experiment folder holds `test_metrics.json`, and allows it when none does.
- End to end: on a tiny synthetic dataset, splits file and saved SmallCNN, the test-scoring function writes `test_metrics.json` with `"split": "test"` and the right scan count.
- `run_experiment` with `augment=True` records it in `config.json` and writes `augmentation.png` (a tiny subset run).

## 7. Working method and pauses

- Code tasks run without check-ins. Robbi runs experiment 4 and the test score.
- **Pause 1, after experiment 4:** review its results against experiment 3 and apply the rule in section 2.
- **Pause 2, after the test score:** review the test results before writing them up. If the test score is well below validation, report it plainly; that gap is a finding, not something to tune away.

## 8. Write-up

- **README:** experiment 4 row in the experiments table, an experiment 4 section with curves, confusion matrix and `augmentation.png`, and a new **Final test results** section stating the pre-set rule, the chosen model, the test scores and the confusion matrix. The status line is updated.
- **Learning notes:** 14 Data augmentation; 15 Deciding the rule before seeing the result; 16 Why the test set is used once. Each ends with a **Say:** line.
- Merge to `main` and push once Robbi has reviewed the wording.

## 9. Out of scope

- Multiple random seeds per configuration (considered and declined: about 4 extra hours).
- The ResNet-18-from-scratch control and the BatchNorm diagnostic (both declined in Stage 2).
- Retraining the final model on training plus validation data.
- Changing the number of epochs, the learning rate or checkpoint selection for experiment 4.
