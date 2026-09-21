# Stage 2: Pretrained ResNet-18 — Design

Supplements `2026-09-17-brain-tumour-mri-design.md`. That document remains the overall design; this one covers Stage 2 only and renumbers the later experiments.

## 1. Purpose

Experiment 1 (`01-small-cnn`) reached 73.7% validation accuracy with a meningioma recall of 0.18 and a validation curve that swung between 23% and 74% from epoch to epoch. Stage 2 asks two questions:

1. Does prior visual knowledge from ImageNet beat learning from scratch on ~2,100 training scans?
2. Is the unstable validation curve caused by the learning rate?

## 2. Why two runs, not one

The overall spec requires one change per experiment. A strict swap to ResNet-18 keeps `lr=1e-3`, which is roughly 10× too high for fine-tuning: pretrained weights are an already-good solution, and updates sized for randomly-initialised weights destroy the features being borrowed. Fine-tuning convention is 1e-4 or lower.

Rather than silently changing two things, Stage 2 runs both configurations. The badly-configured run is deliberate: it isolates the model swap, demonstrates the instability instead of asserting it, and gives the README direct evidence that learning rate depends on where training starts.

## 3. Experiments

| # | Name | Model | LR | Epochs | Question |
|---|---|---|---|---|---|
| 2 | `02-resnet18-lr1e-3` | ResNet-18 | 1e-3 | 15 | Pure one-variable swap: what does pretraining alone do? |
| 3 | `03-resnet18-finetuned` | ResNet-18 | 1e-4 | 15 | Does the correct fine-tuning rate settle the curve and recover meningioma? |

Batch size 32 and seed 42 are unchanged from experiment 1, so the comparison against `01-small-cnn` is fair.

**Renumbering.** Later experiments shift by one: augmentation becomes `04-resnet18-augment`, the Kaggle trap `05-kaggle-trap`. Section 5.3 of the overall spec is updated to match.

## 4. Model

`ResNet18Grey` in `src/models.py` wraps `torchvision.models.resnet18`:

- **Weights:** `IMAGENET1K_V1`, about 45 MB, downloaded once on the first real run (needs internet that one time). Constructible with `weights=None` so tests run offline.
- **Head:** the final fully-connected layer is replaced with a fresh `Linear(512, 3)`.
- **Fine-tuned end to end:** no layers frozen, matching §5.1 of the overall spec.

### 4.1 Input adaptation

Two mismatches between greyscale MRI and a model trained on photographs are handled **inside the wrapper's forward pass**, not in `MRIDataset`:

- **Channels:** the single greyscale channel is repeated three times. This preserves ResNet's pretrained first convolution exactly as ImageNet trained it.
- **Value range:** ImageNet normalisation is applied (mean 0.485/0.456/0.406, std 0.229/0.224/0.225), because pretrained weights expect inputs distributed the way they were trained.

Keeping this inside the model means `MRIDataset` continues to return plain 1-channel tensors in [0, 1], `SmallCNN` is unaffected, and each model owns the input format it requires.

## 5. Training pipeline

No changes. `src/train.py` already accepts `--model` and `--lr`, so both runs are command-line invocations. `build_model` gains one branch for the name `resnet18`.

## 6. Metrics addition

`metrics.json` gains **`best_loss_epoch`**: the epoch with the lowest validation loss, recorded alongside the existing best-accuracy epoch. Training behaviour and checkpoint selection are unchanged — the checkpoint is still the best validation *accuracy*, as in experiment 1, so comparisons hold.

The purpose is honesty about selection. On a noisy curve the two epochs disagree (in experiment 1, accuracy peaked at epoch 9 while loss was lowest at epoch 11), and that disagreement is evidence that the headline figure is partly a lucky pick rather than a stable estimate.

## 7. Testing

Added to `tests/test_models.py`, following the existing pattern and using `weights=None` so `pytest` stays offline and fast:

- `ResNet18Grey` maps a batch of 1-channel 224×224 images to 3 scores per image.
- The channel repeat produces 3 identical channels of the expected shape.
- ImageNet normalisation shifts and scales values as expected.

Added to `tests/test_train.py`:

- `best_loss_epoch` is the epoch with the lowest validation loss in a hand-built history.

## 8. Documentation

- **README:** an experiments comparison table (1 vs 2 vs 3), the curve-stability finding, and updated per-class results.
- **`docs/learning-notes.md`:** three sections — transfer learning and what ImageNet features are worth; freezing versus fine-tuning; why the correct learning rate depends on whether weights start random or already-trained.

## 9. Constraints and risks

- **Runtime:** ResNet-18 is roughly 5× the compute of SmallCNN (about 1.8 GFLOPs per image against 0.36), so expect 8–15 minutes per epoch and 2–4 hours per run on this CPU-only laptop. Both runs together: 4–8 hours. Robbi has accepted this.
- **Disk:** about 11 GB free. Adds torchvision plus 45 MB of weights; well within budget.
- **Memory:** 8 GB RAM, batch size 32 at 224×224 with an 11M-parameter model. If a run exhausts memory, the fallback is batch size 16, recorded in the experiment config.
- **Expected outcome is not guaranteed.** If the fine-tuned run does not improve meningioma recall, that is a real result and gets reported as one, not tuned away.

## 10. Done when

- `pytest` passes in full.
- `experiments/02-resnet18-lr1e-3/` and `experiments/03-resnet18-finetuned/` each hold config, history, metrics and both charts.
- Robbi has reviewed the results against experiment 1.
- README and learning notes updated, committed and pushed.
- **Next:** Stage 3 plan — augmentation (`04-resnet18-augment`).
