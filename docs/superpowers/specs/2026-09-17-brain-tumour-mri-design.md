# Brain Tumour MRI Classifier — Design

**Date:** 2026-09-17
**Status:** Approved in brainstorming, awaiting written-spec review

## 1. Purpose

A portfolio project for internship applications that shows hands-on AI experience in a medical setting. It links to Robbi's Siemens MRI internship, where AI was being used to support cancer detection.

The project trains image classifiers to identify brain tumour type from MRI slices. It is built so that Robbi understands every step: how a model learns, how to tell whether it has really learned, and how results can mislead.

**Success means:**
- A public GitHub repo whose README reads as a clear, honest report.
- A live demo that someone can use in an interview.
- Robbi can explain the training loop, patient-level splitting, overfitting, Grad-CAM and the "99% trap" in their own words.

**Non-goals:** clinical accuracy, beating published results, 3D volumes, segmentation, accelerated MRI reconstruction, cloud/GPU training, slides (made later by Robbi).

## 2. Constraints

- **Hardware:** Windows 11 laptop, i5-1340P, 8 GB RAM, integrated graphics only, so training is CPU-only.
- **Disk:** about 10.8 GB free on C:. The whole project must stay within roughly 3–4 GB.
  - Install CPU-only PyTorch.
  - Delete downloaded archives after conversion.
  - Save only the best checkpoint per experiment.
  - The download script checks free space first and stops with a clear message if there is less than 3 GB.
- **Location:** `C:\Users\Robbi\brain-tumour-mri` (not OneDrive). GitHub is the backup.
- **Python:** 3.13 via `py` (bare `python` is shadowed by the Windows Store alias), with a project `.venv`.
- **Robbi is a beginner.** Code must be readable, and concepts are explained before they are coded.

## 3. Data

### 3.1 Main dataset: Cheng et al. (Figshare)

- **Contents:** about 3,064 T1-weighted contrast-enhanced brain MRI slices from 233 patients.
- **Classes:** glioma (~1,426), meningioma (~708), pituitary (~930).
- **Per-slice fields:** image (512×512), label, patient ID, tumour mask drawn by clinicians.
- **Format:** MATLAB v7.3 `.mat` files, read with `h5py`.
- **Licence:** CC BY 4.0. Cite the dataset in the README.
- **No account needed.** Counts and licence are confirmed against the actual download in Stage 0.

### 3.2 Patient-level split

- **Proportions:** 70% train / 15% validation / 15% test, **by patient**, stratified by tumour type (each patient has one tumour type), with a fixed seed.
- **Rule:** no patient appears in more than one split. A test enforces this.
- **Validation** is used for every decision during development.
- **Test** is used only once, on the final chosen model at the end of Stage 3. Before that, reported numbers are validation numbers and are labelled as such.
- **The split is saved** to `data/splits.csv` (patient ID → split) so every experiment uses the identical split.

### 3.3 Preprocessing

- **Resize:** each image and mask are resized to 224×224.
- **Normalise:** each image is min-max scaled to [0, 1], because MRI intensities have no fixed scale.
- **Channels:** the from-scratch model uses 1 channel. For the pretrained model, the single channel is repeated to 3.
- **Storage:** after conversion, everything is stored in one compressed file, `data/cheng_224.npz` (images, masks, labels, patient IDs). The raw download is deleted.

### 3.4 Secondary dataset: Kaggle "Brain Tumor MRI Dataset" (Stage 6 only)

- **Contents:** about 7,000 2D images, 4 classes (glioma, meningioma, pituitary, no tumour). Combined from several sources, with known near-duplicates and no patient IDs.
- **Download:** Robbi downloads it manually through their own Kaggle account and places the zip in `data/kaggle/`.
- **Licence:** confirmed at download time.
- **Purpose:** only to demonstrate how source bias and duplicates inflate scores. It is never mixed with the main results.

## 4. Project structure

```
brain-tumour-mri/
  data/               gitignored; recreated by scripts
  src/
    download.py       disk-space check, download, convert to .npz, delete raw
    data.py           load .npz, patient split, PyTorch Dataset, augmentation
    models.py         SmallCNN (from scratch) and pretrained ResNet-18
    train.py          training loop, best-checkpoint saving, experiment record
    metrics.py        accuracy, per-class precision/recall, confusion matrix
    evaluate.py       score a checkpoint on validation or test
    explain.py        Grad-CAM, heatmap-in-mask score, mistake gallery
  experiments/
    <experiment-name>/
      config.json     settings used
      history.csv     per-epoch train/val loss and accuracy
      metrics.json    final scores
      *.png           curves, confusion matrix, heatmaps
      model.pt        gitignored (regenerable)
  app/
    app.py            Gradio demo
    requirements.txt  demo-only dependencies
  tests/
  docs/
    learning-notes.md plain-English concept notes (interview prep)
  README.md           the report
  requirements.txt
```

**Boundaries:**
- `metrics.py` is pure functions (no PyTorch) and is testable by hand.
- `models.py` knows nothing about data files.
- `train.py` wires the pieces together.
- `explain.py` depends on a trained model and data, never on training code.

## 5. Models and training

### 5.1 Models

- **SmallCNN:** about 4 convolution blocks (conv → batch norm → ReLU → max pool), then global average pooling and a linear layer to 3 outputs. Small enough to read in one screen and train on CPU.
- **ResNet-18:** torchvision, ImageNet-pretrained weights (~45 MB download), with the final layer replaced for 3 classes. Fine-tuned end to end.

### 5.2 Training loop

- **Per batch:** forward pass → cross-entropy loss → backward pass → optimiser step.
- **Class imbalance:** handled with loss weights inversely proportional to class frequency in the training split.
- **Optimiser:** Adam.
- **After each epoch:** evaluate on validation, record train/val loss and accuracy to `history.csv`, and keep the checkpoint with the best validation accuracy.
- **Epochs:** 10–30, set per experiment.
- **Reproducibility:** fixed seeds for Python, NumPy and PyTorch, plus deterministic data order.

### 5.3 Experiments

| # | Name | Change from previous | Question it answers |
|---|---|---|---|
| 1 | `01-small-cnn` | Baseline from scratch | How far does a simple model get on its own? |
| 2 | `02-resnet18-lr1e-3` | Swap to pretrained ResNet-18, learning rate unchanged | Does prior visual knowledge help with ~2,000 training images? |
| 3 | `03-resnet18-finetuned` | Drop the learning rate to 1e-4 | Does the correct fine-tuning rate settle the validation curve? |
| 4 | `04-resnet18-augment` | Add augmentation (small rotations, flips, slight brightness/contrast changes) | Does extra variety reduce overfitting? |
| 5 | `05-kaggle-trap` | Same pipeline on the Kaggle data, random image-level split | Why can a 99% score be untrustworthy? |

- **Comparisons:** experiments 1–4 are compared on validation, one change at a time.
- **Stage 2 detail:** see `2026-09-21-stage-2-resnet18-design.md`.
- **Final model:** the best of 1–4 becomes the final model, scored once on the test set.
- **Experiment 5:** reported separately.

### 5.4 Metrics

- **Scores:** overall accuracy, per-class precision and recall, and a confusion matrix.
- **Emphasis:** the README highlights recall (missed tumours) and explains why accuracy alone is not enough for medical use.
- **Confidence:** each prediction records the model's softmax confidence, used to flag confident mistakes.

## 6. Explainability and error analysis

- **Grad-CAM:** implemented by hand in `explain.py` using forward/backward hooks on the last convolutional layer, so the mechanism is understandable rather than hidden in a library. Heatmaps are overlaid on scans.
- **Heatmap-in-mask score:** for each scan, the fraction of total heatmap intensity that falls inside the clinician tumour mask.
  - **Reported:** the distribution of this score, plus "correct **and** focused on the tumour" as a percentage.
  - **Focused on the tumour** means a score ≥ 0.5 (at least half the heatmap mass inside the mask). This threshold is stated in the README.
- **Mistake gallery:** every misclassified test scan, showing scan, prediction, true label, confidence and heatmap. Confident mistakes (confidence ≥ 0.9) are flagged. Short written notes describe any patterns.
- **Kaggle trap:**
  - Grad-CAM runs on experiment 4's model, and any focus outside the brain is shown directly.
  - A simple near-duplicate check (downscaled-image hash comparison across train/test) quantifies leakage.
  - Findings are reported as observed, with no outcome assumed in advance.

## 7. Live demo

- **Tool:** Gradio app in `app/`, hosted on Hugging Face Spaces (free CPU tier). Robbi creates the Hugging Face account themselves.
- **Input:** example test scans (with CC BY 4.0 attribution) or an uploaded image.
- **Output:** predicted type, confidence bars for all 3 classes, and a Grad-CAM overlay.
- **Always visible:**
  - "Educational project — not a medical device, not for diagnosis."
  - A note that the model only knows these three tumour types and will still pick one for a healthy scan or a non-MRI image.
- **Privacy:** uploaded images are not stored or logged by the app.
- **Model:** uses the final model checkpoint (uploaded to the Space, not to GitHub).

## 8. README report

Sections, written incrementally as stages finish:
1. Headline: summary, demo link, screenshot
2. Motivation, with the link to the Siemens MRI internship
3. Data: source, citation, patient-level split, class imbalance
4. Experiments table and training curves
5. Final test results: per-class scores and confusion matrix
6. Where the model looks: Grad-CAM examples and heatmap-in-mask results
7. Mistakes: gallery and patterns
8. The 99% trap
9. Limitations: 2D slices, single dataset source, class set of three, not clinically validated
10. How to run

## 9. Testing

`pytest`, using small synthetic data, so no dataset download is needed:
- **Split:** no patient appears in two splits; every patient is assigned; the split is stratified.
- **Preprocessing:** output shape is 224×224; values are within [0, 1]; constant images don't divide by zero.
- **Models:** a batch of the right shape gives 3 outputs per image for both models (the ResNet test uses no pretrained weights, so it runs offline).
- **Learning sanity check:** SmallCNN's loss drops substantially when overfitting 8 synthetic images for a few dozen steps.
- **Metrics:** match hand-worked examples, including a class with zero predictions (no divide-by-zero).
- **Heatmap-in-mask:** matches hand-computed values on constructed heatmaps and masks.
- **Download:** the disk-space check refuses when free space is below the threshold (free space is mocked).

## 10. Build stages

Each stage ends with passing tests, updated README and learning notes, and a push to GitHub.

| Stage | Delivers |
|---|---|
| 0 | Project setup, download + conversion, patient split, data tests |
| 1 | Metrics, SmallCNN, training loop, experiment 1, README v1 (validation results) |
| 2 | ResNet-18, experiments 2 and 3, comparison |
| 3 | Augmentation, experiment 4, final model chosen and scored once on test |
| 4 | Grad-CAM, heatmap-in-mask score, mistake gallery |
| 5 | Gradio demo deployed to Hugging Face Spaces |
| 6 | Kaggle experiment 5 and "99% trap" README section |

## 11. Working method

- **Per stage:** explain the concept → build in small tested pieces (no check-ins between pieces) → Robbi runs the commands → **pause to review results together** and decide the next step → learning notes → commit and push.
- **Outward-facing actions:** creating the GitHub repo, the Hugging Face Space and the Kaggle download each need Robbi's go-ahead or are done by Robbi.
