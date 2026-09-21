# Brain Tumour MRI Classifier

Classifying brain tumour type (glioma, meningioma, pituitary) from MRI slices with deep learning, built step by step to understand how the model learns, and where its results can mislead.

> **Status:** in progress. Experiment 1 (baseline) complete; pretrained models, heatmaps and a live demo are next.
> **Educational project, not a medical device, not for diagnosis.**

## Why this project

During my internship at Siemens working on MRI, I learned that AI is being used both to shorten scan times and to help clinicians detect cancer. This project explores that from the ground up: training a model on real MRI data, and asking not just "how accurate is it?" but "can that number be trusted?"

## Data

- **Source:** 3,064 contrast-enhanced T1-weighted MRI slices from 233 patients (Cheng et al.). Glioma: 1,426 · Meningioma: 708 · Pituitary: 930.
- **Split by patient, not by slice:** 165 / 34 / 34 patients into training / validation / test (70% / 15% / 15%, stratified by tumour type). No patient appears in more than one set; an automated test enforces this. Splitting by slice would let the model see near-identical neighbouring slices of a test patient during training, which inflates the score.
- **Class imbalance:** there are twice as many gliomas as meningiomas. The loss is weighted so rarer types aren't ignored, and results are reported per type.

## Experiment 1: small CNN from scratch

A 4-block convolutional network (98,019 parameters) trained only on these scans for 15 epochs on a laptop CPU, roughly 1–2 minutes per epoch.

![Training curves](experiments/01-small-cnn/curves.png)

**Validation results** (best epoch 9, 457 scans; the test set is untouched until the final model is chosen):

| Tumour type | Precision | Recall |
|---|---|---|
| Glioma | 0.76 | 0.91 |
| Meningioma | 0.73 | 0.18 |
| Pituitary | 0.69 | 0.88 |

Overall validation accuracy: 73.7%.

![Confusion matrix](experiments/01-small-cnn/confusion_matrix.png)

**What this shows.** The model learned something real, but not evenly. Training accuracy climbed steadily to 92% while validation accuracy bounced between 23% and 74% from epoch to epoch — a network with 98k parameters memorising ~2,100 training slices rather than learning tumour appearance in general. Almost all of the error is one class: meningioma recall is 0.18, meaning it catches 19 of 104 meningiomas and misfiles 49 as glioma and 36 as pituitary, while glioma (0.91) and pituitary (0.88) are largely fine. The headline 73.7% also deserves an asterisk: the checkpoint is chosen as the best of 15 noisy validation readings, so some of that peak is luck rather than skill — epoch 11 had a clearly better loss (0.64 vs 0.85) but a lower accuracy. The next experiment tests whether a pretrained ResNet-18, which starts with edge and texture detectors learned from millions of images, steadies that curve and rescues meningioma.

## Limitations

- Single 2D slices, not full 3D scans.
- One dataset, collected at two hospitals in China between 2005 and 2010; performance elsewhere is unknown.
- Only three tumour types; the model cannot say "no tumour".
- Not clinically validated.

## How to run

Requires Windows with Python 3.13.

    py -3.13 -m venv .venv
    .\.venv\Scripts\python.exe -m pip install -r requirements.txt
    .\.venv\Scripts\python.exe -m pytest
    .\.venv\Scripts\python.exe -m src.download
    .\.venv\Scripts\python.exe -m src.train --name 01-small-cnn --model small_cnn --epochs 15

## Data citation

Cheng, Jun (2017). *brain tumor dataset*. figshare. https://doi.org/10.6084/m9.figshare.1512427 (CC BY 4.0).
Cheng, J. et al. (2015). Enhanced Performance of Brain Tumor Classification via Tumor Region Augmentation and Partition. *PLoS ONE* 10(10).
