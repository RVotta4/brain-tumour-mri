# Stage 4: Where the Model Looks (Grad-CAM) — Design

**Date:** 2026-09-27
**Status:** Approved in brainstorming, awaiting written-spec review
**Overall design:** `2026-09-17-brain-tumour-mri-design.md` (section 6). This spec replaces that section's heatmap-in-mask threshold, for the reason given in section 3.

## 1. Purpose

The final model (experiment 3: pretrained ResNet-18, fine-tuned at 1e-4) scored 90.9% on the test set, with meningioma the weakest type (61 of 80 caught). Stage 4 asks:

1. **Is the model looking at the tumour, or at something else?** Measured against the clinicians' tumour outlines, not eyeballed.
2. **What do its mistakes look like?** Every wrong test answer, with its heatmap and confidence.
3. **Are the meningioma misses spread out, or concentrated in a few patients?**

## 2. Ground rules

- **Test scans, model frozen.** The analysis runs on the 430 test scans behind the headline number. Looking at predictions does not change the one-shot score, but acting on what they show would let the test set steer the model. So from this stage on the model is frozen: whatever Stage 4 finds is reported, not fixed. Nothing in Stage 4 trains or saves model weights.
- **Only after the test score exists.** The analysis refuses to run unless the experiment folder already contains `test_metrics.json`. Before analysing, it re-checks that its own predictions reproduce the saved test accuracy and prints MATCH or MISMATCH.
- **No cherry-picking.** Example scans are chosen at random with a fixed seed, and the mistake gallery shows every mistake.

## 3. Why the original in-mask threshold was dropped

The original design called a scan "focused on the tumour" when at least half the heatmap's mass fell inside the tumour outline. The data rules this out:

| Test set | Slices | Patients | Tumour area, median share of image |
|---|---|---|---|
| Meningioma | 80 | 12 | 1.0% |
| Glioma | 222 | 13 | 1.6% |
| Pituitary | 128 | 9 | 0.7% |

Grad-CAM on ResNet-18's last convolutional stage produces a 7×7 map. Each cell covers about 2% of the 224×224 image, which is larger than the typical tumour. Even perfectly placed attention spreads well beyond the outline once the map is enlarged, so a 50% threshold would be failed by almost every scan and would wrongly suggest the model ignores tumours. Two measures that fit the data replace it (section 5).

## 4. Grad-CAM (`src/explain.py`)

`grad_cam(model, image)` takes a `ResNet18Grey` in evaluation mode and one image tensor of shape (1, 224, 224) with values in [0, 1]. It returns `(heatmap, predicted_class, confidence)`, where the heatmap is a 224×224 NumPy array in [0, 1].

The steps:

1. Run the model in two halves, written out explicitly rather than with hooks:
   - **Feature half:** `model.prepare`, then the backbone's `conv1`, `bn1`, `relu`, `maxpool`, and `layer1` to `layer4`. This gives 512 maps of 7×7.
   - **Decision half:** `avgpool`, flatten, then `fc`, giving three class scores.
2. The target is the predicted class, meaning the highest score. The explanation is of the decision the model actually made.
3. `torch.autograd.grad` gives the gradient of the target score with respect to the 512 maps. Each map's weight is its gradient averaged over the 7×7 positions.
4. The heatmap is the ReLU of the weighted sum of the maps. It is enlarged to 224×224 with bilinear interpolation and divided by its maximum. An all-zero map stays all zeros; there is no division by zero.
5. Confidence is the softmax probability of the predicted class.

Anything other than `ResNet18Grey` raises a `TypeError` with a clear message, since Stage 4 explains only the final model. The two-half forward must give the same class scores as `model(image)`, and a test checks this.

## 5. Measurements

Both measures are plain functions of a heatmap and a binary mask, both 224×224.

**Pointing game (headline).** `pointing_hit(heatmap, mask, tolerance=8)` returns True if the heatmap's hottest pixel (the first one, if several tie) lies within 8 pixels of a tumour pixel, measured as Chebyshev distance: a square window. The 8-pixel leeway is a quarter of one heatmap cell and absorbs the blur from enlarging the map.

Its luck baseline is `pointing_chance(mask, tolerance=8)`: the fraction of image pixels that lie within 8 pixels of the tumour. A randomly placed peak would hit that often.

**In-mask share.** `mask_share(heatmap, mask)` is the sum of heatmap values inside the mask divided by the sum over the whole image. An all-zero heatmap gives 0.0.

Its luck baseline is the mask's area fraction: a heatmap spread evenly over the image would score exactly that. It is reported both as a share and as "× luck" (share ÷ area fraction).

## 6. The analysis run

```
.\.venv\Scripts\python.exe -m src.explain --name 03-resnet18-finetuned
```

It processes the 430 test scans one at a time. The expected time is a few minutes on the laptop CPU. It writes to `experiments/03-resnet18-finetuned/explain/`:

| File | Contents |
|---|---|
| `test_scans.csv` | One row per test scan: `index`, `patient_id`, `true`, `predicted`, `confidence`, `correct`, `pointing_hit`, `pointing_chance`, `mask_share`, `mask_area` |
| `summary.json` | Pointing-game hit rate, and mean in-mask share with its "× luck" figure, each overall, per true type, and for correct versus wrong answers. The mean luck baselines alongside them. The number of confident mistakes. |
| `patient_errors.csv` | One row per test patient: `patient_id`, `true`, `slices`, `wrong`, `most_common_wrong_prediction` (empty if none wrong); sorted by `wrong`, most first |
| `gradcam_examples.png` | For each type, 4 correctly classified scans chosen at random (seed 0). Each shows the scan with the tumour outline drawn, then the heatmap laid over the scan. |
| `mistakes_meningioma.png`, `mistakes_glioma.png`, `mistakes_pituitary.png` | Every wrong answer for that true type, as a heatmap over the scan with the tumour outline, titled with the predicted type and confidence. Mistakes with confidence of 0.9 or more are marked "CONFIDENT". A type with no mistakes gets no file. |

**Confident mistake** means a wrong answer with a confidence of 0.9 or higher, the threshold set in the original design.

Loading follows `src/score.py`: read `config.json`, build the model with `pretrained=False`, and load `model.pt`. The code is new, but the pattern is the same.

## 7. Tests (pytest, synthetic data, offline)

- `grad_cam` on `ResNet18Grey(weights=None)` returns a 224×224 heatmap within [0, 1], a class in {0, 1, 2} and a confidence in [0, 1]. Its two-half forward gives the same scores as `model(image)`.
- `grad_cam` raises `TypeError` for `SmallCNN`.
- `pointing_hit`: true for a peak inside the mask, true for a peak 5 pixels outside, false for a peak 20 pixels outside.
- `pointing_chance` and `mask_share` match values worked out by hand on small constructed arrays, including an all-zero heatmap.
- The per-patient summary counts slices and mistakes correctly, and sorts by mistakes, on a hand-built table.
- The gallery and example figures each write a PNG.
- The run refuses to start when `test_metrics.json` is missing.

## 8. Pause and write-up

- **Pause:** after the run, review the figures and numbers with Robbi before writing anything.
- **README:** add a "Where the model looks" section (the example figure, and pointing-game and in-mask results against their luck baselines, overall and per type) and a "Mistakes" section (the galleries, the meningioma patient breakdown, confident mistakes, and any patterns described as observed). Explain in one short paragraph why the 50% threshold was replaced. Update the status line.
- **Learning notes:** 17 Grad-CAM; 18 Measuring attention: the pointing game and luck baselines; 19 Why the heatmap is coarse. Each ends with a **Say:** line.
- Merge to `main` and push once Robbi has approved the wording.

## 9. Out of scope

- Changing, retraining or re-scoring the model because of anything found here.
- Grad-CAM for experiments 1, 2 and 4.
- Finer heatmaps from earlier layers, other explanation methods, and heatmaps for classes other than the predicted one.
- Analysing validation scans.
