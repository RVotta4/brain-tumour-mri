# Learning notes

Plain-English explanations of the ideas in this project, written for interview prep.
Each section ends with **Say:** — a one-liner to use out loud when presenting.

---

## 1. Images are numbers

A greyscale MRI slice is a grid of brightness values, nothing more. Resized to 224×224, each scan is 50,176 numbers between 0 and 1. The network never sees a picture; it sees a spreadsheet of brightnesses and finds patterns in them.

> **Say:** "Each scan is a 224×224 grid of brightness values — about 50,000 numbers. Everything the model learns, it learns from patterns in those numbers."

## 2. Why brightness is rescaled per slice

MRI intensity has no fixed unit. Unlike CT, where a value corresponds to a specific tissue density, MRI brightness depends on the scanner, its settings and the operator — the same grey matter can read 400 on one scan and 1,100 on another.

So each slice is stretched individually: its darkest pixel becomes 0, its brightest becomes 1. Every scan is then on the same footing, and the model cannot be thrown off by one hospital's scanner running bright.

The trade-off: absolute brightness is discarded. If a tumour type were genuinely brighter in raw units, that signal is now hidden. That is deliberate — robustness is worth more than a signal that cannot be trusted across scanners.

> **Say:** "MRI intensity is arbitrary — it varies by scanner and settings — so I normalise each slice to a 0–1 range individually. Otherwise the model could learn the scanner instead of the tumour."

## 3. The patient-level split

Each of the 233 patients contributed several neighbouring slices of the same tumour, and slice 7 looks almost identical to slice 8.

Shuffling all 3,064 slices and splitting those would put slice 7 in training and slice 8 in testing — the model would be tested on a tumour it had already studied, and the score would be inflated. This is **data leakage**, and it is a common reason published accuracies on this dataset look implausibly high.

Instead, whole patients are assigned to exactly one set: 165 training / 34 validation / 34 test. `tests/test_data.py` fails the build if a patient ever appears in two sets.

The three sets do different jobs:

- **Training (165 patients)** — what the model learns from.
- **Validation (34)** — the practice exam, used to make decisions such as which epoch's checkpoint to keep.
- **Test (34)** — the final exam, untouched until the final model is chosen. Because decisions are made against validation, validation stops being an honest estimate; only the test set is.

> **Say:** "I split by patient, not by slice. Neighbouring slices from one patient are nearly identical, so a slice-level split leaks the test set into training and inflates accuracy. That's why my 73.7% is lower than a lot of numbers you'll see on this dataset — and why it's trustworthy."

## 4. Convolution

A filter is a 3×3 grid of numbers that slides across the image. At each position it multiplies against the nine pixels underneath and sums them: a big result where the pattern matches, near zero where it doesn't. Sliding it over the whole slice produces a map of where that pattern occurs.

The filter's nine numbers are not chosen by hand. They start random and training tunes them — nobody told the network to look for edges; it worked out that edges were useful.

This network stacks four such blocks:

| Block | Filters | Image size |
|---|---|---|
| input | — | 224 × 224 |
| 1 | 16 | 112 × 112 |
| 2 | 32 | 56 × 56 |
| 3 | 64 | 28 × 28 |
| 4 | 128 | 14 × 14 |

Two things happen together. The **number of filters grows** (16 → 128): early blocks need only a few simple patterns such as edges, later blocks need many combinations of them — textures and shapes. The **image shrinks** (224 → 14) through max-pooling, which halves width and height by keeping the strongest response in each 2×2 patch. Shrinking means later filters cover a wider area of the original scan.

Average pooling then collapses each of the 128 maps to one number — how strongly that pattern appeared anywhere in the scan — and a single linear layer turns those 128 numbers into three scores, one per tumour type. 98,019 tuneable numbers in total.

> **Say:** "Convolution slides small learned filters over the image to build maps of where patterns occur. Early layers find edges, later layers combine them into textures and shapes. I go from 224×224 with 16 filters down to 14×14 with 128 — less spatial detail, more abstract pattern."

## 5. The training loop

For each batch of 32 scans:

1. **Guess** — the model outputs three scores per scan.
2. **Measure** — the loss condenses "how wrong was that?" into one number.
3. **Blame** — backpropagation works backwards through the network, calculating how much each of the 98,019 weights contributed to the error.
4. **Adjust** — the optimiser nudges every weight slightly in the direction that reduces the error. The learning rate (0.001 here) sets the size of the nudge: too big and it overshoots, too small and it never arrives.

One full pass over all training scans is an **epoch**. Experiment 1 ran 15 of them, at roughly 1–2 minutes each on a laptop CPU.

> **Say:** "It's a loop: predict, measure the error, work out which weights caused it, nudge them. The learning rate controls how big each nudge is."

## 6. Overfitting

The model memorises the training scans instead of learning general patterns — the difference between understanding the material and memorising the answer sheet.

It is visible in the curves from experiment 1: training accuracy climbs smoothly to 92% while validation accuracy never passes 74% and swings wildly between epochs. The training line rising while the validation line doesn't follow *is* overfitting.

Two responses in this code. The checkpoint kept is the one from the **best validation epoch (9)**, not the last (15) — by epoch 14 the model had got worse at generalising, and saving its final state would have shipped a worse model. And the honest caveat: picking the best of 15 noisy validation readings flatters the result, so 73.7% is a slightly lucky peak rather than a stable estimate.

> **Say:** "Training accuracy hit 92% while validation stalled at 74% — classic overfitting for a 98k-parameter network on 2,100 images. I keep the best-validation checkpoint rather than the last, and I'm upfront that picking the peak of a noisy curve flatters the number slightly."

## 7. Class imbalance and class weights

The training data is not balanced: 1,426 gliomas against 708 meningiomas. Left alone, a model can score respectably by leaning towards glioma whenever it is unsure — rewarded for counting, not for medicine.

Class weights make mistakes on rarer types count for more in the loss. The formula in `src/train.py` is `total / (n_classes × count)`: a balanced dataset gives every class 1.0, and a class half as common gets roughly double weight, so a missed meningioma hurts about twice as much as a missed glioma.

It was not enough. Meningioma recall is still 0.18. Class weights reshape the loss; they cannot create information the model can't extract from 708 slices.

> **Say:** "There are twice as many gliomas as meningiomas, so I weighted the loss inversely to class frequency. It helped, but didn't solve it, which is why I report per-class scores rather than just accuracy."

## 8. Precision and recall

Two different questions. For meningioma in experiment 1:

- **Recall** — of all the real meningiomas, how many were caught? 19 of 104 = **0.18**
- **Precision** — of everything called meningioma, how many really were? 19 of 26 = **0.73**

The gap is informative. When the model says "meningioma" it is usually right, it is just almost never willing to say it — a model that has become timid about a class, rather than one that confuses it.

Medically, missing a tumour (low recall) usually costs more than a false alarm: a false alarm is corrected by the next test, a miss goes home undiagnosed. A single accuracy figure can hide a class the model is failing completely.

> **Say:** "Recall is how many real cases I catch; precision is how often I'm right when I claim one. My meningioma precision is 0.73 but recall is 0.18 — when it commits it's usually correct, it just rarely commits."

## 9. The confusion matrix

Rows are the truth, columns are the prediction. The diagonal is correct answers; everything off it names a specific mistake.

| true ↓ / predicted → | Meningioma | Glioma | Pituitary | total |
|---|---|---|---|---|
| **Meningioma** | **19** | 49 | 36 | 104 |
| **Glioma** | 7 | **207** | 13 | 227 |
| **Pituitary** | 0 | 15 | **111** | 126 |

Read row by row, the failure names itself: of 104 real meningiomas, 19 correct, 49 misread as glioma and 36 as pituitary. The errors are scattered rather than one systematic confusion, which suggests the model has not formed a meningioma concept at all. The glioma and pituitary rows are strong.

The diagonal sums to 337 of 457, which is the 73.7% accuracy. That is exactly why the table is worth showing: the same model is 91% on one class and 18% on another, and one number concealed it.

> **Say:** "The confusion matrix is where the real story is. My accuracy looks acceptable at 74%, but almost all the error is one class — 85 of 104 meningiomas misfiled. That's what motivates experiment 2."
