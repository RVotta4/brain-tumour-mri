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

## 10. Transfer learning

ResNet-18 was trained on ImageNet: 1.28 million everyday photographs across 1,000 categories. That taught its filters to detect edges, corners, textures and blobs against backgrounds — and an edge is an edge whether it outlines a terrier or a tumour. The early layers of a photo network are genuinely useful for MRI even though it has never seen a brain.

SmallCNN had to invent all of that from 2,100 scans, with 98,019 parameters. ResNet-18 arrives already knowing it, with 11 million.

The result: 73.7% → 88.0%, and meningioma recall 0.18 → 0.84. One honest caveat. That swap changed two things at once — a bigger, better-designed architecture *and* pretrained starting weights — so it proves the pretrained ResNet-18 is better, not how much of the gain is the pretraining. Separating them would take one more run: ResNet-18 from random weights. I chose not to run it.

> **Say:** "ResNet-18 comes pretrained on 1.28 million photographs, so its early filters already detect edges and textures. Switching to it took me from 74% to 88%. To be precise, that changed the architecture and the starting weights together — isolating pretraining would need a ResNet trained from scratch as a control."

## 11. Freezing versus fine-tuning

There are two ways to reuse a pretrained network.

**Feature extraction (freezing):** lock every existing layer and train only a new final layer. The network becomes a fixed translator — image in, 512 numbers describing its visual content out. Fast, because no updates are computed for the 11 million frozen weights. Limited, because if ImageNet features don't capture what separates tumour types, training the last layer cannot conjure it.

**Fine-tuning:** nothing is locked, so every weight adapts to MRI. A higher ceiling, at the cost of more compute and more risk. This project fine-tunes.

> **Say:** "You can freeze the pretrained layers and train just a new head, which is fast but capped, or fine-tune everything, which is slower but adapts the features to your domain. I fine-tuned, because MRI is far enough from photographs that the features need to move."

## 12. Why the learning rate depends on where you start

The learning rate is the size of each nudge. The right size depends entirely on what the weights currently are.

Training SmallCNN from scratch, the weights began as random noise — big nudges were fine, because there was nothing valuable to damage. 1e-3 was sensible.

Fine-tuning starts from the distilled result of someone else's million-image training run. Nudges sized for random noise shove those carefully balanced filters around, damaging the very thing being borrowed. Convention is 1e-4 or lower.

Experiments 2 and 3 changed nothing but the rate. At 1e-3 validation accuracy swung across 38 points; at 1e-4 it stayed within 9, and accuracy rose from 88.0% to 93.0%.

> **Say:** "Training from scratch is writing on a blank page, so bold strokes are fine. Fine-tuning is editing a good draft — small careful corrections. Dropping the learning rate tenfold cut the swing in validation accuracy from 38 points to 9."

## 13. Reading a collapse: when one epoch guesses one class

Experiment 1's worst epoch scored 22.8% — exactly the fraction of validation scans that are meningioma. Experiment 2's worst scored 50.3%, next to the glioma share of 49.7%. A score that lands on a class's share is the fingerprint of a model answering (nearly) the same class for everything. Meanwhile training accuracy that same epoch was over 90%.

So the model was fine while training and broken while being evaluated. Two explanations fit:

- **Overshooting:** a learning rate too large, so each epoch's weights land somewhere different.
- **BatchNorm lag:** BatchNorm layers normalise with the current batch during training, but with a stored running average during evaluation. If the weights move quickly, that average goes stale, every number gets shifted at evaluation time, and predictions can pile into one class.

Lowering the learning rate cures both, so experiment 3 proves the fix rather than the cause. A cheap way to tell them apart would be to also score the *training* set in evaluation mode each epoch: if that collapses too, BatchNorm is the culprit.

> **Say:** "My worst epochs scored exactly one class's share of the data — the model was predicting one class for everything, while training accuracy was over 90%. That points at a train-versus-evaluation difference, likely BatchNorm statistics lagging a fast-moving model. A lower learning rate fixed it; I'm careful not to claim I've proved which cause it was."

## 14. Data augmentation

A model that has seen the same 2,100 scans fifteen times can simply memorise them — experiment 3 reached 100% training accuracy by epoch 5. Augmentation changes each scan slightly every time it is served: mirrored, tilted a few degrees, a little brighter or darker. The model never sees exactly the same image twice, so memorising pixels stops paying off and it has to learn what a tumour looks like.

The changes must be ones a real scanner could produce. A mirrored brain is still a realistic brain, a slight tilt is a patient's head position, and brightness varies between scanners. An upside-down brain never comes out of an MRI machine, so there are no vertical flips — teaching the model to handle impossible images wastes its capacity. Details matter too: rotation is smoothed so skull edges don't turn jagged, and the brightness change runs after rotation so the filled-in corners match the scan's own background instead of leaving a seam the model could latch onto.

Augmentation is applied to training only. Validation and test scans stay untouched, otherwise their scores would change from run to run for reasons unrelated to the model.

In experiment 4 it helped a little — 27 validation mistakes instead of 32 — but training accuracy still passed 99% by epoch 5. Gentle, realistic changes on 2,100 scans are not enough to stop an 11-million-parameter network memorising.

> **Say:** "Augmentation shows the model a slightly different version of each scan every epoch — mirrored, tilted, brightness shifted — only changes a real scanner could produce, and only on training data. It cut my validation mistakes from 32 to 27, but the network still memorised the training set, so the gain was too small to call."

## 15. Deciding the rule before seeing the result

Experiment 3 scored 93.0% on validation. If experiment 4 came in at 93.5%, is it better? Validation accuracy moved by up to 9 points from one epoch to the next in experiment 3, so half a point is noise.

The danger is choosing the rule after seeing the numbers — "higher accuracy wins" when that suits, "lower loss wins" when that suits. Each choice feels reasonable, and together they quietly pick whatever looks best. So the rule was written into the spec before experiment 4 ran: augmentation had to win by at least 2 points (95.0%), otherwise the simpler model stayed. In science this is called pre-registration.

Experiment 4 scored 94.1%. Higher, but short of the bar, so experiment 3 stayed the final model — even though "pick the bigger number" was tempting in the moment.

> **Say:** "I wrote the model-selection rule down before running the final experiment — it had to win by two points, or the simpler model stayed. Augmentation came in one point higher, so I kept the simpler model. Deciding the rule after seeing results lets you pick whatever looks best without meaning to."

## 16. Why the test set is used once

Every decision in this project — which epoch to keep, which learning rate, which model — was made by looking at validation scores. That makes validation a little flattering: the choices were tuned to it. The 34 test patients influenced nothing, which is the only reason their score is an honest estimate of performance on new patients.

That honesty survives exactly one look. Score the test set, adjust something, score again, and the test set has become a second validation set. So `src/score.py` refuses to score the test set if any experiment already has test results — and it did refuse, when a second attempt was made by accident.

Final model on test: 90.9%, against 93.0% on validation. The biggest drop was meningioma recall, 0.93 → 0.76 — the class that got the most attention during development was the most flattered by validation.

> **Say:** "The test set was touched once, at the very end, and the code refuses a second look. Validation was used for every decision, so it's slightly optimistic — 93% there, 90.9% on test. The biggest drop was meningioma, the class I'd focused on most, which is exactly the bias a held-out test set exists to catch."

## 17. Grad-CAM

A trained network gives an answer, not a reason. Grad-CAM recovers a rough "where". ResNet-18's last stage turns the scan into 512 pattern maps, each a 7×7 grid of how strongly one learned pattern appears where. The decision averages each map and weighs them into the three class scores.

Grad-CAM asks how much the winning score would rise if each map got stronger — the gradient — and uses that as the map's importance. The weighted maps are added up, negative evidence is dropped, and the 7×7 result is stretched over the scan. Red means "the evidence for this answer came from here".

It shows where, not why: a hot region says the model used that area, not what it saw there. Before trusting the hand-written version, it was checked against an independent hook-based implementation on real scans: identical heatmaps.

> **Say:** "Grad-CAM weights the last layer's pattern maps by how much each one pushed the winning class score, then overlays the result on the scan. I wrote it by hand — two halves of the forward pass and one gradient call — so I can explain exactly what the heatmap is."

## 18. Measuring attention: the pointing game and luck baselines

Looking at a few heatmaps proves little; anyone can pick flattering examples. The dataset includes the clinicians' tumour outline for every scan, so attention can be measured.

The pointing game asks whether the heatmap's hottest point lands on the tumour (with 8 pixels of leeway). It hit in 19.8% of test scans. On its own that number is meaningless — so it is compared with luck: a randomly placed point would hit 4.0% of the time, because that is how much of the image lies near a tumour. The in-mask share gets the same treatment: 2.5% of the heatmap fell inside the outline, 1.7 times what an evenly spread heatmap would manage.

The baseline is also what exposed the most interesting result. Pituitary tumours are classified correctly 95% of the time, yet the pointing game hit them in 0.8% of scans — *below* the 2.9% luck level. Without a baseline, "0.8%" just looks low; with one, it says the model is systematically looking away from the tumour it names correctly. That points to a shortcut, most likely the slice's position in the head.

> **Say:** "I measured attention against the clinicians' tumour outlines rather than eyeballing heatmaps. The model's peak lands on the tumour 20% of the time against 4% by chance — but for pituitary it's below chance despite 95% recall, which suggests it's recognising where the slice is, not the tumour."

## 19. Why the heatmap is coarse

ResNet-18's last stage sees the scan as a 7×7 grid, so each Grad-CAM cell covers 32×32 pixels — about 2% of the image. The tumours here cover a median of 0.7–1.6%. A heatmap cell is bigger than the typical tumour, so even perfect attention spreads well beyond the outline.

That is why the original "at least half the heatmap inside the tumour" test was dropped before running anything: it would have failed almost every scan and wrongly suggested the model ignores tumours. Spotting that a measurement cannot fit the data is part of the job.

> **Say:** "The heatmap's resolution is 7×7, and each cell is bigger than most of these tumours, so I replaced a 'half inside the outline' test that could never pass with the pointing game — the measurement has to fit the data."
