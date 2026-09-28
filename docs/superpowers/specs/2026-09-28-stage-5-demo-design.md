# Stage 5: Live Demo on Hugging Face Spaces — Design

**Date:** 2026-09-28
**Status:** Approved in brainstorming, awaiting written-spec review
**Overall design:** `2026-09-17-brain-tumour-mri-design.md` (section 7). This spec fills in that section.

## 1. Purpose

A live web page someone can try during an interview: upload a scan (or click an example), and see the final model's prediction, confidence bars for all three tumour types, and a Grad-CAM heatmap. It runs on Hugging Face Spaces' free CPU tier.

The model is frozen (see Stage 4): the demo shows experiment 3 exactly as scored on the test set and never changes it.

## 2. The page

- **Banner, always visible at the top:**
  - "Educational project — not a medical device, not for diagnosis."
  - "This model only knows glioma, meningioma and pituitary tumours. Given a healthy scan or a non-MRI image, it will still pick one of the three."
  - "Uploads are not saved; temporary copies are deleted within an hour."
- **Left column:**
  - Image upload (PNG or JPG). A note says DICOM files are not supported.
  - Six one-click example scans from the test set (section 4).
- **Right column:**
  - Predicted type and its confidence, as text.
  - Confidence bars for all three types (Gradio `Label`).
  - The scan with the Grad-CAM heatmap overlaid.
  - **For example scans only:** the clinician's tumour outline drawn on the overlay, plus a line giving the true type, whether the model was right, and the example's one-line note.
- **Footer:** dataset credit (Cheng et al., Figshare, CC BY 4.0, with link) and a link to the GitHub repo.

## 3. Processing an image

1. Convert the image to greyscale (colour images and images with transparency included).
2. `preprocess_image()` from `src/data.py`: min-max stretch to 0–255 and resize to 224×224. Non-square images are squashed, as in training.
3. Divide by 255 to get values in [0, 1], shape (1, 224, 224).
4. `grad_cam()` gives the heatmap, predicted class and confidence. The three confidences for the bars come from the softmax of the same forward pass.

These are the same steps the test scans went through, so clicking an example reproduces the confidence recorded for that scan in `experiments/03-resnet18-finetuned/explain/test_scans.csv`. This is the check that the demo runs the real model.

**Recognising an example.** The app compares the prepared 224×224 pixels with each example scan. If they are identical, it is that example, and its outline and note are shown. This also works if someone downloads an example and uploads it again. Any other image is treated as an upload with no outline.

## 4. Example scans

- **Six scans from the test set:**
  - one correctly classified scan per tumour type (3);
  - three real mistakes, including at least one confident mistake from patient 107946 (meningioma called something else at 0.97–1.00).
- **Chosen by hand, and shown to Robbi before they are committed.** Choosing by hand is fine here: the examples illustrate behaviour already reported in the README, they are not a measurement. Each gets a one-line note, e.g. "Correct, but the heatmap misses the tumour — see the README's pituitary finding."
- **Files in `app/examples/`:**
  - `<name>.png`: the 224×224 scan exactly as stored in `data/cheng_224.npz`;
  - `<name>_mask.png`: the clinician outline (0 or 255);
  - `examples.csv`: file name, patient ID, true type, note.
- These are small PNGs taken from a CC BY 4.0 dataset and are committed to GitHub, with attribution on the page.

## 5. Code

| File | Contents |
|---|---|
| `src/gradcam.py` (new) | `feature_maps`, `class_scores`, `grad_cam`, moved unchanged from `src/explain.py`. `explain.py` imports them from here, so its behaviour and tests are unchanged. This keeps the Space from needing the training, scoring and plotting code. |
| `app/demo.py` | The logic, with no Gradio import: load the model, prepare an image, predict with heatmap, draw overlay and outline, load and look up examples. |
| `app/app.py` | The Gradio page only (`gr.Blocks`), calling `demo.py`. No flagging. Temporary upload copies deleted hourly via `delete_cache`. Nothing prints or logs images. |
| `app/make_examples.py` | Copies the chosen test scans and outlines from `cheng_224.npz` into `app/examples/` and writes `examples.csv`. |
| `app/deploy.py` | Creates the Space if missing (public, Gradio, free CPU) and uploads everything it needs in one commit (section 6). |
| `app/requirements.txt` | The Space's dependencies, pointing pip at PyTorch's CPU-only build via `--extra-index-url https://download.pytorch.org/whl/cpu`. Without this, Linux installs the ~2 GB GPU build. |
| `app/README.md` | The Space's page, starting with the settings header Hugging Face reads (title, `sdk: gradio`, pinned `sdk_version`, `app_file: app.py`). |

**Loading the model.** `build_model("resnet18", pretrained=False)` then `load_state_dict(torch.load(path, weights_only=True))`, as `score.py` does. `pretrained=False` avoids downloading ImageNet weights, which would be overwritten anyway. The model file is looked for next to `app.py` first (the Space layout), then at `experiments/03-resnet18-finetuned/model.pt` (the laptop layout), so the same code runs in both places. If neither exists, the app stops with a clear message.

**Imports.** `app.py` and `demo.py` import `src.*`. On the laptop the app runs from the repo root as `python -m app.app`; on the Space, `src/` sits next to `app.py`. Both layouts resolve the same imports.

**Local install.** Gradio goes in `app/requirements.txt`, not the main `requirements.txt`. Robbi installs it into the existing `.venv` for local runs (about 150–250 MB). The pytest suite does not need it, because tests only import `app/demo.py`.

## 6. Deploying

1. Robbi runs `hf auth login` once and pastes a write-access token created on the Hugging Face website. The token never passes through Claude.
2. Robbi runs `python -m app.deploy --space <username>/brain-tumour-mri`. In one commit it uploads:
   - from `app/`: `app.py`, `demo.py`, `requirements.txt`, `README.md`, `examples/`;
   - from `src/`: `__init__.py`, `models.py`, `data.py`, `gradcam.py`;
   - `experiments/03-resnet18-finetuned/model.pt` as `model.pt`.
3. Hugging Face builds the Space (a few minutes). It is then live at `https://huggingface.co/spaces/<username>/brain-tumour-mri`.

`model.pt` goes only to Hugging Face, never to GitHub. Re-running the deploy command updates the Space.

**Free-tier behaviour.** A Space with no visitors for about 48 hours sleeps, and the next visitor waits roughly a minute while it wakes. Open the link a few minutes before an interview.

## 7. Privacy

- No flagging button, no dataset saving, no logging of images.
- Gradio keeps a temporary copy of each upload on the server while processing it. The app sets `delete_cache` so these are deleted within an hour. The banner states this rather than claiming uploads are never stored.

## 8. Testing

`pytest`, with no Gradio, dataset or trained model needed:

- **Prepare:** colour, odd-sized and constant (all one value) images all become (1, 224, 224) in [0, 1] without errors.
- **Predict:** with a random-weight `ResNet18Grey`, the three confidences sum to 1, the predicted class matches the largest, and the heatmap is 224×224 in [0, 1].
- **Overlay:** returns a 224×224×3 uint8 image; outline colour appears only on the mask's edge pixels; with no mask, no outline colour is drawn.
- **Example lookup:** an identical image returns its example; the same image with one pixel changed returns none.
- **Examples folder:** every row of `examples.csv` has its scan and mask files, a valid tumour type, and 224×224 sizes.
- **Refactor:** the existing 63 tests still pass after the move to `src/gradcam.py`.

**Manual checks (Robbi):**
1. Local: `python -m app.app`, open `http://localhost:7860`, click every example and compare its confidence with `test_scans.csv`; upload a non-MRI image and confirm it still predicts one of the three types.
2. Live: the same checks on the deployed Space.

## 9. Pauses and write-up

- **Pause 1:** after the six examples are chosen, before committing them.
- **Pause 2:** after the local run.
- **Pause 3:** after the Space is live.
- **Write-up:**
  - README: demo link and screenshot at the top, status line updated, "How to run" gains the local demo command.
  - Learning notes 20 onwards, each with a **Say:** line: what a Space and Gradio are; why CPU-only PyTorch; how the demo reproduces the test pipeline; the privacy caveat.
- Merge to `main` and push after Robbi approves the wording.

## 10. Out of scope

- DICOM or 3D volume input.
- Retraining, calibration or an "unknown / not a tumour" class; the model stays frozen.
- Automatic sync from GitHub to the Space.
- Paid or GPU hardware.
