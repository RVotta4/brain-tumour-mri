# Stage 5b: The Live Demo Runs in the Browser — Design

**Date:** 2026-09-28
**Status:** Approved in conversation, awaiting written-spec review
**Replaces:** the hosting parts of `2026-09-28-stage-5-demo-design.md` (sections 5 `app/app.py`/`app/README.md`/`app/requirements.txt`/`app/deploy.py`, 6 and 7). The page (section 2), the image steps (section 3) and the examples (section 4) are unchanged.

## 1. Why the change

The first deploy failed with `402 Payment Required`. Since 27 September 2026, Hugging Face requires a PRO subscription for Gradio and Docker Spaces, even on the free CPU hardware. Free accounts can still create **static** Spaces. A static Space only sends files to the visitor's browser; it runs no Python.

Other free Python hosts were checked and rejected: Vercel's Python functions are capped at 500 MB (PyTorch is larger, and Gradio needs a long-running server); Render and Koyeb give 512 MB of memory, which PyTorch plus Gradio is likely to exceed; Google Cloud Run needs a billing card; free ZeroGPU Spaces need an account older than 30 days and PyTorch ≤ 2.13.

So the demo becomes a static web page, and **the model runs in the visitor's browser** with ONNX Runtime Web. The page looks and behaves as designed in Stage 5, and it is still free.

## 2. What this gains and costs

- **Privacy:** an uploaded scan never leaves the visitor's device. Nothing is sent anywhere.
- **No server:** nothing to sleep or wake up. The first visit downloads the 45 MB model once.
- **Costs:** a JavaScript version of the image steps, which must be tested against the Python version.
- **Small, stated differences for uploads only:** the browser reduces 16-bit images to 8-bit, and its JPEG decoder may differ from Python's by a grey level here and there. The example scans (PNG) are unaffected.

## 3. The exported model (`app/export_onnx.py`)

`BrowserModel` wraps the final model and returns two outputs for a (1, 1, 224, 224) scan in [0, 1]:
- `scores`: the three class scores, as before;
- `heatmaps`: one heatmap per tumour type, (1, 3, 224, 224).

**Why no backward pass is needed.** ResNet-18 ends by averaging each of its 512 7×7 maps and weighting the averages with the last layer. So the gradient Grad-CAM computes for map *k* is just that last-layer weight divided by 49. Weighting the maps by the last-layer weights, dropping negatives and enlarging to 224×224 gives Grad-CAM's heatmap exactly, once it is scaled so its peak is 1. (Checked on the six examples: largest difference 0.00001.)

The file is saved as one ONNX file with the weights inside (44.8 MB), at `experiments/03-resnet18-finetuned/model.onnx`, gitignored like `model.pt`. After exporting, the script runs the six examples through the exported file with ONNX Runtime and compares them with `test_scans.csv`, printing MATCH or MISMATCH.

New dependencies (main `requirements.txt`): `onnx`, `onnxscript`, `onnxruntime`.

## 4. The web page (`space/`)

| File | Contents |
|---|---|
| `space/index.html` | Layout: the banner, upload box, examples, results area, credits. Loads ONNX Runtime Web 1.30.0 from the jsDelivr CDN. |
| `space/style.css` | Two columns on a laptop, one on a phone; follows the system's light or dark mode. |
| `space/pipeline.js` | The image steps from `app/demo.py`, as pure functions: greyscale, example lookup, stretch and resize, heatmap colours, outline, overlay, softmax, summary text. No browser APIs, so Node can test it. |
| `space/app.js` | The wiring: decode images, load the examples and the model, run it, draw the results. |
| `space/README.md` | The Space's settings (`sdk: static`) and description. |
| `space/tests/pipeline.test.mjs` | Node tests for `pipeline.js`. |

**The page.** As in Stage 5 section 2, except the banner's third line becomes: "The model runs in your browser: images you upload never leave your device." A status line shows loading progress ("Loading the model…", then "Ready").

**Matching Python exactly.** `pipeline.js` copies the arithmetic of the Python libraries, not just the idea:
- greyscale uses Pillow's integer formula;
- the resize reproduces Pillow's bilinear resampling (including its smoothing when shrinking), one axis at a time, rounding to 32-bit floats as Pillow does;
- rounding is round-half-to-even, as NumPy's;
- the heatmap colours are Matplotlib's 256-entry "jet" table, built the way Matplotlib builds it.

In the prototype, all of these matched Python pixel for pixel on six image sizes, including 999×37.

**Recognising an example.** Same rule as Stage 5: a 224×224 image whose grey pixels equal an example's uses that example's pixels unchanged, and gets its outline and note.

**Model runtime.** ONNX Runtime Web on WebAssembly, single-threaded (a static Space can't send the headers multi-threading needs). ResNet-18 takes well under a second.

## 5. Testing

- **pytest** (`tests/test_export.py`): the `BrowserModel` heatmap equals `grad_cam` on a random-weight model; an exported file is a single file and gives the same scores and heatmaps as PyTorch.
- **pytest runs the Node tests** (`tests/test_space.py`): it writes the Python "correct answers" to a temporary file (greyscale and prepared pixels for five image sizes, the jet table, and an overlay with and without an outline), then runs `node --test` on `space/tests/pipeline.test.mjs` and requires every test to pass with none skipped. It is skipped if Node isn't installed.
- **Node-only tests** that need no Python: rounding, outline pixel counts, softmax, example lookup, heatmap scaling, summary text.
- **Manual checks (Robbi):** the export's MATCH output; the local preview (every example matches its recorded confidence; a non-MRI upload still predicts a type); the live Space, including on a phone.

## 6. Deploying (`app/deploy.py`, rewritten)

- `site_files()` lists everything the Space serves: the four page files, `README.md`, `model.onnx`, the 12 example PNGs, and `examples.json`. The JSON is generated from `app/examples/examples.csv` (name, patient, true type, caption, note), so the example text lives in one place. That makes 19 files.
- `--preview <folder>` writes exactly those files into a local folder, to be viewed with `python -m http.server`. The folder `space_preview/` is gitignored.
- `--space <user>/<name>` creates a public **static** Space if needed and uploads the files in one commit. The model goes to Hugging Face only.
- **If the Space can't serve the 45 MB model file**, the fallback is a free Hugging Face model repository for `model.onnx`, with `app.js` loading it from there.

## 7. What stays and what goes

- **Stays:** `app/demo.py` and its tests (the reference the browser version is checked against); `app/app.py`, the Gradio page, as the way to run the demo locally in Python; `app/examples/`; `app/make_examples.py`.
- **Goes:** `app/README.md` and `app/requirements.txt`, which only served the Gradio Space; the deploy test that checked which Python files the Space needed.

## 8. Pauses

1. After the export: the six MATCH lines.
2. After the local preview.
3. After the Space is live.

Then the write-up (README link and screenshot, learning notes) and merge, as in Stage 5.
