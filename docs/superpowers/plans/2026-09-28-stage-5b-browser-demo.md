# Stage 5b: Browser Demo on a Static Space — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The live demo becomes a free static Hugging Face Space: a web page where the final model runs in the visitor's browser (ONNX Runtime Web), giving pixel-identical results to the Python demo.

**Architecture:** `app/export_onnx.py` saves the model as one ONNX file that returns scores and per-type heatmaps (for ResNet-18, Grad-CAM needs no backward pass). `space/pipeline.js` reproduces `app/demo.py`'s image steps exactly and is tested by Node against answers written by the Python code; `space/app.js` wires the page. `app/deploy.py` previews the site locally or uploads it to a static Space.

**Tech Stack:** Python 3.13, PyTorch 2.14 CPU, onnx 1.23, onnxscript 0.7.2, onnxruntime 1.30.0; plain HTML/CSS/JavaScript (ES modules), ONNX Runtime Web 1.30.0 from jsDelivr; Node 24's built-in test runner; huggingface_hub 1.33.

**Spec:** `docs/superpowers/specs/2026-09-28-stage-5b-browser-demo-design.md` (replaces the hosting parts of `2026-09-28-stage-5-demo-design.md`).

---

## Ground rules for whoever executes this

- **Platform:** Windows. Commands run from `C:\Users\Robbi\brain-tumour-mri`. Use `.\.venv\Scripts\python.exe` (bare `python` opens the Microsoft Store). Node 24 is installed as `node`.
- **Branch:** `feat/stage-5-demo` (Tasks 1–7 of the Stage 5 plan are committed there; 82 tests pass at `43a10a4`).
- **Frozen model:** nothing trains, re-scores, or modifies `experiments/` except writing `experiments/03-resnet18-finetuned/model.onnx` in Task 2.
- **Already installed in the venv:** onnx 1.23.0, onnxscript 0.7.2, onnxruntime 1.30.0.
- **Code tasks run without check-ins.** Tasks 2, 6 and 7 end in a PAUSE; Robbi runs the commands marked **(Robbi)**.
- **Commit messages** end with a second `-m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"`.
- **Write every file as UTF-8** (the texts contain "—" and "…").
- **Verified in a prototype on 2026-09-28:**
  - `torch.onnx.export(..., dynamo=True, external_data=False, verbose=False)` writes a single 44,789,063-byte file. `verbose=False` is required on Windows: the exporter's progress messages contain an emoji the console can't print.
  - The exported file reproduces the six examples' confidences to 5 decimals; its heatmaps equal `grad_cam`'s to within 1.1e-5.
  - The `pipeline.js` code below matched Python exactly (zero differing pixels) for greyscale and preparation at 300×200, 512×512, 100×150, 224×224, 37×999 and 640×480, for the jet table (to 1e-15), and for the overlay with outline.

## File map

| File | Change |
|---|---|
| `app/export_onnx.py` | New: `BrowserModel`, `export`, `run_onnx`, `check_examples` |
| `tests/test_export.py` | New |
| `requirements.txt`, `.gitignore` | Add ONNX packages; ignore `model.onnx` and `space_preview/` |
| `space/pipeline.js` | New: the image steps |
| `space/tests/pipeline.test.mjs` | New: Node tests |
| `tests/test_space.py` | New: writes Python reference answers, runs the Node tests |
| `space/index.html`, `space/style.css`, `space/app.js` | New: the page |
| `space/README.md` | New: the Space's settings |
| `app/deploy.py`, `tests/test_deploy.py` | Rewritten for a static Space |
| `app/README.md`, `app/requirements.txt` | Deleted |
| `README.md`, `docs/learning-notes.md` | Task 8 |

---

### Task 1: Export the model for the browser

**Concept:** ResNet-18's decision is an average of 512 maps weighted by its last layer. Grad-CAM's gradient for each map is therefore just that weight divided by 49, so weighting the maps by the last-layer weights gives the same heatmap without any backward pass. The exported model returns scores and all three heatmaps in one forward pass, which a browser can run.

**Files:**
- Create: `app/export_onnx.py`, `tests/test_export.py`
- Modify: `requirements.txt`, `.gitignore`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_export.py`:

```python
import numpy as np
import pytest
import torch

from app.export_onnx import BrowserModel, export, run_onnx
from src.gradcam import grad_cam
from src.models import ResNet18Grey


def random_model():
    torch.manual_seed(0)
    return ResNet18Grey(weights=None).eval()


def scaled(heatmap):
    peak = heatmap.max()
    return heatmap / peak if peak > 0 else heatmap


def test_the_browser_heatmap_equals_grad_cam():
    model = random_model()
    scan = torch.rand(1, 224, 224)

    expected, predicted, _ = grad_cam(model, scan)
    with torch.no_grad():
        scores, heatmaps = BrowserModel(model)(scan.unsqueeze(0))

    assert int(scores.argmax()) == predicted
    np.testing.assert_allclose(scaled(heatmaps[0, predicted].numpy()), expected, atol=1e-4)


def test_the_exported_file_gives_the_same_answers(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    model = random_model()

    export(model, tmp_path / "model.onnx")

    assert not (tmp_path / "model.onnx.data").exists()  # one file, weights inside
    pixels = np.random.default_rng(3).integers(0, 256, size=(224, 224), dtype=np.uint8)
    scores, heatmaps = run_onnx(onnxruntime.InferenceSession(str(tmp_path / "model.onnx")), pixels)
    with torch.no_grad():
        expected_scores, expected_heatmaps = BrowserModel(model)(torch.from_numpy(pixels / 255.0).float()[None, None])
    np.testing.assert_allclose(scores, expected_scores[0].numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(heatmaps, expected_heatmaps[0].numpy(), rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Run to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_export.py -v
```
Expected: collection error, `ModuleNotFoundError: No module named 'app.export_onnx'`.

- [ ] **Step 3: Create `app/export_onnx.py`**

```python
"""Export the final model for the browser demo, and check the export.

Usage:
    .\\.venv\\Scripts\\python.exe -m app.export_onnx

Writes experiments/03-resnet18-finetuned/model.onnx (uploaded to Hugging Face,
never to GitHub), then runs the six example scans through the exported file
and compares its answers with those recorded in the one-shot test run.
"""

import csv

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.demo import MODEL_PATH, load_examples, load_model
from src.data import CLASS_NAMES
from src.gradcam import class_scores, feature_maps

ONNX_PATH = MODEL_PATH.with_suffix(".onnx")
TEST_SCANS = MODEL_PATH.parent / "explain" / "test_scans.csv"


class BrowserModel(nn.Module):
    """The final model, returning the three scores and a heatmap for each tumour type.

    For this network Grad-CAM needs no backward pass. Each class score is an
    average of the 512 pattern maps, weighted by the last layer's weights, so
    the gradient Grad-CAM uses for map k is just that weight divided by 49
    (the 7x7 cells). Weighting the maps by the last layer's weights therefore
    gives Grad-CAM's heatmap, once it is scaled so its peak is 1. That lets a
    browser compute it with a single forward pass.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, scan):
        maps = feature_maps(self.model, scan)
        scores = class_scores(self.model, maps)
        weights = self.model.backbone.fc.weight  # (3 tumour types, 512 maps)
        cams = torch.relu(torch.einsum("ck,bkhw->bchw", weights, maps))  # keep evidence for, drop evidence against
        heatmaps = F.interpolate(cams, size=scan.shape[-2:], mode="bilinear", align_corners=False)
        return scores, heatmaps


def export(model, path):
    """Save the model as one ONNX file taking a (1, 1, 224, 224) scan with values in [0, 1].

    verbose=False keeps the exporter quiet; its progress messages include an
    emoji that the Windows console can't print.
    """
    torch.onnx.export(BrowserModel(model).eval(), (torch.zeros(1, 1, 224, 224),), path,
                      input_names=["scan"], output_names=["scores", "heatmaps"],
                      dynamo=True, external_data=False, verbose=False)


def run_onnx(session, pixels):
    """Scores (3,) and heatmaps (3, 224, 224) for one uint8 scan, from the exported model."""
    scan = (pixels.astype(np.float32) / 255.0)[None, None]  # as in training: values in [0, 1]
    scores, heatmaps = session.run(None, {"scan": scan})
    return scores[0], heatmaps[0]


def check_examples(path=ONNX_PATH):
    """Compare the exported model's answers on the examples with the recorded test run."""
    import onnxruntime  # only needed for this check

    session = onnxruntime.InferenceSession(str(path))
    with open(TEST_SCANS, newline="") as f:
        recorded = {int(row["index"]): row for row in csv.DictReader(f)}
    all_match = True
    for example in load_examples():
        scores, _ = run_onnx(session, example.scan)
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        predicted = CLASS_NAMES[int(probabilities.argmax())]
        confidence = float(probabilities.max())
        row = recorded[example.index]
        same = predicted == row["predicted"] and abs(confidence - float(row["confidence"])) < 1e-3
        all_match &= same
        print(f"{example.name:30} exported {predicted:10} {confidence:.4f} | test run {row['predicted']:10} "
              f"{row['confidence']} | {'MATCH' if same else 'MISMATCH'}")
    return all_match


def main():
    export(load_model(), ONNX_PATH)
    print(f"Wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
    if not check_examples():
        raise SystemExit("The exported model does not reproduce the test run. Don't deploy it.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Dependencies and ignores**

Append to `requirements.txt`:

```text
# Exporting the final model for the browser demo (app/export_onnx.py).
onnx
onnxscript
onnxruntime
```

Append to `.gitignore`:

```text
experiments/*/model.onnx
space_preview/
```

- [ ] **Step 5: Run the tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_export.py -v
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: 2 passed, then `84 passed`.

- [ ] **Step 6: Commit**

```powershell
git add app/export_onnx.py tests/test_export.py requirements.txt .gitignore
git commit -m "feat: export the final model to ONNX for the browser demo" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2 (Robbi): export the real model — then PAUSE 1

- [ ] **Step 1 (Robbi): export and check**

```powershell
.\.venv\Scripts\python.exe -m app.export_onnx
```
Expected: `Wrote ...\model.onnx (44.8 MB)`, then six lines ending `MATCH`. Warnings from the exporter above them are harmless. Any `MISMATCH` (the script then exits with an error): stop and debug (superpowers:systematic-debugging).

- [ ] **Step 2: PAUSE 1** — show Robbi the MATCH lines. `git status` must not list `model.onnx` (it's gitignored).

---

### Task 3: The image steps in JavaScript, tested against Python

**Concept:** the browser must prepare images exactly as the Python code did, or the model sees different pixels. `pipeline.js` copies the arithmetic of Pillow, NumPy and Matplotlib, not just the idea. The Python side writes its answers to a file, and Node checks the JavaScript gives the same ones. pytest runs the Node tests, so one command still runs everything.

**Files:**
- Create: `space/pipeline.js`, `space/tests/pipeline.test.mjs`, `tests/test_space.py`

- [ ] **Step 1: Write the Node tests**

Create `space/tests/pipeline.test.mjs`:

```javascript
// Tests for pipeline.js. Run them through pytest (tests/test_space.py), which first
// writes the Python code's answers to a file and passes its path in FIXTURES.

import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";

import * as pipeline from "../pipeline.js";

const fixtures = process.env.FIXTURES ? JSON.parse(fs.readFileSync(process.env.FIXTURES, "utf8")) : null;
const needsFixtures = { skip: fixtures ? false : "run through pytest, which writes the Python answers" };

test("halves round to the nearest even number, like NumPy", () => {
  assert.deepEqual([0.5, 1.5, 2.5, 3.5, 2.4, 2.6].map(pipeline.roundHalfEven), [0, 2, 2, 4, 2, 3]);
});

test("the outline is the tumour's border", () => {
  const mask = new Uint8Array(64 * 64);
  for (let y = 20; y < 30; y++) for (let x = 20; x < 30; x++) mask[y * 64 + x] = 1; // a 10x10 square
  const count = (pixels) => pixels.reduce((total, v) => total + v, 0);

  assert.equal(count(pipeline.outline(mask, 64, 64, 1)), 36); // 100 minus the 8x8 inside
  assert.equal(count(pipeline.outline(mask, 64, 64, 2)), 64); // 100 minus the 6x6 inside
  assert.equal(count(pipeline.outline(new Uint8Array(64 * 64), 64, 64)), 0);
});

test("softmax gives probabilities that sum to 1, in score order", () => {
  const probabilities = pipeline.softmax([2, 1, 0]);

  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(probabilities[0] > probabilities[1] && probabilities[1] > probabilities[2]);
});

test("an example is recognised only when every pixel matches", () => {
  const scan = Uint8Array.from({ length: 224 * 224 }, (_, i) => i % 251);
  const example = { name: "e", scan };
  const altered = scan.slice();
  altered[0] += 1;

  assert.equal(pipeline.findExample(scan.slice(), 224, 224, [example]), example);
  assert.equal(pipeline.findExample(altered, 224, 224, [example]), null);
  assert.equal(pipeline.findExample(scan.slice(0, 100), 10, 10, [example]), null);
});

test("the predicted type's heatmap is scaled so its peak is 1", () => {
  const heatmaps = new Float32Array(3 * 4);
  heatmaps.set([0, 1, 2, 4], 4); // the second type's map

  assert.deepEqual(Array.from(pipeline.normaliseHeatmap(heatmaps, 1, 4)), [0, 0.25, 0.5, 1]);
  assert.deepEqual(Array.from(pipeline.normaliseHeatmap(new Float32Array(12), 0, 4)), [0, 0, 0, 0]);
});

test("the summary says whether an example was right", () => {
  const example = { true: "glioma", patient_id: "P1", note: "A note." };

  const right = pipeline.describe("glioma", 0.93, example);
  assert.equal(right.headline, "Prediction: glioma (93.0% confidence)");
  assert.match(right.lines[0], /correct/);

  const wrong = pipeline.describe("meningioma", 0.99, example);
  assert.match(wrong.lines[0], /mistake/);
  assert.equal(wrong.lines[1], "A note.");

  const upload = pipeline.describe("pituitary", 0.5, null);
  assert.match(upload.lines[0], /outline/);
  assert.doesNotMatch(upload.lines.join(" "), /test patient/);
});

test("the same tumour types, in the same order, as the Python code", needsFixtures, () => {
  assert.deepEqual(pipeline.CLASS_NAMES, fixtures.class_names);
});

test("greyscale and preparation match Python pixel for pixel", needsFixtures, () => {
  for (const c of fixtures.cases) {
    const grey = pipeline.toGrey(c.rgba, c.width, c.height);
    assert.deepEqual(Array.from(grey), c.grey, `greyscale, ${c.width}x${c.height}`);
    assert.deepEqual(Array.from(pipeline.preprocess(grey, c.width, c.height)), c.pixels, `prepared, ${c.width}x${c.height}`);
  }
});

test("the jet colours match Matplotlib's table", needsFixtures, () => {
  fixtures.jet.forEach((rgb, i) => rgb.forEach((value, channel) => {
    assert.ok(Math.abs(value - pipeline.JET_TABLE[channel][i]) < 1e-12, `entry ${i}, channel ${channel}`);
  }));
});

test("the overlay matches Python pixel for pixel", needsFixtures, () => {
  const o = fixtures.overlay;
  const pixels = Uint8Array.from(o.pixels);
  const heatmap = Float32Array.from(o.heatmap);
  for (const [mask, expected] of [[Uint8Array.from(o.mask), o.with_outline], [null, o.without_outline]]) {
    const rgb = Array.from(pipeline.overlay(pixels, heatmap, mask)).filter((_, i) => i % 4 !== 3); // drop alpha
    assert.deepEqual(rgb, expected);
  }
});
```

- [ ] **Step 2: Write the pytest wrapper**

Create `tests/test_space.py`:

```python
"""Checks the browser demo's image steps (space/pipeline.js) against the Python code.

The Python functions write their answers to a file; the Node tests compare
the JavaScript's answers with it.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from matplotlib import colormaps
from PIL import Image

from app.demo import load_examples, overlay
from src.data import CLASS_NAMES, preprocess_image

ROOT = Path(__file__).resolve().parent.parent
SIZES = [(300, 200), (100, 150), (37, 99), (224, 224), (400, 333)]  # (height, width): shrink, enlarge, odd, same


def reference_cases():
    rng = np.random.default_rng(0)
    for height, width in SIZES:
        rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        rgb[: height // 3] //= 3  # a darker band, so it isn't pure noise
        grey = np.asarray(Image.fromarray(rgb).convert("L"))
        yield {
            "width": width,
            "height": height,
            "rgba": np.dstack([rgb, np.full((height, width), 255, np.uint8)]).ravel().tolist(),
            "grey": grey.ravel().tolist(),
            "pixels": preprocess_image(grey.astype(np.float32)).ravel().tolist(),
        }


def write_fixtures(path):
    example = load_examples()[2]
    rows, cols = np.mgrid[0:224, 0:224]
    heatmap = np.exp(-((rows - 90) ** 2 + (cols - 140) ** 2) / 900.0).astype(np.float32)
    heatmap /= heatmap.max()
    fixtures = {
        "class_names": CLASS_NAMES,
        "cases": list(reference_cases()),
        "jet": colormaps["jet"](np.arange(256) / 255.0)[:, :3].tolist(),
        "overlay": {
            "pixels": example.scan.ravel().tolist(),
            "mask": example.mask.ravel().tolist(),
            "heatmap": heatmap.ravel().tolist(),
            "with_outline": overlay(example.scan, heatmap, example.mask).ravel().tolist(),
            "without_outline": overlay(example.scan, heatmap).ravel().tolist(),
        },
    }
    path.write_text(json.dumps(fixtures), encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_browser_pipeline_matches_python(tmp_path):
    fixtures = tmp_path / "fixtures.json"
    write_fixtures(fixtures)

    result = subprocess.run(
        ["node", "--test", "--test-reporter=tap", "space/tests/pipeline.test.mjs"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "FIXTURES": str(fixtures)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "# skip 0" in result.stdout, "some browser tests were skipped"
```

- [ ] **Step 3: Run to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_space.py -v
```
Expected: FAIL; the output includes Node's `Cannot find module ...space/pipeline.js`.

- [ ] **Step 4: Create `space/pipeline.js`**

```javascript
// The browser demo's image steps: the same arithmetic as app/demo.py and the
// libraries it uses (Pillow, NumPy, Matplotlib), so the model sees exactly the
// pixels it saw in Python. No browser APIs here, so Node can test every function.

export const CLASS_NAMES = ["meningioma", "glioma", "pituitary"]; // src/data.py's order
export const SIZE = 224;
export const HEAT_ALPHA = 0.45; // how strongly the heatmap tints the scan
export const OUTLINE_COLOUR = [0, 255, 0]; // lime
export const OUTLINE_WIDTH = 2; // pixels

const f32 = Math.fround; // round to a 32-bit float, as NumPy's float32 and Pillow's "F" images do

/** Pillow's convert("L"): grey levels from RGBA pixels, with Pillow's integer rounding. */
export function toGrey(rgba, width, height) {
  const grey = new Uint8Array(width * height);
  for (let i = 0; i < grey.length; i++) {
    const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
    grey[i] = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
  }
  return grey;
}

/** NumPy's np.round: exact halves go to the nearest even number. */
export function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Pillow's bilinear weights for one axis. When shrinking, the filter widens to smooth. */
function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = filterScale; // bilinear reaches 1 pixel, times the widening
  const inverse = 1 / filterScale;
  const rows = [];
  for (let out = 0; out < outSize; out++) {
    const centre = (out + 0.5) * scale;
    const first = Math.max(0, Math.trunc(centre - support + 0.5));
    const last = Math.min(inSize, Math.trunc(centre + support + 0.5));
    const weights = [];
    let total = 0;
    for (let x = first; x < last; x++) {
      const w = Math.max(0, 1 - Math.abs((x - centre + 0.5) * inverse));
      weights.push(w);
      total += w;
    }
    rows.push({ first, weights: weights.map((w) => (total !== 0 ? w / total : w)) });
  }
  return rows;
}

/** Pillow's resize(BILINEAR) of a float image: across, then down, each result rounded to 32 bits. */
function resize(image, width, height, outWidth, outHeight) {
  let current = image;
  let rowLength = width;
  if (outWidth !== width) {
    const columns = coefficients(width, outWidth);
    const next = new Float32Array(outWidth * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < outWidth; x++) {
        const { first, weights } = columns[x];
        let sum = 0;
        for (let k = 0; k < weights.length; k++) sum += current[y * rowLength + first + k] * weights[k];
        next[y * outWidth + x] = sum;
      }
    }
    current = next;
    rowLength = outWidth;
  }
  if (outHeight !== height) {
    const rows = coefficients(height, outHeight);
    const next = new Float32Array(rowLength * outHeight);
    for (let y = 0; y < outHeight; y++) {
      const { first, weights } = rows[y];
      for (let x = 0; x < rowLength; x++) {
        let sum = 0;
        for (let k = 0; k < weights.length; k++) sum += current[(first + k) * rowLength + x] * weights[k];
        next[y * rowLength + x] = sum;
      }
    }
    current = next;
  }
  return current;
}

/** src/data.py's preprocess_image: stretch to [0, 1], resize to 224x224, back to 0-255. */
export function preprocess(grey, width, height) {
  let low = Infinity;
  let high = -Infinity;
  for (const v of grey) {
    if (v < low) low = v;
    if (v > high) high = v;
  }
  const stretched = new Float32Array(grey.length); // stays all zero for a blank image
  if (high > low) {
    const range = f32(high - low);
    for (let i = 0; i < grey.length; i++) stretched[i] = f32(f32(grey[i] - low) / range);
  }
  const resized = resize(stretched, width, height, SIZE, SIZE);
  const pixels = new Uint8Array(SIZE * SIZE);
  for (let i = 0; i < pixels.length; i++) pixels[i] = Math.min(255, Math.max(0, roundHalfEven(f32(resized[i] * 255))));
  return pixels;
}

/** The example whose 224x224 scan equals these grey pixels exactly, or null. */
export function findExample(grey, width, height, examples) {
  if (width !== SIZE || height !== SIZE) return null;
  return examples.find((example) => example.scan.length === grey.length && example.scan.every((v, i) => v === grey[i])) ?? null;
}

/** Scores to probabilities that sum to 1. */
export function softmax(scores) {
  const peak = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - peak));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

/** One tumour type's heatmap from the model's (3, 224, 224) output, scaled so its peak is 1. */
export function normaliseHeatmap(heatmaps, type, pixels = SIZE * SIZE) {
  const heatmap = heatmaps.slice(type * pixels, (type + 1) * pixels);
  let peak = 0;
  for (const v of heatmap) if (v > peak) peak = v;
  return peak > 0 ? heatmap.map((v) => v / peak) : heatmap;
}

// Matplotlib's "jet" colour map, as the points it is defined by: [position, value below, value above].
const JET_POINTS = {
  red: [[0, 0, 0], [0.35, 0, 0], [0.66, 1, 1], [0.89, 1, 1], [1, 0.5, 0.5]],
  green: [[0, 0, 0], [0.125, 0, 0], [0.375, 1, 1], [0.64, 1, 1], [0.91, 0, 0], [1, 0, 0]],
  blue: [[0, 0.5, 0.5], [0.11, 1, 1], [0.34, 1, 1], [0.65, 0, 0], [1, 0, 0]],
};

/** A 256-entry table for one colour channel, built as Matplotlib builds it. */
function channelTable(points, entries = 256) {
  const table = new Float64Array(entries);
  for (let i = 0; i < entries; i++) {
    const x = i / (entries - 1);
    let j = 1;
    while (j < points.length - 1 && points[j][0] < x) j++;
    const [x0, , y0] = points[j - 1];
    const [x1, y1] = points[j];
    table[i] = y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
  }
  table[0] = points[0][2];
  table[entries - 1] = points[points.length - 1][1];
  return table;
}

export const JET_TABLE = ["red", "green", "blue"].map((channel) => channelTable(JET_POINTS[channel]));

/** The jet colour (red, green, blue in [0, 1]) for a heatmap value in [0, 1]. */
export function jet(value) {
  const i = Math.min(255, Math.max(0, Math.trunc(f32(value * 256))));
  return [JET_TABLE[0][i], JET_TABLE[1][i], JET_TABLE[2][i]];
}

/** The tumour's border: tumour pixels within `rounds` pixels of non-tumour (repeated erosion). */
export function outline(mask, width, height, rounds = OUTLINE_WIDTH) {
  const inside = Uint8Array.from(mask, (v) => (v > 0 ? 1 : 0));
  let core = inside;
  for (let r = 0; r < rounds; r++) {
    const next = new Uint8Array(core.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        // A pixel stays only if it and its four neighbours are all tumour (off the image counts as not).
        next[i] = core[i] && y > 0 && core[i - width] && y < height - 1 && core[i + width]
          && x > 0 && core[i - 1] && x < width - 1 && core[i + 1] ? 1 : 0;
      }
    }
    core = next;
  }
  return inside.map((v, i) => (v && !core[i] ? 1 : 0));
}

/** app/demo.py's overlay: 55% grey scan + 45% jet heatmap, lime outline on top. RGBA, ready for a canvas. */
export function overlay(pixels, heatmap, mask = null) {
  const rgba = new Uint8ClampedArray(pixels.length * 4);
  const border = mask ? outline(mask, SIZE, SIZE) : null;
  const scanShare = f32(1 - HEAT_ALPHA); // NumPy does this part in 32-bit floats
  for (let i = 0; i < pixels.length; i++) {
    const grey = f32(scanShare * f32(pixels[i] / 255));
    const colour = border && border[i]
      ? OUTLINE_COLOUR
      : jet(heatmap[i]).map((h) => roundHalfEven((grey + HEAT_ALPHA * h) * 255));
    rgba.set([...colour, 255], 4 * i);
  }
  return rgba;
}

/** The text above the confidence bars (app/demo.py's describe). */
export function describe(predicted, confidence, example) {
  const headline = `Prediction: ${predicted} (${(confidence * 100).toFixed(1)}% confidence)`;
  if (!example) {
    return { headline, lines: ["Uploaded image: no tumour outline is available, so only the heatmap is shown."] };
  }
  const verdict = predicted === example.true ? "correct" : "a mistake";
  return {
    headline,
    lines: [
      `Example from test patient ${example.patient_id}. True type: ${example.true}, so this answer is ${verdict}. `
        + "The green line is the clinicians' tumour outline.",
      example.note,
    ],
  };
}
```

- [ ] **Step 5: Run the tests**

```powershell
node --test space/tests/pipeline.test.mjs
.\.venv\Scripts\python.exe -m pytest tests/test_space.py -v
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: Node alone reports 6 passed and 4 skipped (no fixtures); the pytest wrapper passes; the full suite gives `85 passed`.

- [ ] **Step 6: Commit**

```powershell
git add space/pipeline.js space/tests/pipeline.test.mjs tests/test_space.py
git commit -m "feat: browser image steps, tested against the Python code" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The web page

**Concept:** `index.html` is the page's skeleton, `style.css` its look, and `app.js` the wiring: it decodes an image to pixels (with no colour conversion), runs the model through ONNX Runtime Web and draws the results. All the arithmetic stays in the tested `pipeline.js`.

**Files:**
- Create: `space/index.html`, `space/style.css`, `space/app.js`

- [ ] **Step 1: Create `space/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Brain tumour MRI classifier</title>
  <link rel="stylesheet" href="style.css">
  <script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.js"></script>
  <script type="module" src="app.js"></script>
</head>
<body>
  <header>
    <h1>Brain tumour MRI classifier</h1>
    <p><strong>Educational project — not a medical device, not for diagnosis.</strong></p>
    <p>This model only knows three tumour types: glioma, meningioma and pituitary. Given a healthy scan or a
      non-MRI image, it will still pick one of the three.</p>
    <p>The model runs in your browser: images you upload never leave your device.</p>
  </header>

  <main class="columns">
    <section>
      <label for="upload">MRI slice (PNG or JPG; DICOM files are not supported)</label>
      <input type="file" id="upload" accept="image/png,image/jpeg" disabled>
      <h2>Example test scans</h2>
      <div id="examples" class="examples"></div>
    </section>

    <section>
      <p id="status">Loading the model (45 MB)…</p>
      <div id="summary"></div>
      <div id="bars"></div>
      <canvas id="heatmap" width="224" height="224" hidden></canvas>
      <p id="heatmap-caption" class="caption" hidden>Grad-CAM: where the evidence came from (red = most)</p>
    </section>
  </main>

  <footer>
    <p>ResNet-18 fine-tuned on 2D contrast-enhanced T1-weighted MRI slices: 90.9% accuracy on 430 held-out
      test scans from 34 patients. How it was built, and where it fails:
      <a href="https://github.com/RVotta4/brain-tumour-mri">github.com/RVotta4/brain-tumour-mri</a>.</p>
    <p>Example scans: Cheng, Jun (2017). <em>brain tumor dataset</em>. figshare.
      <a href="https://doi.org/10.6084/m9.figshare.1512427">doi:10.6084/m9.figshare.1512427</a> (CC BY 4.0).</p>
    <p>Runs on <a href="https://onnxruntime.ai/">ONNX Runtime Web</a>.</p>
  </footer>
</body>
</html>
```

- [ ] **Step 2: Create `space/style.css`**

```css
:root {
  --text: #1f2328;
  --muted: #57606a;
  --line: #d0d7de;
  --panel: #f6f8fa;
  --accent: #0969da;
  --background: #ffffff;
}

@media (prefers-color-scheme: dark) {
  :root {
    --text: #e6edf3;
    --muted: #9da7b3;
    --line: #30363d;
    --panel: #161b22;
    --accent: #4493f8;
    --background: #0d1117;
  }
}

* { box-sizing: border-box; }

body {
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 16px;
  font: 16px/1.5 system-ui, sans-serif;
  color: var(--text);
  background: var(--background);
}

a { color: var(--accent); }

header { border-bottom: 1px solid var(--line); margin-bottom: 24px; }
header h1 { margin-top: 0; }

.columns { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; }
@media (max-width: 760px) { .columns { grid-template-columns: 1fr; } }

label { display: block; margin-bottom: 8px; font-weight: 600; }
h2 { font-size: 1rem; margin: 24px 0 8px; }

.examples { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }

.example {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 6px;
  border: 1px solid var(--line);
  border-radius: 8px;
  background: var(--panel);
  color: var(--text);
  font: inherit;
  font-size: 0.8rem;
  text-align: left;
  cursor: pointer;
}
.example:hover, .example:focus-visible { border-color: var(--accent); }
.example img { width: 100%; border-radius: 4px; }

#status { color: var(--muted); min-height: 1.5em; }

.bar { display: grid; grid-template-columns: 7rem 1fr 3rem; align-items: center; gap: 8px; margin: 4px 0; }
.track { height: 12px; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.fill { display: block; height: 100%; background: var(--accent); }

#heatmap { display: block; width: 100%; max-width: 448px; margin-top: 16px; border-radius: 8px; }
#heatmap[hidden] { display: none; }
.caption { color: var(--muted); font-size: 0.85rem; }

footer { margin-top: 40px; border-top: 1px solid var(--line); color: var(--muted); font-size: 0.85rem; }
```

- [ ] **Step 3: Create `space/app.js`**

```javascript
// The page's wiring: read images, run the model, show the results.
// All the image arithmetic lives in pipeline.js, which is tested against the Python code.

import {
  CLASS_NAMES, SIZE, describe, findExample, normaliseHeatmap, overlay, preprocess, softmax, toGrey,
} from "./pipeline.js";

// ort (ONNX Runtime Web) is loaded by index.html. Its WebAssembly files come from the same CDN.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1; // a static Space can't send the headers that multi-threading needs

const page = {
  status: document.getElementById("status"),
  upload: document.getElementById("upload"),
  examples: document.getElementById("examples"),
  summary: document.getElementById("summary"),
  bars: document.getElementById("bars"),
  heatmap: document.getElementById("heatmap"),
  heatmapCaption: document.getElementById("heatmap-caption"),
};

/** Decode an image file to grey levels, exactly as stored: no colour-profile conversion. */
async function readImage(blob) {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  return { grey: toGrey(data, bitmap.width, bitmap.height), width: bitmap.width, height: bitmap.height };
}

async function fetchBlob(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url} (${response.status})`);
  return response.blob();
}

/** The six example scans, with their outlines, from examples.json. */
async function loadExamples() {
  const list = await (await fetch("examples.json")).json();
  return Promise.all(list.map(async (example) => {
    const url = `examples/${example.name}.png`;
    const scan = await readImage(await fetchBlob(url));
    const mask = await readImage(await fetchBlob(`examples/${example.name}_mask.png`));
    return { ...example, url, scan: scan.grey, mask: mask.grey.map((v) => (v > 127 ? 1 : 0)) };
  }));
}

function showExamples(examples, onPick) {
  for (const example of examples) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example";
    const image = document.createElement("img");
    image.src = example.url;
    image.alt = example.caption;
    const caption = document.createElement("span");
    caption.textContent = example.caption;
    button.append(image, caption);
    button.addEventListener("click", () => onPick(example));
    page.examples.append(button);
  }
}

function paragraph(text) {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}

function bar(name, probability) {
  const row = document.createElement("div");
  row.className = "bar";
  const track = document.createElement("span");
  track.className = "track";
  const fill = document.createElement("span");
  fill.className = "fill";
  fill.style.width = `${(probability * 100).toFixed(1)}%`;
  track.append(fill);
  const label = document.createElement("span");
  label.textContent = name;
  const value = document.createElement("span");
  value.textContent = `${Math.round(probability * 100)}%`;
  row.append(label, track, value);
  return row;
}

function showResult(probabilities, text, picture) {
  const headline = paragraph("");
  const strong = document.createElement("strong");
  strong.textContent = text.headline;
  headline.append(strong);
  page.summary.replaceChildren(headline, ...text.lines.map(paragraph));

  const ranked = CLASS_NAMES.map((name, i) => [name, probabilities[i]]).sort((a, b) => b[1] - a[1]);
  page.bars.replaceChildren(...ranked.map(([name, probability]) => bar(name, probability)));

  page.heatmap.getContext("2d").putImageData(new ImageData(picture, SIZE, SIZE), 0, 0);
  page.heatmap.hidden = false;
  page.heatmapCaption.hidden = false;
}

async function main() {
  const [session, examples] = await Promise.all([ort.InferenceSession.create("model.onnx"), loadExamples()]);

  async function analyse(blob) {
    page.status.textContent = "Running the model…";
    try {
      const { grey, width, height } = await readImage(blob);
      const example = findExample(grey, width, height, examples);
      const pixels = example ? example.scan : preprocess(grey, width, height);
      const scan = new ort.Tensor("float32", Float32Array.from(pixels, (p) => p / 255), [1, 1, SIZE, SIZE]);
      const { scores, heatmaps } = await session.run({ scan });
      const probabilities = softmax(Array.from(scores.data));
      const predicted = probabilities.indexOf(Math.max(...probabilities));
      const heatmap = normaliseHeatmap(heatmaps.data, predicted);
      showResult(probabilities, describe(CLASS_NAMES[predicted], probabilities[predicted], example),
        overlay(pixels, heatmap, example ? example.mask : null));
      page.status.textContent = "";
    } catch (error) {
      page.status.textContent = `Something went wrong: ${error.message}`;
    }
  }

  showExamples(examples, async (example) => analyse(await fetchBlob(example.url)));
  page.upload.addEventListener("change", () => {
    if (page.upload.files[0]) analyse(page.upload.files[0]);
  });
  page.upload.disabled = false;
  page.status.textContent = "Ready. Upload a slice or click an example.";
}

main().catch((error) => {
  page.status.textContent = `Could not start the demo: ${error.message}`;
});
```

- [ ] **Step 4: Check the JavaScript parses and the suite passes**

```powershell
node --check space/app.js
node --check space/pipeline.js
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: no output from either `node --check`, then `85 passed`. (The page is tried in a browser in Task 6.)

- [ ] **Step 5: Commit**

```powershell
git add space/index.html space/style.css space/app.js
git commit -m "feat: static web page for the browser demo" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Deploy to a static Space (and preview locally)

**Concept:** a static Space only hands files to the browser, so it is free. `app/deploy.py` gathers exactly the files the page needs. `--preview` writes them into a local folder, so the site can be tried on the laptop exactly as it will be served; `--space` uploads them.

**Files:**
- Create: `space/README.md`
- Rewrite: `app/deploy.py`, `tests/test_deploy.py`
- Delete: `app/README.md`, `app/requirements.txt`

- [ ] **Step 1: Create `space/README.md`**

```markdown
---
title: Brain Tumour MRI Classifier
emoji: 🧠
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
short_description: Brain tumour MRI classifier that runs in your browser
---

# Brain tumour MRI classifier

**Educational project, not a medical device, not for diagnosis.**

Upload a 2D brain MRI slice (PNG or JPG), or click one of six test-set examples. The page shows the model's
predicted tumour type (glioma, meningioma or pituitary), its confidence for all three, and a Grad-CAM
heatmap of where the evidence for its answer came from. For the examples, the clinicians' tumour outline
is drawn in green.

The model runs entirely in your browser with ONNX Runtime Web: images you upload never leave your device.

The model is a ResNet-18 fine-tuned on the Cheng et al. dataset, scoring 90.9% on 430 held-out test scans
from 34 patients. The full report, including where it fails, is at
[github.com/RVotta4/brain-tumour-mri](https://github.com/RVotta4/brain-tumour-mri).

Example scans: Cheng, Jun (2017). *brain tumor dataset*. figshare.
[doi:10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427) (CC BY 4.0).
```

- [ ] **Step 2: Write the failing tests**

Replace the contents of `tests/test_deploy.py` with:

```python
import csv
import json

import pytest

from app.deploy import MODEL, PAGE, preview, site_files


def make_fake_repo(root):
    for name in PAGE + ["README.md"]:
        (root / "space").mkdir(exist_ok=True)
        (root / "space" / name).write_text("x")
    (root / MODEL).parent.mkdir(parents=True)
    (root / MODEL).write_text("x")
    examples = root / "app" / "examples"
    examples.mkdir(parents=True)
    for name in ["a.png", "a_mask.png"]:
        (examples / name).write_text("x")
    with open(examples / "examples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "index", "patient_id", "true", "caption", "note"])
        writer.writeheader()
        writer.writerow({"name": "a", "index": 7, "patient_id": "P1", "true": "glioma",
                         "caption": "Glioma — correct", "note": "Hi, there."})


def test_site_files_lists_the_page_model_and_examples(tmp_path):
    make_fake_repo(tmp_path)

    files = site_files(tmp_path)

    assert sorted(files) == sorted(PAGE + ["README.md", "model.onnx", "examples/a.png", "examples/a_mask.png",
                                           "examples.json"])
    assert files["model.onnx"] == tmp_path / MODEL


def test_examples_json_carries_what_the_page_shows(tmp_path):
    make_fake_repo(tmp_path)

    examples = json.loads(site_files(tmp_path)["examples.json"])

    assert examples == [{"name": "a", "patient_id": "P1", "true": "glioma", "caption": "Glioma — correct",
                         "note": "Hi, there."}]


def test_site_files_refuses_when_something_is_missing(tmp_path):
    make_fake_repo(tmp_path)
    (tmp_path / MODEL).unlink()

    with pytest.raises(FileNotFoundError, match="model.onnx"):
        site_files(tmp_path)


def test_preview_writes_the_site_to_a_folder(tmp_path):
    make_fake_repo(tmp_path)

    preview(tmp_path / "out", root=tmp_path)

    assert (tmp_path / "out" / "index.html").read_text() == "x"
    assert (tmp_path / "out" / "examples" / "a_mask.png").exists()
    assert json.loads((tmp_path / "out" / "examples.json").read_text(encoding="utf-8"))[0]["name"] == "a"
```

- [ ] **Step 3: Run to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_deploy.py -v
```
Expected: collection error, `ImportError: cannot import name 'MODEL' from 'app.deploy'`.

- [ ] **Step 4: Rewrite `app/deploy.py`**

```python
"""Put the live demo on a free static Hugging Face Space, or preview it locally.

Usage:
    .\\.venv\\Scripts\\python.exe -m app.deploy --preview space_preview
    .\\.venv\\Scripts\\python.exe -m app.deploy --space <username>/brain-tumour-mri   (after `hf auth login`)

A static Space only hands files to the visitor's browser, which runs the model
itself. The model goes to Hugging Face only, never to GitHub. Running --space
again uploads the current files as a new commit on the Space.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ["index.html", "style.css", "pipeline.js", "app.js"]
MODEL = "experiments/03-resnet18-finetuned/model.onnx"
EXAMPLE_FIELDS = ["name", "patient_id", "true", "caption", "note"]  # what the page shows about each example


def examples_json(examples_dir):
    """The example list the page reads, made from app/examples/examples.csv so the text lives in one place."""
    with open(examples_dir / "examples.csv", newline="", encoding="utf-8") as f:
        rows = [{field: row[field] for field in EXAMPLE_FIELDS} for row in csv.DictReader(f)]
    return json.dumps(rows, indent=2, ensure_ascii=False).encode("utf-8")


def site_files(root=ROOT):
    """Everything the Space serves: its path there -> a local file, or the bytes to write."""
    root = Path(root)
    files = {name: root / "space" / name for name in PAGE}
    files["README.md"] = root / "space" / "README.md"
    files["model.onnx"] = root / MODEL
    examples_dir = root / "app" / "examples"
    for path in sorted(examples_dir.glob("*.png")):
        files[f"examples/{path.name}"] = path
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing: " + ", ".join(missing))
    files["examples.json"] = examples_json(examples_dir)
    return files


def preview(folder, root=ROOT):
    """Write the site into a local folder, exactly as it will be uploaded."""
    folder = Path(folder)
    for remote, source in site_files(root).items():
        target = folder / remote
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source if isinstance(source, bytes) else source.read_bytes())
    print(f"Wrote the site to {folder}. To view it, run\n"
          f"    .\\.venv\\Scripts\\python.exe -m http.server 8000 --directory {folder}\n"
          "then open http://localhost:8000 (Ctrl+C stops the server).")


def deploy(space):
    """Create the static Space if needed (public, free), then upload everything in one commit."""
    from huggingface_hub import CommitOperationAdd, HfApi  # comes with Gradio; not needed by the tests

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=remote,
                                     path_or_fileobj=source if isinstance(source, bytes) else str(source))
                  for remote, source in site_files().items()]
    api.create_commit(repo_id=space, repo_type="space", operations=operations,
                      commit_message="Deploy the demo from github.com/RVotta4/brain-tumour-mri")
    print(f"Uploaded {len(operations)} files. It should be live within a minute: "
          f"https://huggingface.co/spaces/{space}")


def main():
    parser = argparse.ArgumentParser(description="Put the live demo on a static Hugging Face Space.")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--space", help="<username>/<space-name> on Hugging Face")
    where.add_argument("--preview", help="a local folder to write the site into")
    args = parser.parse_args()
    if args.preview:
        preview(args.preview)
    else:
        deploy(args.space)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Delete the Gradio Space files**

```powershell
git rm app/README.md app/requirements.txt
```

- [ ] **Step 6: Run the tests and check the real file list**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -c "from app.deploy import site_files; f = site_files(); print(len(f)); print(sorted(f))"
```
Expected: `86 passed` (85 − 3 old deploy tests + 4 new); then `19` and the list: `README.md`, `app.js`, `examples.json`, 12 `examples/*.png`, `index.html`, `model.onnx`, `pipeline.js`, `style.css`. (The second command needs Task 2's `model.onnx`.)

- [ ] **Step 7: Commit**

```powershell
git add space/README.md app/deploy.py tests/test_deploy.py
git commit -m "feat: deploy the demo as a free static Space, with a local preview" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6 (Robbi): preview on the laptop — then PAUSE 2

- [ ] **Step 1 (Robbi): build the preview and serve it**

```powershell
.\.venv\Scripts\python.exe -m app.deploy --preview space_preview
.\.venv\Scripts\python.exe -m http.server 8000 --directory space_preview
```
The second command keeps running; it is a tiny local web server (needed because browsers won't load the model from a plain file).

- [ ] **Step 2 (Robbi): try it** at `http://localhost:8000`:
1. The status goes from "Loading the model (45 MB)…" to "Ready…".
2. Click each example. The prediction and confidence match Task 2's MATCH list; the green outline appears.
3. Upload a non-MRI image. It still predicts a type, with no outline.
4. Narrow the window (or open it on a phone on the same Wi-Fi): the columns stack.

The controller can also open `http://localhost:8000` in the built-in browser and read the page text to confirm step 2.

- [ ] **Step 3: PAUSE 2** — agree any changes; Ctrl+C stops the server.

---

### Task 7 (Robbi): deploy — then PAUSE 3

- [ ] **Step 1 (Robbi): deploy** (already logged in with `hf auth login`)

```powershell
.\.venv\Scripts\python.exe -m app.deploy --space RVotta4/brain-tumour-mri
```
Expected: `Uploaded 19 files. It should be live within a minute: https://huggingface.co/spaces/RVotta4/brain-tumour-mri`.

- [ ] **Step 2 (Robbi): live checks** — repeat Task 6 Step 2 on the Space, including on a phone.

**If the page says it could not load `model.onnx`** (the Space didn't serve the large file): create a free model repository and load the model from there:
```powershell
.\.venv\Scripts\hf.exe repo create RVotta4/brain-tumour-mri-model --repo-type model
.\.venv\Scripts\hf.exe upload RVotta4/brain-tumour-mri-model experiments/03-resnet18-finetuned/model.onnx model.onnx
```
then change `ort.InferenceSession.create("model.onnx")` in `space/app.js` to
`ort.InferenceSession.create("https://huggingface.co/RVotta4/brain-tumour-mri-model/resolve/main/model.onnx")`, commit, and re-run Step 1.

- [ ] **Step 3: PAUSE 3** — record the live URL for the write-up.

---

### Task 8: Write-up

**Files:** `README.md`, `docs/learning-notes.md`, `docs/demo.png` (screenshot)

- [ ] **Step 1 (Robbi): screenshot** — on the live Space, click "Meningioma — called glioma", take a screenshot of the page (Win+Shift+S) and save it as `docs/demo.png`.

- [ ] **Step 2: README** (`README.md`)
1. Status line becomes: `> **Status:** in progress. Experiments 1–4, the one-shot test score, the Grad-CAM analysis and the live demo are complete; the "99% trap" experiment is next.`
2. Under the `**Result:**` line add:
   ```markdown
   **Try it:** [live demo on Hugging Face](https://huggingface.co/spaces/RVotta4/brain-tumour-mri). Upload a slice or click an example to see the prediction, the confidence for each type and a Grad-CAM heatmap. The model runs in your browser, so uploaded images never leave your device.

   ![The live demo](docs/demo.png)
   ```
3. In **How to run**, after the `src.explain` line add:
   ```
       .\.venv\Scripts\python.exe -m app.make_examples
       .\.venv\Scripts\python.exe -m app.export_onnx
       .\.venv\Scripts\python.exe -m app.deploy --preview space_preview
       .\.venv\Scripts\python.exe -m http.server 8000 --directory space_preview
   ```
   and after the ImageNet-weights sentence add: `The browser demo then runs at http://localhost:8000. Its image steps are tested against the Python code, which needs Node.js for that one test.`

- [ ] **Step 3: Learning notes 20–23** — append to `docs/learning-notes.md`:

```markdown
## 20. Where the demo runs: a static Space

The first plan was a Gradio app on a Hugging Face Space: a Python server that runs the model for each visitor. The day before deploying, Hugging Face started charging for those. Free accounts can still host **static** Spaces, which only hand files to the browser.

So the demo became a web page that runs the model itself, in the visitor's browser, using ONNX Runtime Web. The model is exported from PyTorch to ONNX, a portable format. There is no server to pay for, sleep or wake, and an uploaded scan never leaves the visitor's device.

> **Say:** "The demo runs the model in your browser, so it's free to host and the scan you upload never leaves your computer. I moved to that when Hugging Face started charging for Python Spaces."

## 21. Grad-CAM without a backward pass

A browser runtime only runs the model forwards, and Grad-CAM needs gradients. But for ResNet-18 the gradients are known in advance. The last step averages each of the 512 maps and weights the averages to get each class score, so the gradient for map *k* is that map's weight divided by 49 (the 7×7 cells). Weighting the maps by the last-layer weights therefore gives exactly Grad-CAM's heatmap, once it is scaled so its peak is 1. This version is known as CAM. The exported model computes it in the same forward pass, and it matched the hand-written Grad-CAM to within 0.00001.

> **Say:** "For a network that ends in global average pooling, Grad-CAM reduces to CAM: the gradients are just the last layer's weights. So I exported a model that returns the heatmaps directly and checked it against my Grad-CAM."

## 22. Making the browser give the same answers as Python

A demo is only honest if it runs the model exactly as it was tested. Two things could quietly change the pixels the model sees:
- **the example scans:** they were stretched before being resized, so stretching them again would change them by up to 39 grey levels. The demo recognises an example by its pixels and uses the stored scan unchanged;
- **the browser code:** resizing an image "the same way" isn't enough. The JavaScript copies the exact arithmetic of Pillow's bilinear resize, NumPy's rounding and Matplotlib's colour map.

Both are tested. The six examples reproduce the test run's confidences, and the JavaScript matches the Python pixel for pixel on five image sizes. The Python code writes its answers to a file, and pytest runs Node to compare.

> **Say:** "I tested the browser version against the Python one pixel for pixel, and checked that every example reproduces its test-set confidence, so the demo is the model I evaluated, not an approximation of it."

## 23. What happens to an uploaded image

Nothing leaves the device. The page reads the file in the browser, runs the model there, and draws the result. The only downloads are the page, the model and the example scans. Two small, stated differences affect uploads only: the browser reduces 16-bit images to 8-bit, and its JPEG decoder can differ from Python's by a grey level here and there. The example scans (PNG) are unaffected.

> **Say:** "Uploaded scans never leave the visitor's device, because the model runs in the browser. For a medical demo that's the privacy design I'd want anyway."
```

- [ ] **Step 4: Run the whole suite and commit**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
git add README.md docs/learning-notes.md docs/demo.png
git commit -m "docs: stage 5 live demo link, screenshot and notes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
Expected: `86 passed`.

---

### Task 9: Final review, merge and push

- [ ] **Step 1: Final code review** of the whole branch against both Stage 5 specs (superpowers:requesting-code-review).
- [ ] **Step 2: Robbi approves the README and notes wording.**
- [ ] **Step 3: Merge and push**

```powershell
git checkout main
git merge --no-ff feat/stage-5-demo -m "Merge stage 5: live demo in the browser on a static Hugging Face Space" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
.\.venv\Scripts\python.exe -m pytest -q
git push origin main
```
Expected: `86 passed`, push succeeds.
