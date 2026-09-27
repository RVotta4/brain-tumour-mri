# Stage 4: Where the Model Looks (Grad-CAM) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Explain every test prediction of the final model with hand-written Grad-CAM, measure whether it looks at the tumour (pointing game and in-mask share against luck baselines), and show every mistake — without changing the model.

**Architecture:** A new module `src/explain.py` holds pure measurement functions (NumPy/PyTorch on arrays), `grad_cam` (an explicit two-half forward through `ResNet18Grey` plus `torch.autograd.grad`), per-patient and summary tables, and `run_explanation`, which reloads the scored final model and writes everything to `experiments/<name>/explain/`. Two drawing functions join the existing ones in `src/plots.py`.

**Tech Stack:** Python 3.13 (Windows, `py` launcher), PyTorch 2.14 CPU, torchvision 0.29, NumPy, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-27-stage-4-gradcam-design.md`. Overall design: `docs/superpowers/specs/2026-09-17-brain-tumour-mri-design.md`.

---

## Ground rules for whoever executes this

- **Platform:** Windows, PowerShell. All commands run from `C:\Users\Robbi\brain-tumour-mri`.
- **Always call the venv's Python directly:** `.\.venv\Scripts\python.exe`. Bare `python` on this machine opens the Microsoft Store.
- **The model is frozen and the test set already scored.** Nothing in this stage trains, saves model weights, or re-scores. Never run `python -m src.score` (except `--help`) and never delete or modify anything in `experiments/03-resnet18-finetuned/` other than creating its `explain/` folder in Task 7.
- **Code tasks (1–6) run straight through without check-ins.** Task 8 is a PAUSE for Robbi.
- **Commit messages** end with the trailer `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (passed as a second `-m`).
- **Verified facts as of 2026-09-27:** 51 tests passing; `main` at `6bb79f1` (spec commit, unpushed). `data/cheng_224.npz` holds `masks` (3064, 224, 224) uint8 in {0, 1}, none empty. Test split: 430 scans — meningioma 80 (12 patients), glioma 222 (13), pituitary 128 (9); median tumour area 0.7–1.6% of the image. Final model: `experiments/03-resnet18-finetuned/` with `model.pt`, `config.json` (`"model": "resnet18"`), `metrics.json`, and `test_metrics.json` (accuracy 0.9093).

## File map

| File | Change |
|---|---|
| `src/explain.py` | New: measurements, `grad_cam`, tables, `run_explanation`, command line |
| `src/plots.py` | Add `draw_outline`, `plot_gradcam_examples`, `plot_mistakes` |
| `tests/test_explain.py` | New |
| `tests/test_plots.py` | Two figure tests |
| `README.md`, `docs/learning-notes.md` | Task 9 |

---

### Task 1: Branch

- [ ] **Step 1**

```powershell
git checkout -b feat/stage-4-gradcam
```
Expected: `Switched to a new branch 'feat/stage-4-gradcam'`.

---

### Task 2: Measurements

**Concept:** a measurement needs a luck baseline. Saying "the heatmap's peak landed on the tumour in 60% of scans" means nothing until you know a random peak would land there only ~4% of the time. Every score in this stage is reported next to what luck alone would give.

**Files:**
- Create: `src/explain.py`, `tests/test_explain.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_explain.py`:

```python
import numpy as np
import pytest

from src.explain import mask_share, pointing_chance, pointing_hit


def square_mask(size=64, top=20, left=20, width=10):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[top:top + width, left:left + width] = 1
    return mask


def heatmap_peaking_at(row, col, size=64):
    heatmap = np.zeros((size, size), dtype=np.float32)
    heatmap[row, col] = 1.0
    return heatmap


def test_pointing_hit_inside_near_and_far():
    mask = square_mask()  # rows and columns 20-29

    assert pointing_hit(heatmap_peaking_at(25, 25), mask)
    assert pointing_hit(heatmap_peaking_at(25, 34), mask)  # 5 pixels right of the tumour
    assert not pointing_hit(heatmap_peaking_at(25, 49), mask)  # 20 pixels right


def test_pointing_chance_is_the_share_of_pixels_near_the_tumour():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32, 32] = 1

    assert pointing_chance(mask, tolerance=2) == pytest.approx(25 / 4096)  # a 5x5 window


def test_mask_share_is_the_fraction_of_heatmap_inside_the_tumour():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :2] = 1

    assert mask_share(np.ones((4, 4), dtype=np.float32), mask) == pytest.approx(0.25)
    assert mask_share(np.zeros((4, 4), dtype=np.float32), mask) == 0.0
```

- [ ] **Step 2: Run them to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_explain.py -v
```
Expected: `ModuleNotFoundError: No module named 'src.explain'`.

- [ ] **Step 3: Create `src/explain.py`**

```python
"""Where the final model looks: Grad-CAM heatmaps measured against the tumour outlines.

Usage:
    .\\.venv\\Scripts\\python.exe -m src.explain --name 03-resnet18-finetuned

Runs on the test scans, and only once the test score exists. It never
changes the model: whatever it finds is reported, not fixed.
"""

import numpy as np
import torch
from torch.nn import functional as F

TOLERANCE = 8  # pixels of leeway in the pointing game: a quarter of one 32-pixel heatmap cell


def near_tumour(mask, tolerance=TOLERANCE):
    """Every pixel within `tolerance` pixels of the tumour (a square window).

    Max-pooling with a (2 * tolerance + 1)-wide window marks a pixel if any
    tumour pixel falls inside the window around it: the outline grown outwards.
    """
    grown = F.max_pool2d(torch.from_numpy(mask.astype(np.float32))[None, None],
                         kernel_size=2 * tolerance + 1, stride=1, padding=tolerance)
    return grown[0, 0].numpy() > 0


def pointing_hit(heatmap, mask, tolerance=TOLERANCE):
    """Pointing game: is the heatmap's hottest pixel on, or within tolerance of, the tumour?"""
    peak = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return bool(near_tumour(mask, tolerance)[peak])


def pointing_chance(mask, tolerance=TOLERANCE):
    """How often a randomly placed peak would score a hit: the pointing game's luck baseline."""
    return float(near_tumour(mask, tolerance).mean())


def mask_share(heatmap, mask):
    """Fraction of the heatmap's total that falls inside the tumour outline.

    A heatmap spread evenly over the scan would score the tumour's area
    fraction, so that area is the luck baseline to compare against.
    """
    total = heatmap.sum()
    if total == 0:
        return 0.0
    return float(heatmap[mask > 0].sum() / total)
```

- [ ] **Step 4: Run the tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_explain.py -v
```
Expected: 3 passed. Then `.\.venv\Scripts\python.exe -m pytest -q` — 54 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/explain.py tests/test_explain.py
git commit -m "feat: pointing game and in-mask share with luck baselines" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Grad-CAM

**Concept:** Grad-CAM. ResNet-18's last stage turns a scan into 512 pattern maps, each a 7×7 grid of how strongly one learned pattern appears where. The decision averages each map to one number and weighs them into three class scores. Grad-CAM asks PyTorch how much the winning score would change if each map got stronger (its gradient), weights the maps by that, and keeps the positive total: a 7×7 picture of where the evidence for the decision came from, stretched to the scan's size.

**Files:**
- Modify: `src/explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_explain.py`, replace the import block at the top with:

```python
import numpy as np
import pytest
import torch

from src.explain import class_scores, feature_maps, grad_cam, mask_share, pointing_chance, pointing_hit
from src.models import ResNet18Grey, SmallCNN
```

Append:

```python
def test_grad_cam_gives_a_224_heatmap_in_unit_range():
    torch.manual_seed(0)
    model = ResNet18Grey(weights=None).eval()

    heatmap, predicted, confidence = grad_cam(model, torch.rand(1, 224, 224))

    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0
    assert predicted in (0, 1, 2)
    assert 0.0 <= confidence <= 1.0


def test_two_half_forward_matches_the_model():
    torch.manual_seed(0)
    model = ResNet18Grey(weights=None).eval()
    images = torch.rand(2, 1, 224, 224)

    with torch.no_grad():
        assert torch.allclose(class_scores(model, feature_maps(model, images)), model(images), atol=1e-5)


def test_grad_cam_only_explains_the_resnet():
    with pytest.raises(TypeError, match="ResNet18Grey"):
        grad_cam(SmallCNN(), torch.rand(1, 224, 224))
```

- [ ] **Step 2: Run them to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_explain.py -v
```
Expected: `ImportError: cannot import name 'class_scores' from 'src.explain'`.

- [ ] **Step 3: Implement**

In `src/explain.py`, add below `from torch.nn import functional as F`:

```python

from src.models import ResNet18Grey
```

Append to `src/explain.py`:

```python
def feature_maps(model, images):
    """First half of ResNet-18: scans in, 512 pattern maps of 7x7 out."""
    backbone = model.backbone
    x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(model.prepare(images)))))
    return backbone.layer4(backbone.layer3(backbone.layer2(backbone.layer1(x))))


def class_scores(model, maps):
    """Second half: average each pattern map to one number, then score each tumour type."""
    return model.backbone.fc(torch.flatten(model.backbone.avgpool(maps), 1))


def grad_cam(model, image):
    """Heatmap of where the evidence for the model's decision came from.

    image: one scan, shape (1, height, width), values in [0, 1].
    Returns (heatmap, predicted class, confidence). The heatmap has the
    scan's height and width, scaled so its hottest point is 1.
    """
    if not isinstance(model, ResNet18Grey):
        raise TypeError(f"grad_cam explains the final ResNet18Grey model, not {type(model).__name__}")
    model.eval()
    maps = feature_maps(model, image.unsqueeze(0))
    scores = class_scores(model, maps)[0]
    predicted = int(scores.argmax())
    confidence = float(torch.softmax(scores, dim=0)[predicted])

    # How much would the winning score rise if each map got stronger?
    (gradients,) = torch.autograd.grad(scores[predicted], maps)
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # one importance per map
    cam = torch.relu((weights * maps).sum(dim=1, keepdim=True))  # keep evidence for, drop evidence against
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    heatmap = cam.detach().numpy()
    peak = heatmap.max()
    return (heatmap / peak if peak > 0 else heatmap), predicted, confidence
```

- [ ] **Step 4: Run the tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_explain.py -v
```
Expected: 6 passed. Then `.\.venv\Scripts\python.exe -m pytest -q` — 57 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/explain.py tests/test_explain.py
git commit -m "feat: hand-written Grad-CAM for the final ResNet-18" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Patient and summary tables

**Concept:** count by patient, not just by slice. Neighbouring slices of one patient look almost identical, so 19 missed slices could be 19 separate failures or one hard patient seen 19 times. Those are very different findings.

**Files:**
- Modify: `src/explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_explain.py`, change the `src.explain` import line to:

```python
from src.explain import (class_scores, feature_maps, grad_cam, mask_share, patient_errors, pointing_chance,
                         pointing_hit, summarise)
```

Append:

```python
def scan_row(patient, true, predicted, hit=True, share=0.2, area=0.01, chance=0.05, confidence=0.8):
    return {"patient_id": patient, "true": true, "predicted": predicted, "correct": true == predicted,
            "confidence": confidence, "pointing_hit": hit, "pointing_chance": chance,
            "mask_share": share, "mask_area": area}


def test_patient_errors_counts_and_sorts_by_mistakes():
    rows = [
        scan_row("A", "glioma", "glioma"), scan_row("A", "glioma", "meningioma"),
        scan_row("B", "meningioma", "glioma"), scan_row("B", "meningioma", "glioma"),
        scan_row("B", "meningioma", "pituitary"),
        scan_row("C", "pituitary", "pituitary"),
    ]

    table = patient_errors(rows)

    assert [row["patient_id"] for row in table] == ["B", "A", "C"]
    assert table[0] == {"patient_id": "B", "true": "meningioma", "slices": 3, "wrong": 3,
                        "most_common_wrong_prediction": "glioma"}
    assert table[2]["wrong"] == 0 and table[2]["most_common_wrong_prediction"] == ""


def test_summarise_compares_with_luck():
    rows = [
        scan_row("A", "glioma", "glioma", hit=True, share=0.2, area=0.01),
        scan_row("B", "meningioma", "glioma", hit=False, share=0.0, area=0.03, confidence=0.95),
    ]

    summary = summarise(rows)

    assert summary["overall"]["pointing_hit_rate"] == 0.5
    assert summary["overall"]["mask_share_times_luck"] == 5.0  # mean share 0.1 / mean area 0.02
    assert summary["correct"]["scans"] == 1 and summary["wrong"]["scans"] == 1
    assert summary["by_true_type"]["pituitary"] is None
    assert summary["confident_mistakes"] == 1
```

- [ ] **Step 2: Run them to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_explain.py -v
```
Expected: `ImportError: cannot import name 'patient_errors' from 'src.explain'`.

- [ ] **Step 3: Implement**

In `src/explain.py`, change the import block (everything between the module docstring and `TOLERANCE`) to:

```python
from collections import Counter

import numpy as np
import torch
from torch.nn import functional as F

from src.data import CLASS_NAMES
from src.models import ResNet18Grey
```

and add directly below the `TOLERANCE` line:

```python
CONFIDENT = 0.9  # a wrong answer at or above this confidence is a "confident mistake"
```

Append:

```python
def patient_errors(rows):
    """Mistakes counted per patient, most mistakes first.

    Neighbouring slices of one patient look alike, so many missed slices can
    be one hard patient rather than many separate failures.
    """
    by_patient = {}
    for row in rows:
        by_patient.setdefault(row["patient_id"], []).append(row)
    table = []
    for patient_id, scans in by_patient.items():
        wrong = [scan["predicted"] for scan in scans if not scan["correct"]]
        table.append({
            "patient_id": patient_id,
            "true": scans[0]["true"],
            "slices": len(scans),
            "wrong": len(wrong),
            "most_common_wrong_prediction": Counter(wrong).most_common(1)[0][0] if wrong else "",
        })
    return sorted(table, key=lambda row: (-row["wrong"], row["patient_id"]))


def group_stats(rows):
    """Pointing-game hit rate and in-mask share for a group of scans, beside their luck baselines."""
    if not rows:
        return None
    share = float(np.mean([row["mask_share"] for row in rows]))
    area = float(np.mean([row["mask_area"] for row in rows]))
    return {
        "scans": len(rows),
        "pointing_hit_rate": round(float(np.mean([row["pointing_hit"] for row in rows])), 4),
        "pointing_chance": round(float(np.mean([row["pointing_chance"] for row in rows])), 4),
        "mean_mask_share": round(share, 4),
        "mean_mask_area": round(area, 4),
        "mask_share_times_luck": round(share / area, 1) if area > 0 else None,
    }


def summarise(rows):
    """The headline numbers: overall, correct versus wrong, and per true tumour type."""
    return {
        "overall": group_stats(rows),
        "correct": group_stats([row for row in rows if row["correct"]]),
        "wrong": group_stats([row for row in rows if not row["correct"]]),
        "by_true_type": {name: group_stats([row for row in rows if row["true"] == name]) for name in CLASS_NAMES},
        "confident_mistakes": sum(1 for row in rows if not row["correct"] and row["confidence"] >= CONFIDENT),
    }
```

- [ ] **Step 4: Run the tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_explain.py -v
```
Expected: 8 passed. Then `.\.venv\Scripts\python.exe -m pytest -q` — 59 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/explain.py tests/test_explain.py
git commit -m "feat: per-patient mistakes and attention summary" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Heatmap figures

**Concept:** show the outline and the heatmap together. The heatmap alone looks convincing whatever it does; drawing the clinician's outline next to it is what lets a reader judge whether the model looked in the right place.

**Files:**
- Modify: `src/plots.py`
- Test: `tests/test_plots.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_plots.py`, replace the import line at the top with:

```python
import numpy as np

from src.plots import plot_augmentation, plot_confusion_matrix, plot_gradcam_examples, plot_history, plot_mistakes
```

Append:

```python
def tiny_panel(seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    return rng.random((16, 16)), mask, rng.random((16, 16))


def test_plot_gradcam_examples_writes_png(tmp_path):
    examples = {"glioma": [tiny_panel(0), tiny_panel(1)], "pituitary": [tiny_panel(2)]}
    path = tmp_path / "examples.png"

    plot_gradcam_examples(examples, path)

    assert path.read_bytes()[:4] == b"\x89PNG"


def test_plot_mistakes_writes_png(tmp_path):
    items = []
    for i in range(3):
        image, mask, heatmap = tiny_panel(i)
        items.append({"image": image, "mask": mask, "heatmap": heatmap,
                      "label": f"said glioma (0.9{i})", "confident": i == 0})
    path = tmp_path / "mistakes.png"

    plot_mistakes(items, "True meningioma: every test mistake", path)

    assert path.read_bytes()[:4] == b"\x89PNG"
```

(The existing `test_plot_augmentation_writes_png` keeps its own inner `import numpy as np`; that is harmless.)

- [ ] **Step 2: Run them to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_plots.py -v
```
Expected: `ImportError: cannot import name 'plot_gradcam_examples' from 'src.plots'`.

- [ ] **Step 3: Implement**

In `src/plots.py`, add `import math` as the first import (above `import matplotlib`), then append:

```python
def draw_outline(ax, mask):
    """Draw the clinician's tumour outline in green."""
    ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.8)


def plot_gradcam_examples(examples, path):
    """Correctly classified scans, one row per tumour type.

    examples maps type name -> list of (image, mask, heatmap). Each example is
    a pair of panels: the scan with the tumour outline, then the Grad-CAM
    heatmap over the scan (red = the regions that drove the decision).
    """
    columns = 2 * max(len(items) for items in examples.values())
    fig, axes = plt.subplots(len(examples), columns, figsize=(1.9 * columns, 2.1 * len(examples)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for row, (class_name, items) in enumerate(examples.items()):
        for k, (image, mask, heatmap) in enumerate(items):
            scan_ax, heat_ax = axes[row, 2 * k], axes[row, 2 * k + 1]
            scan_ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            draw_outline(scan_ax, mask)
            scan_ax.set_title(class_name, fontsize=9)
            heat_ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            heat_ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            heat_ax.set_title("heatmap", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_mistakes(items, title, path, columns=5):
    """Every mistake for one true tumour type: heatmap over the scan, with the tumour outline.

    items: dicts with image, mask, heatmap, label and confident.
    Confident mistakes are titled in red.
    """
    rows = math.ceil(len(items) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 2.5 * rows + 0.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, items):
        ax.imshow(item["image"], cmap="gray", vmin=0, vmax=1)
        ax.imshow(item["heatmap"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        draw_outline(ax, item["mask"])
        ax.set_title(item["label"], fontsize=7, color="red" if item["confident"] else "black")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Run the tests**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_plots.py -v
```
Expected: 5 passed. Then `.\.venv\Scripts\python.exe -m pytest -q` — 61 passed.

- [ ] **Step 5: Commit**

```powershell
git add src/plots.py tests/test_plots.py
git commit -m "feat: Grad-CAM example and mistake gallery figures" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The analysis run

**Concept:** explain only after scoring. The run refuses to start until the test score exists, and re-checks that its predictions reproduce that score. It cannot be used to steer the model, only to understand the one that was chosen.

**Files:**
- Modify: `src/explain.py`
- Test: `tests/test_explain.py`

- [ ] **Step 1: Write the failing tests**

In `tests/test_explain.py`, replace the whole import block at the top with:

```python
import json

import numpy as np
import pytest
import torch

from src.explain import (class_scores, feature_maps, grad_cam, mask_share, patient_errors, pointing_chance,
                         pointing_hit, run_explanation, summarise)
from src.models import ResNet18Grey, SmallCNN
from src.score import score_experiment
from tests.test_train import write_tiny_dataset
```

Append:

```python
def tiny_scored_resnet(tmp_path):
    """A tiny dataset with real tumour masks, and an untrained ResNet already scored on its test set."""
    dataset_path, splits_path = write_tiny_dataset(tmp_path)
    with np.load(dataset_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["masks"] = (arrays["images"] == 255).astype(np.uint8)  # the bright square is the "tumour"
    np.savez_compressed(dataset_path, **arrays)

    experiments_dir = tmp_path / "experiments"
    folder = experiments_dir / "final"
    folder.mkdir(parents=True)
    torch.manual_seed(0)
    torch.save(ResNet18Grey(weights=None).state_dict(), folder / "model.pt")
    (folder / "config.json").write_text(json.dumps({"model": "resnet18", "subset": 0}))
    (folder / "metrics.json").write_text(json.dumps({"best_epoch": 1, "accuracy": 0.0}))
    paths = {"dataset_path": dataset_path, "splits_path": splits_path, "experiments_dir": experiments_dir}
    score_experiment("final", "test", **paths)
    return paths


def test_explanation_refuses_before_the_test_score(tmp_path):
    with pytest.raises(RuntimeError, match="test_metrics"):
        run_explanation("final", experiments_dir=tmp_path)


def test_explanation_writes_every_output(tmp_path):
    paths = tiny_scored_resnet(tmp_path)

    summary = run_explanation("final", **paths)

    out_dir = paths["experiments_dir"] / "final" / "explain"
    assert summary["reproduces_saved_test_score"] is True
    assert summary["overall"]["scans"] == 9
    assert len((out_dir / "test_scans.csv").read_text().strip().splitlines()) == 10  # header + 9 scans
    assert len((out_dir / "patient_errors.csv").read_text().strip().splitlines()) == 4  # header + 3 patients
    assert json.loads((out_dir / "summary.json").read_text())["overall"]["scans"] == 9
    assert (out_dir / "gradcam_examples.png").exists()
```

- [ ] **Step 2: Run them to verify they fail**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_explain.py -v
```
Expected: `ImportError: cannot import name 'run_explanation' from 'src.explain'`.

- [ ] **Step 3: Implement**

In `src/explain.py`, replace the import block (everything between the module docstring and `TOLERANCE`) with:

```python
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data import CLASS_NAMES, DATASET_PATH, SPLITS_PATH, MRIDataset, indices_for_split, load_dataset, load_splits
from src.models import ResNet18Grey, build_model
from src.plots import plot_gradcam_examples, plot_mistakes
from src.score import TEST_RESULTS
from src.train import EXPERIMENTS_DIR
```

and add below the `CONFIDENT` line:

```python
EXAMPLES_PER_TYPE = 4
```

Append:

```python
def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_explanation(name, dataset_path=DATASET_PATH, splits_path=SPLITS_PATH, experiments_dir=EXPERIMENTS_DIR,
                    seed=0):
    """Explain every test prediction of the scored final model and save the results.

    Refuses to run before the test score exists: this stage explains the
    final model's test answers; it never helps choose or change a model.
    Writes test_scans.csv, patient_errors.csv, summary.json and the figures
    into the experiment's explain/ folder.
    """
    folder = Path(experiments_dir) / name
    if not (folder / TEST_RESULTS).exists():
        raise RuntimeError(f"{name} has no {TEST_RESULTS}: only the final model is explained, "
                           "after its one test score.")
    saved = json.loads((folder / TEST_RESULTS).read_text())
    config = json.loads((folder / "config.json").read_text())
    model = build_model(config["model"], n_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(folder / "model.pt", weights_only=True))
    model.eval()

    data = load_dataset(dataset_path)
    test_idx = indices_for_split(data["patient_ids"], load_splits(splits_path), "test")
    dataset = MRIDataset(data["images"], data["labels"], test_idx)

    rows, heatmaps = [], []
    for position, index in enumerate(test_idx):
        image, label = dataset[position]
        heatmap, predicted, confidence = grad_cam(model, image)
        mask = data["masks"][index]
        rows.append({
            "index": int(index),
            "patient_id": str(data["patient_ids"][index]),
            "true": CLASS_NAMES[label],
            "predicted": CLASS_NAMES[predicted],
            "confidence": round(confidence, 4),
            "correct": predicted == label,
            "pointing_hit": pointing_hit(heatmap, mask),
            "pointing_chance": round(pointing_chance(mask), 5),
            "mask_share": round(mask_share(heatmap, mask), 5),
            "mask_area": round(float(mask.mean()), 5),
        })
        heatmaps.append(heatmap)

    accuracy = round(float(np.mean([row["correct"] for row in rows])), 4)
    summary = summarise(rows)
    summary["test_accuracy"] = accuracy
    summary["reproduces_saved_test_score"] = accuracy == saved["accuracy"]
    print(f"Test accuracy {accuracy:.1%} against saved {saved['accuracy']:.1%} -> "
          f"{'MATCH' if summary['reproduces_saved_test_score'] else 'MISMATCH'}")

    out_dir = folder / "explain"
    out_dir.mkdir(exist_ok=True)
    write_csv(rows, out_dir / "test_scans.csv")
    write_csv(patient_errors(rows), out_dir / "patient_errors.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def panel(position):
        index = rows[position]["index"]
        return data["images"][index] / 255.0, data["masks"][index], heatmaps[position]

    # Examples are drawn at random with a fixed seed, never hand-picked.
    rng = np.random.default_rng(seed)
    examples = {}
    for class_name in CLASS_NAMES:
        correct = [p for p, row in enumerate(rows) if row["true"] == class_name and row["correct"]]
        if correct:
            chosen = sorted(rng.choice(correct, size=min(EXAMPLES_PER_TYPE, len(correct)), replace=False))
            examples[class_name] = [panel(p) for p in chosen]
    if examples:
        plot_gradcam_examples(examples, out_dir / "gradcam_examples.png")

    for class_name in CLASS_NAMES:
        items = []
        for p, row in enumerate(rows):
            if row["true"] != class_name or row["correct"]:
                continue
            image, mask, heatmap = panel(p)
            confident = row["confidence"] >= CONFIDENT
            label = (f"said {row['predicted']} ({row['confidence']:.2f})"
                     f"{' CONFIDENT' if confident else ''}\npatient {row['patient_id']}")
            items.append({"image": image, "mask": mask, "heatmap": heatmap, "label": label, "confident": confident})
        if items:
            plot_mistakes(items, f"True {class_name}: every test mistake", out_dir / f"mistakes_{class_name}.png")

    overall = summary["overall"]
    print(f"Pointing game: peak on the tumour in {overall['pointing_hit_rate']:.1%} of scans "
          f"(luck: {overall['pointing_chance']:.1%})")
    print(f"In-mask share: {overall['mean_mask_share']:.1%} of the heatmap inside the outline "
          f"(luck: {overall['mean_mask_area']:.1%})")
    print(f"Confident mistakes: {summary['confident_mistakes']}. Saved to {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Explain the final model's test predictions with Grad-CAM.")
    parser.add_argument("--name", required=True, help="experiment folder name, e.g. 03-resnet18-finetuned")
    args = parser.parse_args()
    run_explanation(args.name)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the full suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: 63 passed. Then `.\.venv\Scripts\python.exe -m src.explain --help` (help only) — prints usage with `--name`. Then confirm the real final experiment is untouched:

```powershell
Test-Path experiments\03-resnet18-finetuned\explain
git status --short
```
Expected: `False`, and a clean working tree after the commit below.

- [ ] **Step 5: Commit**

```powershell
git add src/explain.py tests/test_explain.py
git commit -m "feat: explain the final model's test predictions" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Run the analysis (Robbi runs this, or asks Claude to)

- [ ] **Step 1: Run it**

```powershell
.\.venv\Scripts\python.exe -m src.explain --name 03-resnet18-finetuned
```
Expected: `Test accuracy 90.9% against saved 90.9% -> MATCH`, then the pointing-game, in-mask and confident-mistake lines, and `Saved to ...explain`. A few minutes on the CPU. If it says MISMATCH, stop and investigate with superpowers:systematic-debugging.

- [ ] **Step 2: Commit the outputs**

```powershell
git add experiments/03-resnet18-finetuned/explain
git commit -m "exp: Grad-CAM analysis of the final model's test predictions" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: ⏸ PAUSE — review with Robbi

- [ ] **Step 1: Gather**

Read `explain/summary.json` and `explain/patient_errors.csv`; view `gradcam_examples.png` and each `mistakes_*.png` (Read tool).

- [ ] **Step 2: Discuss**

- Pointing-game hit rate against luck, overall and per type; in-mask share as "× luck".
- Correct versus wrong answers: does attention drift off the tumour when the model is wrong?
- Meningioma: how many of the 19 misses come from how many patients?
- Confident mistakes and any visible patterns in the galleries (describe only what is seen).

Report findings plainly, including unflattering ones. Do not change the model.

---

### Task 9: Report, notes and publish

**Files:**
- Modify: `README.md`, `docs/learning-notes.md`

- [ ] **Step 1: Update `README.md`**

Fill every angle-bracket value from `summary.json`, `patient_errors.csv` and the Task 8 discussion; do not invent figures.

1. Status line:

```markdown
> **Status:** in progress. Experiments 1–4, the one-shot test score and the Grad-CAM analysis are complete; a live demo is next.
```

2. Insert after the `## Final test results` section and before `## Limitations`:

```markdown
## Where the model looks

Grad-CAM, written by hand in `src/explain.py`, turns each decision into a heatmap of where the evidence came from, taken from ResNet-18's last convolutional stage. It was run on all 430 test scans after the test score was final; nothing found here was used to change the model.

![Grad-CAM examples](experiments/03-resnet18-finetuned/explain/gradcam_examples.png)

*Random correctly classified test scans (seed 0, not hand-picked): tumour outline in green, then the heatmap — red is where the evidence came from.*

**Measuring it.** The original plan was to call a scan "focused" if at least half the heatmap fell inside the clinicians' tumour outline. The data rules that out: tumours cover a median of 0.7–1.6% of the image, while each cell of ResNet-18's 7×7 heatmap covers about 2%, so even perfectly placed attention spills far outside the outline. Two measures that fit, each with the score luck alone would give:

- **Pointing game:** is the heatmap's hottest pixel on the tumour (within 8 pixels)?
- **In-mask share:** what fraction of the heatmap falls inside the outline?

| | Scans | Pointing game | Luck | In-mask share | Luck | × luck |
|---|---|---|---|---|---|---|
| All test scans | 430 | <rate> | <chance> | <share> | <area> | <×> |
| Correct answers | <n> | <rate> | <chance> | <share> | <area> | <×> |
| Wrong answers | <n> | <rate> | <chance> | <share> | <area> | <×> |
| Meningioma | 80 | <rate> | <chance> | <share> | <area> | <×> |
| Glioma | 222 | <rate> | <chance> | <share> | <area> | <×> |
| Pituitary | 128 | <rate> | <chance> | <share> | <area> | <×> |

<2–4 sentences agreed in Task 8.>

## Mistakes

All <n> wrong test answers, grouped by true type. Titles give the model's answer, its confidence and the patient; red titles are confident mistakes (confidence ≥ 0.9): <number> of them.

![Meningioma mistakes](experiments/03-resnet18-finetuned/explain/mistakes_meningioma.png)
![Glioma mistakes](experiments/03-resnet18-finetuned/explain/mistakes_glioma.png)
![Pituitary mistakes](experiments/03-resnet18-finetuned/explain/mistakes_pituitary.png)

**Meningioma misses by patient:**

| Patient | Slices | Wrong | Mostly called |
|---|---|---|---|
| <one row per meningioma patient with at least one mistake, from patient_errors.csv> |

<2–4 sentences agreed in Task 8: concentration by patient, confident mistakes, visible patterns.>
```

Omit any gallery image line whose file was not produced.

3. In `## How to run`, add after the last `src.score` line:

```
    .\.venv\Scripts\python.exe -m src.explain --name 03-resnet18-finetuned
```

- [ ] **Step 2: Append three sections to `docs/learning-notes.md`**

Fill the angle-bracket figures; keep the rest as written.

```markdown
## 17. Grad-CAM

A trained network gives an answer, not a reason. Grad-CAM recovers a rough "where". ResNet-18's last stage turns the scan into 512 pattern maps, each a 7×7 grid of how strongly one learned pattern appears where. The decision averages each map and weighs them into the three class scores.

Grad-CAM asks how much the winning score would rise if each map got stronger — the gradient — and uses that as the map's importance. The weighted maps are added up, negative evidence is dropped, and the 7×7 result is stretched over the scan. Red means "the evidence for this answer came from here".

It shows where, not why: a hot region says the model used that area, not what it saw there.

> **Say:** "Grad-CAM weights the last layer's pattern maps by how much each one pushed the winning class score, then overlays the result on the scan. I wrote it by hand — two halves of the forward pass and one gradient call — so I can explain exactly what the heatmap is."

## 18. Measuring attention: the pointing game and luck baselines

Looking at a few heatmaps proves little; anyone can pick flattering examples. The dataset includes the clinicians' tumour outline for every scan, so attention can be measured.

The pointing game asks whether the heatmap's hottest point lands on the tumour (with 8 pixels of leeway). It hit in <rate> of test scans. On its own that number is meaningless — so it is compared with luck: a randomly placed point would hit <chance> of the time, because that is how much of the image lies near a tumour. The in-mask share gets the same treatment: <share> of the heatmap fell inside the outline, <×> times what an evenly spread heatmap would manage.

> **Say:** "I measured attention against the clinicians' tumour outlines rather than eyeballing heatmaps. The model's peak lands on the tumour <rate> of the time, against <chance> by chance — and every number I report comes with what luck alone would score."

## 19. Why the heatmap is coarse

ResNet-18's last stage sees the scan as a 7×7 grid, so each Grad-CAM cell covers 32×32 pixels — about 2% of the image. The tumours here cover a median of 0.7–1.6%. A heatmap cell is bigger than the typical tumour, so even perfect attention spreads well beyond the outline.

That is why the original "at least half the heatmap inside the tumour" test was dropped before running anything: it would have failed almost every scan and wrongly suggested the model ignores tumours. Spotting that a measurement cannot fit the data is part of the job.

> **Say:** "The heatmap's resolution is 7×7, and each cell is bigger than most of these tumours, so I replaced a 'half inside the outline' test that could never pass with the pointing game — the measurement has to fit the data."
```

- [ ] **Step 3: Check nothing was left unfilled**

```powershell
Select-String -Path README.md, docs\learning-notes.md -Pattern '<[a-z0-9]'
```
Expected: no matches.

- [ ] **Step 4: Run the full suite**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
Expected: 63 passed.

- [ ] **Step 5: Commit**

```powershell
git add README.md docs/learning-notes.md
git commit -m "docs: stage 4 Grad-CAM results, mistakes and notes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 6: Robbi reviews the wording, then merge and push**

Only after Robbi approves:

```powershell
git checkout main
git merge feat/stage-4-gradcam --ff-only
git push
git branch -d feat/stage-4-gradcam
```

---

## Done when

- `.\.venv\Scripts\python.exe -m pytest` passes in full (63 tests).
- `experiments/03-resnet18-finetuned/explain/` holds `test_scans.csv`, `patient_errors.csv`, `summary.json` and the figures, committed; the summary shows the saved test score reproduced.
- Robbi has reviewed the findings and the wording; README and notes are merged to `main` and pushed.
- **Next:** Stage 5 plan — the Gradio demo on Hugging Face Spaces.
