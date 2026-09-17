# Stages 0–1: Data Pipeline and First Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download and prepare the Cheng et al. brain MRI dataset with a patient-level split, then train a small CNN from scratch and record experiment 1 with charts and per-class scores.

**Architecture:** `src/download.py` turns the four Figshare zips into one compressed `data/cheng_224.npz` plus `data/splits.csv`. `src/data.py` holds preprocessing, the split and the PyTorch `Dataset`. `src/metrics.py` is pure Python scoring. `src/models.py` defines the network. `src/evaluate.py` scores a model. `src/train.py` wires everything together and writes one folder per experiment under `experiments/`.

**Tech Stack:** Python 3.13 (Windows, `py` launcher), PyTorch (CPU, from PyPI), NumPy, h5py, Pillow, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-09-17-brain-tumour-mri-design.md`. This plan covers spec §10 Stages 0 and 1 only. Stages 2–6 get their own plans after the Stage 1 results review.

---

## Ground rules for whoever executes this

- **Platform:** Windows, PowerShell. All commands run from `C:\Users\Robbi\brain-tumour-mri`.
- **Always call the venv's Python directly:** `.\.venv\Scripts\python.exe`. Bare `python` on this machine opens the Microsoft Store.
- **Robbi is a beginner and learning.**
  - Each task starts with a **Concept** line. Explain that concept to Robbi in plain English before writing the code.
  - Tasks run straight through without check-ins, **except** the marked **PAUSE** in Task 11.
- **Disk:** about 10.8 GB free. Don't install anything not listed here.
- **Facts verified on 2026-09-17** against the real dataset:
  - Files are MATLAB v7.3 (HDF5) at the zip root, named `1.mat`, `2.mat`, …
  - Group `cjdata` has `PID` (uint16 character codes, shape (6,1)), `image` (int16, 512×512), `label` (float64 (1,1), 1=meningioma 2=glioma 3=pituitary), `tumorBorder`, `tumorMask` (uint8 512×512).
- **Commit messages** end with the trailer `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>` (passed as a second `-m`).

## File map

| File | Responsibility |
|---|---|
| `requirements.txt` | Python dependencies |
| `.gitignore` | Keep data, venv, checkpoints out of git |
| `src/__init__.py` | Makes `src` importable as a package |
| `src/metrics.py` | Confusion matrix, accuracy, per-class precision/recall (no PyTorch) |
| `src/data.py` | Class names, paths, image/mask preprocessing, patient split, split CSV, `MRIDataset` |
| `src/download.py` | Disk-space check, download, read `.mat`, convert zips → `.npz`, make split, delete raw |
| `src/models.py` | `SmallCNN` and `build_model` |
| `src/evaluate.py` | `evaluate_model`: loss, accuracy, predictions, confidences |
| `src/plots.py` | Training-curve and confusion-matrix PNGs |
| `src/train.py` | Seeds, class weights, one training epoch, full experiment run, CLI |
| `tests/test_*.py` | One test file per `src` module |
| `docs/learning-notes.md` | Plain-English concept notes |
| `README.md` | The report (v1) |

---

### Task 1: Project setup and environment

**Concept:** what a virtual environment is (a private copy of Python and libraries per project) and what `requirements.txt` records.

**Files:**
- Create: `requirements.txt`, `.gitignore`, `src/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Create `requirements.txt`**

```text
# PyTorch from PyPI is CPU-only on Windows, which is what this laptop needs.
torch>=2.6
numpy
h5py
pillow
matplotlib
pytest
```

- [ ] **Step 2: Create `.gitignore`**

```text
.venv/
__pycache__/
.pytest_cache/
data/
experiments/*/model.pt
experiments/smoke/
```

- [ ] **Step 3: Create empty package markers**

Create `src/__init__.py` and `tests/__init__.py`, each containing a single empty line.

- [ ] **Step 4: Create the virtual environment and install**

Run these one at a time:
```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Expected: ends with `Successfully installed ...` including `torch`, `h5py`, `matplotlib`, `pytest`. The torch download is about 200 MB.

- [ ] **Step 5: Confirm the libraries load**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import torch, h5py, numpy, PIL, matplotlib; print('torch', torch.__version__, '| GPU available:', torch.cuda.is_available())"
```
Expected: `torch 2.x.x+cpu | GPU available: False` (the `+cpu` suffix may be absent). `False` is correct for this laptop.

- [ ] **Step 6: Commit**

```powershell
git add requirements.txt .gitignore src/__init__.py tests/__init__.py
git commit -m "chore: project skeleton and dependencies" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: Metrics

**Concept:** accuracy vs precision vs recall, and reading a confusion matrix. Why a missed tumour (low recall) matters more than accuracy alone.

**Files:**
- Create: `src/metrics.py`
- Test: `tests/test_metrics.py`

Hand-worked example used by the tests: `y_true = [0,0,0,1,1,2]`, `y_pred = [0,0,1,1,1,0]`.
- **Confusion matrix** (row = true, column = predicted): `[[2,1,0],[0,2,0],[1,0,0]]`.
- **Accuracy:** 4 correct of 6 = 0.6667.
- **Class 0:** predicted-as-0 = 2+0+1 = 3 → precision 2/3; actually-0 = 3 → recall 2/3.
- **Class 1:** predicted-as-1 = 3 → precision 2/3; actually-1 = 2 → recall 1.0.
- **Class 2:** never predicted → precision 0.0 (no divide-by-zero); actually-2 = 1 → recall 0.0.

- [ ] **Step 1: Write the failing tests**

`tests/test_metrics.py`:
```python
import pytest

from src.metrics import accuracy, confusion_matrix, precision_recall

Y_TRUE = [0, 0, 0, 1, 1, 2]
Y_PRED = [0, 0, 1, 1, 1, 0]


def test_confusion_matrix_rows_are_true_columns_are_predicted():
    assert confusion_matrix(Y_TRUE, Y_PRED, n_classes=3) == [
        [2, 1, 0],
        [0, 2, 0],
        [1, 0, 0],
    ]


def test_accuracy_is_fraction_correct():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)


def test_accuracy_rejects_empty_input():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_precision_and_recall_match_hand_calculation():
    matrix = confusion_matrix(Y_TRUE, Y_PRED, n_classes=3)
    scores = precision_recall(matrix)

    assert scores[0]["precision"] == pytest.approx(2 / 3)
    assert scores[0]["recall"] == pytest.approx(2 / 3)
    assert scores[1]["precision"] == pytest.approx(2 / 3)
    assert scores[1]["recall"] == pytest.approx(1.0)


def test_class_never_predicted_scores_zero_instead_of_crashing():
    matrix = confusion_matrix(Y_TRUE, Y_PRED, n_classes=3)
    scores = precision_recall(matrix)

    assert scores[2] == {"precision": 0.0, "recall": 0.0}


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0], n_classes=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_metrics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.metrics'`

- [ ] **Step 3: Write the implementation**

`src/metrics.py`:
```python
"""Scoring functions for classification results.

Plain Python with no PyTorch, so every function can be checked against
examples worked out by hand.
"""


def confusion_matrix(y_true, y_pred, n_classes):
    """Count how often each true class was predicted as each class.

    Row = true class, column = predicted class. matrix[1][0] is the number
    of class-1 scans the model called class 0.
    """
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for true, predicted in zip(y_true, y_pred, strict=True):
        matrix[true][predicted] += 1
    return matrix


def accuracy(y_true, y_pred):
    """Fraction of predictions that are correct."""
    if len(y_true) == 0:
        raise ValueError("no predictions to score")
    correct = sum(1 for true, predicted in zip(y_true, y_pred, strict=True) if true == predicted)
    return correct / len(y_true)


def precision_recall(matrix):
    """Per-class precision and recall from a confusion matrix.

    precision: of the scans predicted as this class, the fraction that really were.
    recall:    of the scans that really are this class, the fraction the model caught.

    A class with nothing to divide by scores 0.0 rather than crashing.
    """
    n = len(matrix)
    scores = []
    for c in range(n):
        true_positive = matrix[c][c]
        predicted_as_c = sum(matrix[row][c] for row in range(n))
        actually_c = sum(matrix[c])
        scores.append({
            "precision": true_positive / predicted_as_c if predicted_as_c else 0.0,
            "recall": true_positive / actually_c if actually_c else 0.0,
        })
    return scores
```

Note: `zip(..., strict=True)` raises `ValueError` when lengths differ. This is what the mismatched-lengths test relies on.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_metrics.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```powershell
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: accuracy, confusion matrix and per-class precision/recall" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Image and mask preprocessing

**Concept:** an image is just a grid of numbers. Why MRI brightness must be rescaled per slice, and why masks use nearest-neighbour resizing (so they stay 0/1).

**Files:**
- Create: `src/data.py`
- Test: `tests/test_data.py`

Images are stored as `uint8` (0–255) to keep the dataset about 4× smaller than float32. `MRIDataset` (Task 5) divides by 255 to give the [0, 1] range the spec requires.

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:
```python
import numpy as np

from src.data import IMAGE_SIZE, preprocess_image, preprocess_mask


def test_preprocess_image_resizes_to_224_uint8():
    raw = np.random.default_rng(0).integers(0, 3000, size=(512, 512)).astype(np.int16)

    out = preprocess_image(raw)

    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE) == (224, 224)
    assert out.dtype == np.uint8


def test_preprocess_image_stretches_to_full_brightness_range():
    raw = np.zeros((512, 512), dtype=np.int16)
    raw[:256, :] = 1000
    raw[256:, :] = 3000

    out = preprocess_image(raw)

    assert out.min() == 0
    assert out.max() == 255


def test_preprocess_image_handles_blank_slice_without_dividing_by_zero():
    raw = np.full((512, 512), 7, dtype=np.int16)

    out = preprocess_image(raw)

    assert (out == 0).all()


def test_preprocess_mask_stays_binary_and_keeps_tumour():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1

    out = preprocess_mask(mask)

    assert out.shape == (224, 224)
    assert set(np.unique(out)) == {0, 1}
    assert out.sum() > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data'`

- [ ] **Step 3: Write the implementation**

`src/data.py`:
```python
"""Loading, preparing and splitting the brain MRI scans."""

from pathlib import Path

import numpy as np
from PIL import Image

CLASS_NAMES = ["meningioma", "glioma", "pituitary"]
IMAGE_SIZE = 224

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_PATH = DATA_DIR / "cheng_224.npz"
SPLITS_PATH = DATA_DIR / "splits.csv"


def preprocess_image(image, size=IMAGE_SIZE):
    """Rescale a raw MRI slice to 0-255 and resize it to size x size.

    MRI brightness has no fixed scale, so each slice is stretched so its
    darkest pixel becomes 0 and its brightest 255. Stored as uint8 to keep
    the dataset small; MRIDataset divides by 255 to give values in [0, 1].
    """
    image = image.astype(np.float32)
    low, high = image.min(), image.max()
    if high > low:
        image = (image - low) / (high - low)
    else:
        image = np.zeros_like(image)
    resized = Image.fromarray(image).resize((size, size), Image.Resampling.BILINEAR)
    return np.clip(np.round(np.asarray(resized) * 255), 0, 255).astype(np.uint8)


def preprocess_mask(mask, size=IMAGE_SIZE):
    """Resize a tumour mask while keeping every pixel exactly 0 or 1.

    Nearest-neighbour resizing copies existing pixels instead of blending
    them, so no in-between values appear at the tumour edge.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    resized = Image.fromarray(binary).resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(resized) > 127).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```powershell
git add src/data.py tests/test_data.py
git commit -m "feat: MRI slice and tumour mask preprocessing" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Patient-level split

**Concept:** data leakage. Why neighbouring slices from one patient in both training and test data inflate the score, and why the split is stratified by tumour type.

**Files:**
- Modify: `src/data.py` (add imports at top; add functions at bottom)
- Test: `tests/test_data.py` (append)

Hand check for the stratification test: 20 patients per class, 15% each for validation and test.
- **Per class:** `round(20 × 0.15) = 3` validation, 3 test, 14 train.
- **Across 3 classes:** 9 validation, 9 test, 42 train patients.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:
```python
import pytest

from src.data import indices_for_split, load_splits, save_splits, split_patients


def make_patients(per_class=20, slices_per_patient=3):
    """Synthetic slice-level patient IDs and labels: every patient has one label."""
    patient_ids, labels = [], []
    for label in range(3):
        for p in range(per_class):
            for _ in range(slices_per_patient):
                patient_ids.append(f"{label}{p:03d}")
                labels.append(label)
    return patient_ids, labels


def test_every_patient_is_assigned_exactly_once():
    patient_ids, labels = make_patients()

    assignment = split_patients(patient_ids, labels, seed=42)

    assert set(assignment) == set(patient_ids)
    assert set(assignment.values()) == {"train", "validation", "test"}


def test_no_slice_indices_shared_between_splits():
    patient_ids, labels = make_patients()
    assignment = split_patients(patient_ids, labels, seed=42)

    train = set(indices_for_split(patient_ids, assignment, "train"))
    val = set(indices_for_split(patient_ids, assignment, "validation"))
    test = set(indices_for_split(patient_ids, assignment, "test"))

    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train) + len(val) + len(test) == len(patient_ids)


def test_split_is_stratified_by_tumour_type():
    patient_ids, labels = make_patients(per_class=20)
    assignment = split_patients(patient_ids, labels, seed=42)
    label_of = dict(zip(patient_ids, labels))

    for split, expected in [("train", 14), ("validation", 3), ("test", 3)]:
        for label in range(3):
            count = sum(1 for pid, s in assignment.items() if s == split and label_of[pid] == label)
            assert count == expected, (split, label, count)


def test_same_seed_gives_same_split_and_different_seed_differs():
    patient_ids, labels = make_patients()

    first = split_patients(patient_ids, labels, seed=42)
    again = split_patients(patient_ids, labels, seed=42)
    other = split_patients(patient_ids, labels, seed=7)

    assert first == again
    assert first != other


def test_patient_with_two_tumour_types_is_rejected():
    with pytest.raises(ValueError):
        split_patients(["A", "A"], [0, 1], seed=42)


def test_splits_survive_saving_and_loading(tmp_path):
    patient_ids, labels = make_patients(per_class=4)
    assignment = split_patients(patient_ids, labels, seed=42)
    path = tmp_path / "splits.csv"

    save_splits(assignment, path)

    assert load_splits(path) == assignment
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: FAIL with `ImportError: cannot import name 'indices_for_split'`

- [ ] **Step 3: Write the implementation**

In `src/data.py`, replace the import block at the top with:
```python
import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
```

Append to the bottom of `src/data.py`:
```python
def split_patients(patient_ids, labels, seed=42, val_fraction=0.15, test_fraction=0.15):
    """Assign every patient to 'train', 'validation' or 'test'.

    The split is by patient, never by slice: all slices from one patient go
    to the same set, so the model is never scored on a patient it trained on.
    It is stratified: each tumour type is split separately, so every set
    gets a similar mix of types.

    Returns a dict of patient_id -> split name.
    """
    label_of = {}
    for pid, label in zip(patient_ids, labels, strict=True):
        pid, label = str(pid), int(label)
        if label_of.setdefault(pid, label) != label:
            raise ValueError(f"patient {pid} has slices with different tumour types")

    patients_by_label = defaultdict(list)
    for pid, label in label_of.items():
        patients_by_label[label].append(pid)

    rng = random.Random(seed)
    assignment = {}
    for label in sorted(patients_by_label):
        patients = sorted(patients_by_label[label])
        rng.shuffle(patients)
        n_val = round(len(patients) * val_fraction)
        n_test = round(len(patients) * test_fraction)
        for pid in patients[:n_val]:
            assignment[pid] = "validation"
        for pid in patients[n_val:n_val + n_test]:
            assignment[pid] = "test"
        for pid in patients[n_val + n_test:]:
            assignment[pid] = "train"
    return assignment


def save_splits(assignment, path):
    """Write patient_id -> split to CSV so every experiment reuses the same split."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["patient_id", "split"])
        for pid in sorted(assignment):
            writer.writerow([pid, assignment[pid]])


def load_splits(path):
    """Read the patient_id -> split mapping written by save_splits."""
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["patient_id"]: row["split"] for row in csv.DictReader(handle)}


def indices_for_split(patient_ids, assignment, split):
    """Positions of every slice whose patient belongs to the given split."""
    return np.array(
        [i for i, pid in enumerate(patient_ids) if assignment[str(pid)] == split],
        dtype=np.int64,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```powershell
git add src/data.py tests/test_data.py
git commit -m "feat: stratified patient-level train/validation/test split" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Loading the dataset for PyTorch

**Concept:** what a tensor is, and how a `Dataset` hands the model one (image, label) pair at a time, which a `DataLoader` groups into batches.

**Files:**
- Modify: `src/data.py` (imports + append)
- Test: `tests/test_data.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_data.py`:
```python
import torch

from src.data import MRIDataset, load_dataset


def test_dataset_returns_scaled_single_channel_tensor_and_label():
    images = np.array([np.zeros((8, 8)), np.full((8, 8), 255)], dtype=np.uint8)
    labels = np.array([2, 0], dtype=np.int64)
    dataset = MRIDataset(images, labels, indices=np.array([1, 0]))

    image, label = dataset[0]

    assert len(dataset) == 2
    assert image.shape == (1, 8, 8)
    assert image.dtype == torch.float32
    assert image.min() >= 0.0 and image.max() <= 1.0
    assert image.max() == 1.0
    assert label == 0


def test_load_dataset_reads_all_arrays(tmp_path):
    path = tmp_path / "tiny.npz"
    np.savez_compressed(
        path,
        images=np.zeros((2, 4, 4), dtype=np.uint8),
        masks=np.zeros((2, 4, 4), dtype=np.uint8),
        labels=np.array([0, 1]),
        patient_ids=np.array(["100360", "101016"]),
    )

    data = load_dataset(path)

    assert set(data) == {"images", "masks", "labels", "patient_ids"}
    assert data["patient_ids"][1] == "101016"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: FAIL with `ImportError: cannot import name 'MRIDataset'`

- [ ] **Step 3: Write the implementation**

In `src/data.py`, add these imports under `from PIL import Image`:
```python
import torch
from torch.utils.data import Dataset
```

Append to the bottom of `src/data.py`:
```python
def load_dataset(path=DATASET_PATH):
    """Load the prepared arrays: images, masks, labels, patient_ids."""
    with np.load(path) as archive:
        return {name: archive[name] for name in archive.files}


class MRIDataset(Dataset):
    """Serves (image tensor, label) pairs for the slices at the given indices.

    Images come out as float32 in [0, 1] with shape (1, height, width):
    one channel because MRI slices are greyscale.
    """

    def __init__(self, images, labels, indices):
        self.images = images
        self.labels = labels
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        i = self.indices[position]
        image = torch.from_numpy(self.images[i].astype(np.float32) / 255.0).unsqueeze(0)
        return image, int(self.labels[i])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_data.py -v`
Expected: 12 passed

- [ ] **Step 5: Commit**

```powershell
git add src/data.py tests/test_data.py
git commit -m "feat: PyTorch dataset over prepared MRI arrays" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Download and conversion script

**Concept:** where the data comes from, what a `.mat` file is, and why labels are shifted from 1–3 to 0–2 (Python counts from 0).

**Files:**
- Create: `src/download.py`
- Test: `tests/test_download.py`

The tests build fake `.mat` files with h5py in memory, so no download is needed. The image in the orientation test is deliberately non-square (4×6). That way a missing `.T` transpose fails loudly instead of silently rotating every scan.

- [ ] **Step 1: Write the failing tests**

`tests/test_download.py`:
```python
import io
import zipfile
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.download import check_disk_space, convert_zip, read_mat


def make_mat_bytes(image, mask, label, patient_id):
    """Build bytes shaped like a real dataset file (MATLAB v7.3 = HDF5).

    MATLAB writes arrays column-first, so the real files hold them transposed.
    """
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as f:
        group = f.create_group("cjdata")
        group["image"] = image.T.astype(np.int16)
        group["tumorMask"] = mask.T.astype(np.uint8)
        group["label"] = np.array([[float(label)]])
        group["PID"] = np.array([[ord(c)] for c in patient_id], dtype=np.uint16)
    return buffer.getvalue()


def gigabytes(n):
    return lambda path: SimpleNamespace(free=int(n * 1024**3))


def test_disk_check_refuses_when_space_is_low(tmp_path):
    with pytest.raises(RuntimeError, match="GB free"):
        check_disk_space(tmp_path, required_gb=3.0, disk_usage=gigabytes(1.5))


def test_disk_check_passes_when_space_is_enough(tmp_path):
    assert check_disk_space(tmp_path, required_gb=3.0, disk_usage=gigabytes(8)) == pytest.approx(8)


def test_read_mat_restores_orientation_label_and_patient_id():
    image = np.arange(24).reshape(4, 6)
    mask = np.zeros((4, 6))
    mask[1, 2] = 1

    out_image, out_mask, class_index, patient_id = read_mat(make_mat_bytes(image, mask, 2, "100360"))

    assert np.array_equal(out_image, image)
    assert np.array_equal(out_mask, mask)
    assert class_index == 1  # dataset label 2 = glioma = CLASS_NAMES[1]
    assert patient_id == "100360"


def test_convert_zip_reads_slices_in_numeric_order(tmp_path):
    zip_path = tmp_path / "part.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, label in [("10.mat", 3), ("2.mat", 2), ("1.mat", 1)]:
            image = np.random.default_rng(0).integers(0, 500, size=(64, 64))
            archive.writestr(name, make_mat_bytes(image, np.zeros((64, 64)), label, "123456"))

    slices = list(convert_zip(zip_path))

    assert [class_index for _, _, class_index, _ in slices] == [0, 1, 2]
    image, mask, _, patient_id = slices[0]
    assert image.shape == (224, 224) and image.dtype == np.uint8
    assert mask.shape == (224, 224)
    assert patient_id == "123456"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_download.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.download'`

- [ ] **Step 3: Write the implementation**

`src/download.py`:
```python
"""Download the Cheng et al. brain tumour MRI dataset and prepare it.

Usage:
    .\\.venv\\Scripts\\python.exe -m src.download

Dataset: Cheng, Jun (2017). brain tumor dataset. figshare.
https://doi.org/10.6084/m9.figshare.1512427 (CC BY 4.0)
"""

import io
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from src.data import (
    CLASS_NAMES,
    DATA_DIR,
    DATASET_PATH,
    SPLITS_PATH,
    preprocess_image,
    preprocess_mask,
    save_splits,
    split_patients,
)

ZIP_URLS = {
    "brainTumorDataPublic_1-766.zip": "https://ndownloader.figshare.com/files/3381290",
    "brainTumorDataPublic_767-1532.zip": "https://ndownloader.figshare.com/files/3381296",
    "brainTumorDataPublic_1533-2298.zip": "https://ndownloader.figshare.com/files/3381293",
    "brainTumorDataPublic_2299-3064.zip": "https://ndownloader.figshare.com/files/3381302",
}
RAW_DIR = DATA_DIR / "raw"
REQUIRED_FREE_GB = 3.0


def check_disk_space(path, required_gb=REQUIRED_FREE_GB, disk_usage=shutil.disk_usage):
    """Stop early with a clear message rather than filling the drive."""
    free_gb = disk_usage(path).free / 1024**3
    if free_gb < required_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free on this drive; need at least {required_gb:.1f} GB. "
            "Free up some space and run again."
        )
    return free_gb


def read_mat(raw_bytes):
    """Read one slice from a MATLAB v7.3 .mat file (HDF5 inside).

    Returns (image, mask, class_index, patient_id).
    - The dataset labels are 1=meningioma, 2=glioma, 3=pituitary; we return
      0, 1, 2 so they index CLASS_NAMES.
    - MATLAB stores arrays column-first, so h5py returns them transposed;
      .T puts rows and columns back the right way round.
    - The patient ID is stored as a column of character codes.
    """
    with h5py.File(io.BytesIO(raw_bytes), "r") as f:
        data = f["cjdata"]
        image = np.array(data["image"]).T
        mask = np.array(data["tumorMask"]).T
        class_index = int(np.array(data["label"]).ravel()[0]) - 1
        patient_id = "".join(chr(code) for code in np.array(data["PID"]).ravel())
    return image, mask, class_index, patient_id


def convert_zip(zip_path):
    """Yield (image, mask, class_index, patient_id) for each slice, resized to 224x224.

    Slices are read in numeric file order (1.mat, 2.mat, 10.mat) so the
    prepared dataset is always in the same order.
    """
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(
            (name for name in archive.namelist() if name.endswith(".mat")),
            key=lambda name: int(Path(name).stem),
        )
        for name in names:
            image, mask, class_index, patient_id = read_mat(archive.read(name))
            yield preprocess_image(image), preprocess_mask(mask), class_index, patient_id


def download_file(url, destination):
    """Download url to destination, reusing a previous complete download."""
    if destination.exists():
        print(f"  already downloaded: {destination.name}")
        return
    partial = destination.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    partial.rename(destination)


def build_arrays(zip_paths):
    """Convert every zip and stack the results into arrays."""
    images, masks, labels, patient_ids = [], [], [], []
    for zip_path in zip_paths:
        print(f"  converting {zip_path.name}")
        for image, mask, class_index, patient_id in convert_zip(zip_path):
            images.append(image)
            masks.append(mask)
            labels.append(class_index)
            patient_ids.append(patient_id)
    return {
        "images": np.stack(images),
        "masks": np.stack(masks),
        "labels": np.array(labels, dtype=np.int64),
        "patient_ids": np.array(patient_ids),
    }


def print_summary(arrays, assignment):
    labels, patient_ids = arrays["labels"], arrays["patient_ids"]
    print(f"\nSlices: {len(labels)}   Patients: {len(set(patient_ids))}")
    for index, name in enumerate(CLASS_NAMES):
        print(f"  {name:<11} {int((labels == index).sum()):>5} slices")
    print("\nSplit (patients / slices):")
    slices_per_split = Counter(assignment[str(pid)] for pid in patient_ids)
    patients_per_split = Counter(assignment.values())
    for split in ["train", "validation", "test"]:
        print(f"  {split:<11} {patients_per_split[split]:>4} / {slices_per_split[split]:>5}")


def main():
    if DATASET_PATH.exists():
        print(f"{DATASET_PATH} already exists. Delete the data folder to rebuild it.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    free_gb = check_disk_space(DATA_DIR)
    print(f"{free_gb:.1f} GB free - OK")

    zip_paths = []
    for name, url in ZIP_URLS.items():
        print(f"Downloading {name} (about 220 MB)...")
        download_file(url, RAW_DIR / name)
        zip_paths.append(RAW_DIR / name)

    print("Converting slices to 224x224...")
    arrays = build_arrays(zip_paths)
    np.savez_compressed(DATASET_PATH, **arrays)

    assignment = split_patients(arrays["patient_ids"], arrays["labels"])
    save_splits(assignment, SPLITS_PATH)

    shutil.rmtree(RAW_DIR)
    print(f"Saved {DATASET_PATH.name} and {SPLITS_PATH.name}; deleted raw downloads.")
    print_summary(arrays, assignment)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_download.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```powershell
git add src/download.py tests/test_download.py
git commit -m "feat: download and convert the Cheng et al. MRI dataset" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Run the download (Robbi runs this)

**Concept:** what the summary numbers mean, and checking them against the dataset's published counts.

- [ ] **Step 1: Robbi runs the download**

Run: `.\.venv\Scripts\python.exe -m src.download`
Expected: four downloads of ~220 MB each (a few minutes each), then conversion, then a summary. It must show:
- `Slices: 3064   Patients: 233`
- `meningioma 708`, `glioma 1426`, `pituitary 930`

Split rows show roughly 70/15/15 of patients.

If the patient count is not 233, or `split_patients` raises "different tumour types", stop and investigate. Don't work around it. It means the data doesn't match the published description.

- [ ] **Step 2: Check disk usage after cleanup**

Run: `Get-ChildItem data | Select-Object Name, @{n='MB';e={[math]::Round($_.Length/1MB,1)}}`
Expected: only `cheng_224.npz` (roughly 100–250 MB) and `splits.csv`. No `raw` folder.

- [ ] **Step 3: Visual sanity check**

Run:
```powershell
.\.venv\Scripts\python.exe -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; from src.data import load_dataset, CLASS_NAMES; d = load_dataset(); fig, axes = plt.subplots(1, 3, figsize=(9, 3)); [(ax.imshow(d['images'][i], cmap='gray'), ax.contour(d['masks'][i], colors='r', linewidths=0.8), ax.set_title(CLASS_NAMES[d['labels'][i]]), ax.axis('off')) for ax, i in zip(axes, [0, 1000, 2500])]; fig.savefig('data/sample.png', dpi=100)"
```
Open `data\sample.png`. Expected: three upright brain slices, each with a red outline sitting on a visible tumour. If outlines don't line up with tumours, the transpose is wrong. Stop and fix `read_mat`.

(No commit: `data/` is gitignored.)

---

### Task 8: The small CNN

**Concept:** convolution (a small filter sliding across the image looking for a pattern), ReLU, pooling, and how stacked layers go from edges to shapes to "tumour-like texture".

**Files:**
- Create: `src/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_models.py`:
```python
import pytest
import torch

from src.models import SmallCNN, build_model


def test_small_cnn_gives_three_scores_per_image():
    model = SmallCNN()

    output = model(torch.zeros(2, 1, 224, 224))

    assert output.shape == (2, 3)


def test_small_cnn_works_on_smaller_images_too():
    output = SmallCNN()(torch.zeros(4, 1, 32, 32))

    assert output.shape == (4, 3)


def test_build_model_by_name():
    assert isinstance(build_model("small_cnn"), SmallCNN)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("not_a_model")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models'`

- [ ] **Step 3: Write the implementation**

`src/models.py`:
```python
"""The neural networks."""

from torch import nn


def conv_block(in_channels, out_channels):
    """One pattern-finding stage.

    Conv2d:      slides 3x3 filters over the image, each looking for a pattern
    BatchNorm2d: keeps the numbers in a steady range so training is stable
    ReLU:        keeps positive responses, zeroes the rest
    MaxPool2d:   halves width and height, keeping the strongest responses
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    """A small convolutional network trained from scratch.

    Four blocks find increasingly complex patterns (16 -> 128 filters) while
    shrinking the image 224 -> 14. Average pooling turns each of the 128
    pattern maps into one number, and a final linear layer turns those 128
    numbers into one score per tumour type.
    """

    def __init__(self, in_channels=1, n_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(in_channels, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def build_model(name, n_classes=3):
    """Create a model from its name, as used on the command line."""
    if name == "small_cnn":
        return SmallCNN(in_channels=1, n_classes=n_classes)
    raise ValueError(f"unknown model: {name}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_models.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```powershell
git add src/models.py tests/test_models.py
git commit -m "feat: small CNN trained from scratch" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 9: Evaluation, training loop and class weights

**Concept:** the training loop (guess → loss → backward → step), what "loss" and "gradient" mean intuitively, and why class weights stop the model from favouring glioma.

**Files:**
- Create: `src/evaluate.py`, `src/train.py`
- Test: `tests/test_train.py`

Hand check for class weights: labels `[0,0,0,1]`, 2 classes → counts `[3,1]` → weights `4/(2×3) = 0.6667` and `4/(2×1) = 2.0`.

The "can it learn" test gives each class a bright square in a different position, on 8 tiny images. A working model plus loop must drive the loss well down. A broken loop, where for example the optimiser never steps, leaves the loss flat.

- [ ] **Step 1: Write the failing tests**

`tests/test_train.py`:
```python
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import evaluate_model
from src.models import SmallCNN
from src.train import class_weights, set_seed, train_one_epoch


def tiny_learnable_batch():
    """8 images, 32x32: each class has a bright square in its own row band."""
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    images = torch.rand(8, 1, 32, 32) * 0.1
    for i, label in enumerate(labels.tolist()):
        top = 2 + 10 * label
        images[i, 0, top:top + 8, 12:20] = 1.0
    return images, labels


def test_class_weights_balance_rare_classes():
    weights = class_weights([0, 0, 0, 1], n_classes=2)

    assert weights.tolist() == pytest.approx([4 / 6, 2.0])


def test_class_weights_reject_a_missing_class():
    with pytest.raises(ValueError, match="at least one"):
        class_weights([0, 0, 1], n_classes=3)


def test_model_can_memorise_a_tiny_batch():
    set_seed(0)
    images, labels = tiny_learnable_batch()
    loader = DataLoader(TensorDataset(images, labels), batch_size=8)
    model = SmallCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    first = train_one_epoch(model, loader, loss_fn, optimizer)["loss"]
    for _ in range(59):
        last = train_one_epoch(model, loader, loss_fn, optimizer)["loss"]

    assert last < first * 0.5


def test_evaluate_model_reports_predictions_and_confidences():
    set_seed(0)
    images, labels = tiny_learnable_batch()
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    result = evaluate_model(SmallCNN(), loader, nn.CrossEntropyLoss())

    assert result["y_true"] == labels.tolist()
    assert len(result["y_pred"]) == 8
    assert all(0.0 <= c <= 1.0 for c in result["confidences"])
    assert 0.0 <= result["accuracy"] <= 1.0
    assert result["loss"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.evaluate'`

- [ ] **Step 3: Write `src/evaluate.py`**

```python
"""Scoring a model on a set of scans, without learning from them."""

import torch

from src.metrics import accuracy


@torch.no_grad()
def evaluate_model(model, loader, loss_fn):
    """Run the model over every batch and collect its answers.

    Returns average loss, accuracy, and per-scan lists of true labels,
    predictions and confidences (the probability the model gave to the
    class it chose).
    """
    model.eval()
    total_loss = 0.0
    y_true, y_pred, confidences = [], [], []
    for images, labels in loader:
        logits = model(images)
        total_loss += loss_fn(logits, labels).item() * len(labels)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted = probabilities.max(dim=1)
        y_true += labels.tolist()
        y_pred += predicted.tolist()
        confidences += confidence.tolist()
    return {
        "loss": total_loss / len(y_true),
        "accuracy": accuracy(y_true, y_pred),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidences": confidences,
    }
```

- [ ] **Step 4: Write `src/train.py` (building blocks only; Task 10 adds the experiment runner)**

```python
"""Train a model and record the experiment."""

import random

import numpy as np
import torch


def set_seed(seed):
    """Fix every source of randomness so a run can be repeated exactly."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def class_weights(labels, n_classes):
    """Loss weights that make rare tumour types count as much as common ones.

    weight = total / (n_classes * count). A perfectly balanced dataset gets
    1.0 for every class; a class half as common gets double weight.
    """
    counts = np.bincount(np.asarray(labels), minlength=n_classes)
    if (counts == 0).any():
        raise ValueError(f"every class needs at least one example, got counts {counts.tolist()}")
    return torch.tensor(len(labels) / (n_classes * counts), dtype=torch.float32)


def train_one_epoch(model, loader, loss_fn, optimizer):
    """One full pass over the training data. For every batch:

    1. guess:   the model scores each tumour type
    2. measure: the loss says how wrong those scores were
    3. blame:   backward() works out how each weight contributed to the error
    4. adjust:  the optimiser nudges every weight to reduce the error
    """
    model.train()
    total_loss, correct, seen = 0.0, 0, 0
    for images, labels in loader:
        logits = model(images)
        loss = loss_fn(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += len(labels)
    return {"loss": total_loss / seen, "accuracy": correct / seen}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_train.py -v`
Expected: 4 passed.

If `test_model_can_memorise_a_tiny_batch` fails, print `first` and `last`:
- **Loss barely moved:** the loop is broken. Check `optimizer.step()` and `zero_grad()`.
- **Loss moved but not below half:** don't just raise the step count or loosen the threshold. Find out why first.

- [ ] **Step 6: Commit**

```powershell
git add src/evaluate.py src/train.py tests/test_train.py
git commit -m "feat: training loop, evaluation and class weights" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 10: Plots and the experiment runner

**Concept:** epochs, the validation set as a practice exam, saving the best checkpoint, and how the train-vs-validation curves reveal overfitting.

**Files:**
- Create: `src/plots.py`
- Modify: `src/train.py` (replace imports; append runner + CLI)
- Test: `tests/test_plots.py`, `tests/test_train.py` (append)

- [ ] **Step 1: Write the failing tests**

`tests/test_plots.py`:
```python
from src.plots import plot_confusion_matrix, plot_history


def test_plot_history_writes_png(tmp_path):
    history = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "train_accuracy": 0.4, "val_accuracy": 0.35},
        {"epoch": 2, "train_loss": 0.7, "val_loss": 0.9, "train_accuracy": 0.6, "val_accuracy": 0.5},
    ]
    path = tmp_path / "curves.png"

    plot_history(history, path)

    assert path.read_bytes()[:4] == b"\x89PNG"


def test_plot_confusion_matrix_writes_png(tmp_path):
    path = tmp_path / "cm.png"

    plot_confusion_matrix([[5, 1, 0], [2, 7, 1], [0, 0, 4]], ["a", "b", "c"], path)

    assert path.read_bytes()[:4] == b"\x89PNG"
```

Append to `tests/test_train.py`:
```python
import json

import numpy as np

from src.data import save_splits, split_patients
from src.train import run_experiment


def write_tiny_dataset(folder):
    """12 patients (4 per class), 3 slices each, 32x32 images with a class-specific square."""
    rng = np.random.default_rng(0)
    images, labels, patient_ids = [], [], []
    for label in range(3):
        for patient in range(4):
            for _ in range(3):
                image = (rng.random((32, 32)) * 30).astype(np.uint8)
                top = 2 + 10 * label
                image[top:top + 8, 12:20] = 255
                images.append(image)
                labels.append(label)
                patient_ids.append(f"{label}{patient}")
    dataset_path = folder / "tiny.npz"
    np.savez_compressed(
        dataset_path,
        images=np.stack(images),
        masks=np.zeros((len(images), 32, 32), dtype=np.uint8),
        labels=np.array(labels, dtype=np.int64),
        patient_ids=np.array(patient_ids),
    )
    splits_path = folder / "splits.csv"
    save_splits(split_patients(patient_ids, labels, seed=42), splits_path)
    return dataset_path, splits_path


def test_run_experiment_writes_a_complete_record(tmp_path):
    dataset_path, splits_path = write_tiny_dataset(tmp_path)

    metrics = run_experiment(
        name="tiny", model_name="small_cnn", epochs=2, batch_size=4, lr=1e-3, seed=42,
        dataset_path=dataset_path, splits_path=splits_path, experiments_dir=tmp_path / "experiments",
    )

    folder = tmp_path / "experiments" / "tiny"
    for filename in ["config.json", "history.csv", "metrics.json", "model.pt", "curves.png", "confusion_matrix.png"]:
        assert (folder / filename).exists(), filename
    assert metrics["split"] == "validation"
    assert metrics["n_scans"] == 9  # 1 validation patient per class x 3 slices
    assert set(metrics["per_class"]) == {"meningioma", "glioma", "pituitary"}
    assert json.loads((folder / "config.json").read_text())["epochs"] == 2
    assert len((folder / "history.csv").read_text().strip().splitlines()) == 3  # header + 2 epochs
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_plots.py tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.plots'`

- [ ] **Step 3: Write `src/plots.py`**

```python
"""Charts saved alongside each experiment."""

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt


def plot_history(history, path):
    """Loss and accuracy per epoch, training vs validation, side by side.

    If the training line keeps improving while validation flattens or gets
    worse, the model is memorising the training scans: overfitting.
    """
    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_accuracy) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, [row["train_loss"] for row in history], marker="o", label="training")
    ax_loss.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    ax_loss.set_title("Loss (lower is better)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    ax_accuracy.plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="training")
    ax_accuracy.plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="validation")
    ax_accuracy.set_title("Accuracy (higher is better)")
    ax_accuracy.set_xlabel("Epoch")
    ax_accuracy.set_ylim(0, 1)
    ax_accuracy.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(matrix, class_names, path):
    """Grid of true type (rows) vs predicted type (columns), with counts."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n), labels=class_names)
    ax.set_yticks(range(n), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    largest = max(max(row) for row in matrix) or 1
    for row in range(n):
        for col in range(n):
            value = matrix[row][col]
            colour = "white" if value > largest / 2 else "black"
            ax.text(col, row, value, ha="center", va="center", color=colour)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: Extend `src/train.py`**

Replace the module docstring and import block at the top of `src/train.py` with:
```python
"""Train a model and record the experiment.

Usage:
    .\\.venv\\Scripts\\python.exe -m src.train --name 01-small-cnn --model small_cnn --epochs 15
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data import CLASS_NAMES, DATASET_PATH, SPLITS_PATH, MRIDataset, indices_for_split, load_dataset, load_splits
from src.evaluate import evaluate_model
from src.metrics import confusion_matrix, precision_recall
from src.models import build_model
from src.plots import plot_confusion_matrix, plot_history

EXPERIMENTS_DIR = Path(__file__).resolve().parent.parent / "experiments"
```

Append to the bottom of `src/train.py`:
```python
def write_history(history, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def run_experiment(name, model_name, epochs, batch_size, lr, seed, subset=0,
                   dataset_path=DATASET_PATH, splits_path=SPLITS_PATH, experiments_dir=EXPERIMENTS_DIR):
    """Train on the training patients, pick the best epoch on validation, save the record.

    subset: if above 0, use only this many training and validation scans
            (for a quick check that everything runs).
    """
    set_seed(seed)
    out_dir = Path(experiments_dir) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    config = {"name": name, "model": model_name, "epochs": epochs, "batch_size": batch_size,
              "lr": lr, "seed": seed, "subset": subset}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    data = load_dataset(dataset_path)
    assignment = load_splits(splits_path)
    train_idx = indices_for_split(data["patient_ids"], assignment, "train")
    val_idx = indices_for_split(data["patient_ids"], assignment, "validation")
    if subset:
        rng = np.random.default_rng(seed)
        train_idx = np.sort(rng.permutation(train_idx)[:subset])
        val_idx = np.sort(rng.permutation(val_idx)[:subset])

    train_loader = DataLoader(
        MRIDataset(data["images"], data["labels"], train_idx),
        batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(MRIDataset(data["images"], data["labels"], val_idx), batch_size=batch_size)

    model = build_model(model_name, n_classes=len(CLASS_NAMES))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(data["labels"][train_idx], len(CLASS_NAMES)))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training {model_name} on {len(train_idx)} scans, validating on {len(val_idx)}")
    history, best_accuracy, best_epoch = [], -1.0, 0
    for epoch in range(1, epochs + 1):
        started = time.time()
        train_scores = train_one_epoch(model, train_loader, loss_fn, optimizer)
        val_scores = evaluate_model(model, val_loader, loss_fn)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_scores["loss"], 4),
            "train_accuracy": round(train_scores["accuracy"], 4),
            "val_loss": round(val_scores["loss"], 4),
            "val_accuracy": round(val_scores["accuracy"], 4),
        })
        write_history(history, out_dir / "history.csv")

        improved = val_scores["accuracy"] > best_accuracy
        if improved:
            best_accuracy, best_epoch = val_scores["accuracy"], epoch
            torch.save(model.state_dict(), out_dir / "model.pt")
        print(f"epoch {epoch:>2}/{epochs}  "
              f"train loss {train_scores['loss']:.3f} acc {train_scores['accuracy']:.1%}  |  "
              f"val loss {val_scores['loss']:.3f} acc {val_scores['accuracy']:.1%}  "
              f"({time.time() - started:.0f}s){'  <- best so far' if improved else ''}")

    model.load_state_dict(torch.load(out_dir / "model.pt", weights_only=True))
    final = evaluate_model(model, val_loader, loss_fn)
    matrix = confusion_matrix(final["y_true"], final["y_pred"], len(CLASS_NAMES))
    per_class = precision_recall(matrix)
    metrics = {
        "split": "validation",
        "best_epoch": best_epoch,
        "n_scans": len(final["y_true"]),
        "accuracy": round(final["accuracy"], 4),
        "loss": round(final["loss"], 4),
        "per_class": {name: {k: round(v, 4) for k, v in per_class[i].items()}
                      for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": matrix,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_history(history, out_dir / "curves.png")
    plot_confusion_matrix(matrix, CLASS_NAMES, out_dir / "confusion_matrix.png")

    print(f"\nBest epoch {best_epoch}: validation accuracy {final['accuracy']:.1%}")
    for class_name, scores in metrics["per_class"].items():
        print(f"  {class_name:<11} precision {scores['precision']:.1%}  recall {scores['recall']:.1%}")
    print(f"Saved to {out_dir}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a brain tumour classifier.")
    parser.add_argument("--name", required=True, help="experiment folder name, e.g. 01-small-cnn")
    parser.add_argument("--model", default="small_cnn")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate: how big each nudge is")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--subset", type=int, default=0, help="use only this many scans (quick check)")
    args = parser.parse_args()
    run_experiment(args.name, args.model, args.epochs, args.batch_size, args.lr, args.seed, args.subset)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the full test suite**

Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: 33 passed (6 metrics + 12 data + 4 download + 4 models + 5 train + 2 plots), none skipped.

- [ ] **Step 6: Commit**

```powershell
git add src/plots.py src/train.py tests/test_plots.py tests/test_train.py
git commit -m "feat: experiment runner with curves, confusion matrix and metrics record" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 11: Run experiment 1 (Robbi runs this)

**Concept:** reading the per-epoch printout live. What it looks like when the model is learning, plateauing or overfitting.

- [ ] **Step 1: Quick check that the real pipeline runs end to end**

Run: `.\.venv\Scripts\python.exe -m src.train --name smoke --epochs 1 --subset 64`
Expected: one epoch line, then scores and `Saved to ...experiments\smoke`. It takes under a minute. (`experiments/smoke/` is gitignored.)

- [ ] **Step 2: Robbi runs experiment 1**

Run: `.\.venv\Scripts\python.exe -m src.train --name 01-small-cnn --model small_cnn --epochs 15`
Expected: 15 epoch lines, each showing its time in seconds. The total is likely 20–60 minutes on this laptop. Note the real per-epoch time for the README.

- [ ] **Step 3: Commit the experiment record**

```powershell
git add experiments/01-small-cnn
git commit -m "exp: 01 small CNN from scratch (validation results)" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```
Expected: `config.json`, `history.csv`, `metrics.json`, `curves.png` and `confusion_matrix.png` are committed; `model.pt` is excluded.

- [ ] **Step 4: ⏸ PAUSE: review results together with Robbi**

Open `experiments\01-small-cnn\curves.png`, `confusion_matrix.png` and `metrics.json`. Discuss:
- **Learning:** did validation accuracy climb, and at which epoch did it peak?
- **Overfitting:** is there a gap between the training and validation lines?
- **Weakest type:** which tumour type has the lowest recall, and which type is it confused with?
- **Hypothesis for experiment 2:** what might a pretrained model change?

Do not continue until Robbi has seen and discussed the results.

---

### Task 12: Learning notes, README v1 and GitHub

**Concept:** writing results honestly. Validation scores are not final test scores, and a CV project is judged on clarity.

**Files:**
- Create: `docs/learning-notes.md`, `README.md`

- [ ] **Step 1: Write `docs/learning-notes.md`**

Draft with this content, then ask Robbi to reword anything that doesn't sound like them:

```markdown
# Learning notes

Plain-English explanations of the ideas in this project, written for interview prep.

## Images are numbers
A greyscale MRI slice is a grid of brightness values. Resized to 224×224, that is 50,176 numbers. The model only ever sees these numbers.

## Why rescale brightness per slice
MRI intensity has no fixed unit, so the same tissue can have different values on different scans. Stretching each slice so its darkest pixel is 0 and brightest is 1 puts every scan on the same footing.

## The patient-level split
Slices from the same patient look alike. If one slice is in training and its neighbour is in testing, the model is partly tested on something it has already seen, and the score is inflated. So whole patients go to exactly one of: training (70%), validation (15%) or test (15%).
- **Training:** what the model learns from.
- **Validation:** a practice exam used to make decisions, such as which epoch was best.
- **Test:** the final exam, used once at the end, so our decisions can't be tuned to it.

## Convolution
A small 3×3 filter slides across the image and responds strongly wherever its pattern appears. Early layers find edges; later layers combine them into shapes and textures. Pooling halves the image size so later filters see a wider area.

## The training loop
For each batch of scans:
1. **Guess:** the model scores each tumour type.
2. **Measure:** the loss is one number saying how wrong the guess was.
3. **Blame:** backpropagation works out how much each of the model's weights contributed to the error.
4. **Adjust:** the optimiser nudges every weight slightly to reduce the error. The learning rate sets how big the nudge is.

One pass over all training scans is an **epoch**.

## Overfitting
The model memorises the training scans instead of learning general patterns. You see it when training accuracy keeps rising but validation accuracy stalls or falls. That's why we keep the checkpoint from the best validation epoch, not the last one.

## Class imbalance and class weights
There are about twice as many glioma scans as meningioma. Without correction, the model can score well by leaning towards glioma. Class weights make each mistake on a rarer type count for more.

## Precision and recall
- **Recall** for glioma: of all real gliomas, how many did the model catch?
- **Precision** for glioma: of everything the model called glioma, how many really were?

In medicine, missing a tumour type (low recall) usually costs more than a false alarm, so accuracy alone is not enough.

## Confusion matrix
A table of true type (rows) vs predicted type (columns). The diagonal is correct answers; everything off it shows exactly which types get mixed up.
```

- [ ] **Step 2: Write `README.md` v1**

Use this structure. Every number comes from `experiments/01-small-cnn/metrics.json` and `history.csv` from Task 11. Copy the values exactly; don't round differently or invent figures.

```markdown
# Brain Tumour MRI Classifier

Classifying brain tumour type (glioma, meningioma, pituitary) from MRI slices with deep learning, built step by step to understand how the model learns, and where its results can mislead.

> **Status:** in progress. Experiment 1 (baseline) complete; pretrained models, heatmaps and a live demo are next.
> **Educational project, not a medical device, not for diagnosis.**

## Why this project
During my internship at Siemens working on MRI, I learned that AI is being used both to shorten scan times and to help clinicians detect cancer. This project explores that from the ground up: training a model on real MRI data, and asking not just "how accurate is it?" but "can that number be trusted?"

## Data
- **Source:** 3,064 contrast-enhanced T1-weighted MRI slices from 233 patients (Cheng et al.). Glioma: 1,426 · Meningioma: 708 · Pituitary: 930.
- **Split by patient, not by slice:** 70% training / 15% validation / 15% test (stratified by tumour type). No patient appears in more than one set; an automated test enforces this. Splitting by slice would let the model see near-identical neighbouring slices of a test patient during training.
- **Class imbalance:** there are twice as many gliomas as meningiomas. The loss is weighted so rarer types aren't ignored, and results are reported per type.

## Experiment 1: small CNN from scratch
A 4-block convolutional network (≈100k parameters) trained only on these scans for 15 epochs on a laptop CPU (<seconds per epoch, noted from the console in Task 11> per epoch).

![Training curves](experiments/01-small-cnn/curves.png)

**Validation results** (best epoch <best_epoch>, <n_scans> scans; the test set is untouched until the final model is chosen):

| Tumour type | Precision | Recall |
|---|---|---|
| Glioma | <from metrics.json> | <from metrics.json> |
| Meningioma | <from metrics.json> | <from metrics.json> |
| Pituitary | <from metrics.json> | <from metrics.json> |

Overall validation accuracy: <from metrics.json>.

![Confusion matrix](experiments/01-small-cnn/confusion_matrix.png)

**What this shows:** <2–4 sentences agreed with Robbi in the Task 11 review: where it plateaued, any overfitting gap, which type is weakest and what it's confused with.>

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
```

The angle-bracket items above are **data to fill from Task 11 outputs**, not open design questions. Fill every one before committing, and replace "≈100k parameters" with the exact count:
```powershell
.\.venv\Scripts\python.exe -c "from src.models import SmallCNN; print(sum(p.numel() for p in SmallCNN().parameters()))"
```
Then check nothing was missed: `Select-String -Path README.md -Pattern '<'`. Expected: no matches.

- [ ] **Step 3: Commit**

```powershell
git add README.md docs/learning-notes.md
git commit -m "docs: README v1 with experiment 1 results and learning notes" -m "Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Create the GitHub repo (ask Robbi first)**

This publishes the repo publicly. Ask Robbi to confirm the name `brain-tumour-mri` and public visibility, then run:
```powershell
gh repo create brain-tumour-mri --public --source . --push --description "Brain tumour type classification from MRI with PyTorch: patient-level splits, honest evaluation, explainability"
```
Expected: prints `https://github.com/RVotta4/brain-tumour-mri`. Open it and check the README renders with both images.

---

## Done when

- `.\.venv\Scripts\python.exe -m pytest` passes in full.
- `data/cheng_224.npz` and `data/splits.csv` exist; the summary matched 3,064 slices / 233 patients / 708-1426-930.
- `experiments/01-small-cnn/` holds config, history, metrics and both charts, and Robbi has reviewed them.
- README v1 and learning notes are committed and pushed to GitHub.
- **Next:** write the Stage 2 plan (pretrained ResNet-18).
