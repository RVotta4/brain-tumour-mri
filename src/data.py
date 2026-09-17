"""Loading, preparing and splitting the brain MRI scans."""

import csv
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

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
