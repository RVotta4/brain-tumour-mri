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
