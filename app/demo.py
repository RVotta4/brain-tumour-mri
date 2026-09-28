"""The live demo's logic, kept free of Gradio so pytest can check it.

app/app.py lays out the page; everything the page shows is computed here.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import colormaps
from PIL import Image

from src.data import preprocess_image

EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

HEAT_ALPHA = 0.45  # how strongly the heatmap tints the scan, as in the Stage 4 figures
OUTLINE_COLOUR = (0, 255, 0)  # lime, as in the Stage 4 figures
OUTLINE_WIDTH = 2  # pixels


@dataclass
class Example:
    """One of the demo's test scans, with what is known about it."""

    name: str
    index: int  # position in data/cheng_224.npz
    path: Path  # the PNG the page offers as a one-click example
    scan: np.ndarray  # (224, 224) uint8, exactly as stored in cheng_224.npz
    mask: np.ndarray  # (224, 224) uint8, 1 inside the clinicians' tumour outline
    patient_id: str
    true: str
    caption: str
    note: str


def read_png(path):
    with Image.open(path) as image:
        return np.asarray(image)


def load_examples(folder=EXAMPLES_DIR):
    """The example scans listed in folder/examples.csv."""
    folder = Path(folder)
    with open(folder / "examples.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        Example(
            name=row["name"],
            index=int(row["index"]),
            path=folder / f"{row['name']}.png",
            scan=read_png(folder / f"{row['name']}.png"),
            mask=(read_png(folder / f"{row['name']}_mask.png") > 127).astype(np.uint8),
            patient_id=row["patient_id"],
            true=row["true"],
            caption=row["caption"],
            note=row["note"],
        )
        for row in rows
    ]


def to_grey(image):
    """Any PIL image as a 2D float32 array of grey levels.

    16-bit and floating-point images keep their full range (preprocess_image
    stretches it to 0-255 later); everything else becomes ordinary 0-255 grey.
    """
    mode = "F" if image.mode in ("I", "I;16", "I;16B", "I;16L", "F") else "L"
    return np.asarray(image.convert(mode), dtype=np.float32)


def find_example(grey, examples):
    """The example whose scan is pixel-for-pixel identical to grey, or None."""
    for example in examples:
        if grey.shape == example.scan.shape and np.array_equal(grey, example.scan):
            return example
    return None


def scan_pixels(image, examples):
    """The 224x224 uint8 scan the model will see, and which example it is (or None).

    An example's stored pixels are used as they are: stretching them a second
    time would change them, and the model would no longer see the scan it was
    tested on. Anything else goes through the same preparation as training.
    """
    grey = to_grey(image)
    example = find_example(grey, examples)
    if example is not None:
        return example.scan, example
    return preprocess_image(grey), None


def outline(mask, width=OUTLINE_WIDTH):
    """The tumour's border: tumour pixels within `width` pixels of non-tumour.

    Each round of erosion peels one pixel off the tumour's edge (a pixel stays
    only if it and its four neighbours are all tumour); what was peeled off is
    the border.
    """
    inside = mask > 0
    core = inside.copy()
    for _ in range(width):
        padded = np.pad(core, 1, constant_values=False)
        core = (padded[1:-1, 1:-1] & padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    return inside & ~core


def overlay(pixels, heatmap, mask=None):
    """The scan in grey, tinted by the heatmap, with the tumour outline if one is known.

    pixels: (height, width) uint8 scan. heatmap: same size, values in [0, 1].
    Returns a (height, width, 3) uint8 colour image.
    """
    grey = np.repeat(pixels[..., None].astype(np.float32) / 255.0, 3, axis=2)
    heat = colormaps["jet"](heatmap)[..., :3]  # drop the alpha channel
    image = np.round(((1 - HEAT_ALPHA) * grey + HEAT_ALPHA * heat) * 255).astype(np.uint8)
    if mask is not None:
        image[outline(mask)] = OUTLINE_COLOUR
    return image
