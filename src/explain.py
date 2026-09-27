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
