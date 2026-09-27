"""Where the final model looks: Grad-CAM heatmaps measured against the tumour outlines.

Usage:
    .\\.venv\\Scripts\\python.exe -m src.explain --name 03-resnet18-finetuned

Runs on the test scans, and only once the test score exists. It never
changes the model: whatever it finds is reported, not fixed.
"""

import numpy as np
import torch
from torch.nn import functional as F

from src.models import ResNet18Grey

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
