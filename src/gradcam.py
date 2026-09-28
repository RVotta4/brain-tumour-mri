"""Grad-CAM for the final ResNet-18: where the evidence for its decision came from.

Kept apart from explain.py so the live demo can use it without the
training, scoring and plotting code.
"""

import torch
from torch.nn import functional as F

from src.models import ResNet18Grey


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
    confidence = float(torch.softmax(scores.detach(), dim=0)[predicted])

    # How much would the winning score rise if each map got stronger?
    (gradients,) = torch.autograd.grad(scores[predicted], maps)
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # one importance per map
    cam = torch.relu((weights * maps).sum(dim=1, keepdim=True))  # keep evidence for, drop evidence against
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    heatmap = cam.detach().numpy()
    peak = heatmap.max()
    return (heatmap / peak if peak > 0 else heatmap), predicted, confidence
