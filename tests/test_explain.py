import numpy as np
import pytest
import torch

from src.explain import class_scores, feature_maps, grad_cam, mask_share, pointing_chance, pointing_hit
from src.models import ResNet18Grey, SmallCNN


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
