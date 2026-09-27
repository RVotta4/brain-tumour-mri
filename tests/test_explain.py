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
