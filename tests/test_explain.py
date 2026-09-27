import numpy as np
import pytest
import torch

from src.explain import (class_scores, feature_maps, grad_cam, mask_share, patient_errors, pointing_chance,
                         pointing_hit, summarise)
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
