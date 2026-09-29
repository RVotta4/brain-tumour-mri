import json

import numpy as np
import pytest
import torch

from src.explain import mask_share, patient_errors, pointing_chance, pointing_hit, run_explanation, summarise
from src.models import ResNet18Grey
from src.score import score_experiment
from tests.test_train import write_tiny_dataset


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


def tiny_scored_resnet(tmp_path):
    """A tiny dataset with real tumour masks, and an untrained ResNet already scored on its test set."""
    dataset_path, splits_path = write_tiny_dataset(tmp_path)
    with np.load(dataset_path) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["masks"] = (arrays["images"] == 255).astype(np.uint8)  # the bright square is the "tumour"
    np.savez_compressed(dataset_path, **arrays)

    experiments_dir = tmp_path / "experiments"
    folder = experiments_dir / "final"
    folder.mkdir(parents=True)
    torch.manual_seed(0)
    torch.save(ResNet18Grey(weights=None).state_dict(), folder / "model.pt")
    (folder / "config.json").write_text(json.dumps({"model": "resnet18", "subset": 0}))
    (folder / "metrics.json").write_text(json.dumps({"best_epoch": 1, "accuracy": 0.0}))
    paths = {"dataset_path": dataset_path, "splits_path": splits_path, "experiments_dir": experiments_dir}
    score_experiment("final", "test", **paths)
    return paths


def test_explanation_refuses_before_the_test_score(tmp_path):
    with pytest.raises(RuntimeError, match="test_metrics"):
        run_explanation("final", experiments_dir=tmp_path)


def test_explanation_writes_every_output(tmp_path):
    paths = tiny_scored_resnet(tmp_path)

    summary = run_explanation("final", **paths)

    out_dir = paths["experiments_dir"] / "final" / "explain"
    assert summary["reproduces_saved_test_score"] is True
    assert summary["overall"]["scans"] == 9
    assert len((out_dir / "test_scans.csv").read_text().strip().splitlines()) == 10  # header + 9 scans
    assert len((out_dir / "patient_errors.csv").read_text().strip().splitlines()) == 4  # header + 3 patients
    assert json.loads((out_dir / "summary.json").read_text())["overall"]["scans"] == 9
    assert (out_dir / "gradcam_examples.png").exists()
