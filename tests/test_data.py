import numpy as np

from src.data import IMAGE_SIZE, preprocess_image, preprocess_mask


def test_preprocess_image_resizes_to_224_uint8():
    raw = np.random.default_rng(0).integers(0, 3000, size=(512, 512)).astype(np.int16)

    out = preprocess_image(raw)

    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE) == (224, 224)
    assert out.dtype == np.uint8


def test_preprocess_image_stretches_to_full_brightness_range():
    raw = np.zeros((512, 512), dtype=np.int16)
    raw[:256, :] = 1000
    raw[256:, :] = 3000

    out = preprocess_image(raw)

    assert out.min() == 0
    assert out.max() == 255


def test_preprocess_image_handles_blank_slice_without_dividing_by_zero():
    raw = np.full((512, 512), 7, dtype=np.int16)

    out = preprocess_image(raw)

    assert (out == 0).all()


def test_preprocess_mask_stays_binary_and_keeps_tumour():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1

    out = preprocess_mask(mask)

    assert out.shape == (224, 224)
    assert set(np.unique(out)) == {0, 1}
    assert out.sum() > 0


import pytest

from src.data import indices_for_split, load_splits, save_splits, split_patients


def make_patients(per_class=20, slices_per_patient=3):
    """Synthetic slice-level patient IDs and labels: every patient has one label."""
    patient_ids, labels = [], []
    for label in range(3):
        for p in range(per_class):
            for _ in range(slices_per_patient):
                patient_ids.append(f"{label}{p:03d}")
                labels.append(label)
    return patient_ids, labels


def test_every_patient_is_assigned_exactly_once():
    patient_ids, labels = make_patients()

    assignment = split_patients(patient_ids, labels, seed=42)

    assert set(assignment) == set(patient_ids)
    assert set(assignment.values()) == {"train", "validation", "test"}


def test_no_slice_indices_shared_between_splits():
    patient_ids, labels = make_patients()
    assignment = split_patients(patient_ids, labels, seed=42)

    train = set(indices_for_split(patient_ids, assignment, "train"))
    val = set(indices_for_split(patient_ids, assignment, "validation"))
    test = set(indices_for_split(patient_ids, assignment, "test"))

    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train) + len(val) + len(test) == len(patient_ids)


def test_split_is_stratified_by_tumour_type():
    patient_ids, labels = make_patients(per_class=20)
    assignment = split_patients(patient_ids, labels, seed=42)
    label_of = dict(zip(patient_ids, labels))

    for split, expected in [("train", 14), ("validation", 3), ("test", 3)]:
        for label in range(3):
            count = sum(1 for pid, s in assignment.items() if s == split and label_of[pid] == label)
            assert count == expected, (split, label, count)


def test_same_seed_gives_same_split_and_different_seed_differs():
    patient_ids, labels = make_patients()

    first = split_patients(patient_ids, labels, seed=42)
    again = split_patients(patient_ids, labels, seed=42)
    other = split_patients(patient_ids, labels, seed=7)

    assert first == again
    assert first != other


def test_patient_with_two_tumour_types_is_rejected():
    with pytest.raises(ValueError):
        split_patients(["A", "A"], [0, 1], seed=42)


def test_splits_survive_saving_and_loading(tmp_path):
    patient_ids, labels = make_patients(per_class=4)
    assignment = split_patients(patient_ids, labels, seed=42)
    path = tmp_path / "splits.csv"

    save_splits(assignment, path)

    assert load_splits(path) == assignment
