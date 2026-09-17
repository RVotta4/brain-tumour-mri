import io
import zipfile
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from src.download import check_disk_space, convert_zip, read_mat


def make_mat_bytes(image, mask, label, patient_id):
    """Build bytes shaped like a real dataset file (MATLAB v7.3 = HDF5).

    MATLAB writes arrays column-first, so the real files hold them transposed.
    """
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as f:
        group = f.create_group("cjdata")
        group["image"] = image.T.astype(np.int16)
        group["tumorMask"] = mask.T.astype(np.uint8)
        group["label"] = np.array([[float(label)]])
        group["PID"] = np.array([[ord(c)] for c in patient_id], dtype=np.uint16)
    return buffer.getvalue()


def gigabytes(n):
    return lambda path: SimpleNamespace(free=int(n * 1024**3))


def test_disk_check_refuses_when_space_is_low(tmp_path):
    with pytest.raises(RuntimeError, match="GB free"):
        check_disk_space(tmp_path, required_gb=3.0, disk_usage=gigabytes(1.5))


def test_disk_check_passes_when_space_is_enough(tmp_path):
    assert check_disk_space(tmp_path, required_gb=3.0, disk_usage=gigabytes(8)) == pytest.approx(8)


def test_read_mat_restores_orientation_label_and_patient_id():
    image = np.arange(24).reshape(4, 6)
    mask = np.zeros((4, 6))
    mask[1, 2] = 1

    out_image, out_mask, class_index, patient_id = read_mat(make_mat_bytes(image, mask, 2, "100360"))

    assert np.array_equal(out_image, image)
    assert np.array_equal(out_mask, mask)
    assert class_index == 1  # dataset label 2 = glioma = CLASS_NAMES[1]
    assert patient_id == "100360"


def test_convert_zip_reads_slices_in_numeric_order(tmp_path):
    zip_path = tmp_path / "part.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for name, label in [("10.mat", 3), ("2.mat", 2), ("1.mat", 1)]:
            image = np.random.default_rng(0).integers(0, 500, size=(64, 64))
            archive.writestr(name, make_mat_bytes(image, np.zeros((64, 64)), label, "123456"))

    slices = list(convert_zip(zip_path))

    assert [class_index for _, _, class_index, _ in slices] == [0, 1, 2]
    image, mask, _, patient_id = slices[0]
    assert image.shape == (224, 224) and image.dtype == np.uint8
    assert mask.shape == (224, 224)
    assert patient_id == "123456"
