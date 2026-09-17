"""Download the Cheng et al. brain tumour MRI dataset and prepare it.

Usage:
    .\\.venv\\Scripts\\python.exe -m src.download

Dataset: Cheng, Jun (2017). brain tumor dataset. figshare.
https://doi.org/10.6084/m9.figshare.1512427 (CC BY 4.0)
"""

import io
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import h5py
import numpy as np

from src.data import (
    CLASS_NAMES,
    DATA_DIR,
    DATASET_PATH,
    SPLITS_PATH,
    preprocess_image,
    preprocess_mask,
    save_splits,
    split_patients,
)

ZIP_URLS = {
    "brainTumorDataPublic_1-766.zip": "https://ndownloader.figshare.com/files/3381290",
    "brainTumorDataPublic_767-1532.zip": "https://ndownloader.figshare.com/files/3381296",
    "brainTumorDataPublic_1533-2298.zip": "https://ndownloader.figshare.com/files/3381293",
    "brainTumorDataPublic_2299-3064.zip": "https://ndownloader.figshare.com/files/3381302",
}
RAW_DIR = DATA_DIR / "raw"
REQUIRED_FREE_GB = 3.0


def check_disk_space(path, required_gb=REQUIRED_FREE_GB, disk_usage=shutil.disk_usage):
    """Stop early with a clear message rather than filling the drive."""
    free_gb = disk_usage(path).free / 1024**3
    if free_gb < required_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free on this drive; need at least {required_gb:.1f} GB. "
            "Free up some space and run again."
        )
    return free_gb


def read_mat(raw_bytes):
    """Read one slice from a MATLAB v7.3 .mat file (HDF5 inside).

    Returns (image, mask, class_index, patient_id).
    - The dataset labels are 1=meningioma, 2=glioma, 3=pituitary; we return
      0, 1, 2 so they index CLASS_NAMES.
    - MATLAB stores arrays column-first, so h5py returns them transposed;
      .T puts rows and columns back the right way round.
    - The patient ID is stored as a column of character codes.
    """
    with h5py.File(io.BytesIO(raw_bytes), "r") as f:
        data = f["cjdata"]
        image = np.array(data["image"]).T
        mask = np.array(data["tumorMask"]).T
        class_index = int(np.array(data["label"]).ravel()[0]) - 1
        patient_id = "".join(chr(code) for code in np.array(data["PID"]).ravel())
    return image, mask, class_index, patient_id


def convert_zip(zip_path):
    """Yield (image, mask, class_index, patient_id) for each slice, resized to 224x224.

    Slices are read in numeric file order (1.mat, 2.mat, 10.mat) so the
    prepared dataset is always in the same order.
    """
    with zipfile.ZipFile(zip_path) as archive:
        names = sorted(
            (name for name in archive.namelist() if name.endswith(".mat")),
            key=lambda name: int(Path(name).stem),
        )
        for name in names:
            image, mask, class_index, patient_id = read_mat(archive.read(name))
            yield preprocess_image(image), preprocess_mask(mask), class_index, patient_id


def download_file(url, destination):
    """Download url to destination, reusing a previous complete download."""
    if destination.exists():
        print(f"  already downloaded: {destination.name}")
        return
    partial = destination.with_suffix(".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1024 * 1024)
    partial.rename(destination)


def build_arrays(zip_paths):
    """Convert every zip and stack the results into arrays."""
    images, masks, labels, patient_ids = [], [], [], []
    for zip_path in zip_paths:
        print(f"  converting {zip_path.name}")
        for image, mask, class_index, patient_id in convert_zip(zip_path):
            images.append(image)
            masks.append(mask)
            labels.append(class_index)
            patient_ids.append(patient_id)
    return {
        "images": np.stack(images),
        "masks": np.stack(masks),
        "labels": np.array(labels, dtype=np.int64),
        "patient_ids": np.array(patient_ids),
    }


def print_summary(arrays, assignment):
    labels, patient_ids = arrays["labels"], arrays["patient_ids"]
    print(f"\nSlices: {len(labels)}   Patients: {len(set(patient_ids))}")
    for index, name in enumerate(CLASS_NAMES):
        print(f"  {name:<11} {int((labels == index).sum()):>5} slices")
    print("\nSplit (patients / slices):")
    slices_per_split = Counter(assignment[str(pid)] for pid in patient_ids)
    patients_per_split = Counter(assignment.values())
    for split in ["train", "validation", "test"]:
        print(f"  {split:<11} {patients_per_split[split]:>4} / {slices_per_split[split]:>5}")


def main():
    if DATASET_PATH.exists():
        print(f"{DATASET_PATH} already exists. Delete the data folder to rebuild it.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    free_gb = check_disk_space(DATA_DIR)
    print(f"{free_gb:.1f} GB free - OK")

    zip_paths = []
    for name, url in ZIP_URLS.items():
        print(f"Downloading {name} (about 220 MB)...")
        download_file(url, RAW_DIR / name)
        zip_paths.append(RAW_DIR / name)

    print("Converting slices to 224x224...")
    arrays = build_arrays(zip_paths)
    np.savez_compressed(DATASET_PATH, **arrays)

    assignment = split_patients(arrays["patient_ids"], arrays["labels"])
    save_splits(assignment, SPLITS_PATH)

    shutil.rmtree(RAW_DIR)
    print(f"Saved {DATASET_PATH.name} and {SPLITS_PATH.name}; deleted raw downloads.")
    print_summary(arrays, assignment)


if __name__ == "__main__":
    main()
