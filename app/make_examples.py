"""Copy the demo's example scans out of the test set.

Usage:
    .\\.venv\\Scripts\\python.exe -m app.make_examples

Each example is a test scan saved exactly as the model saw it, with the
clinicians' tumour outline beside it. They were chosen by hand to show
behaviour the README already reports; they illustrate, they don't measure.
"""

import csv

import numpy as np
from PIL import Image

from app.demo import EXAMPLES_DIR
from src.data import CLASS_NAMES, SPLITS_PATH, indices_for_split, load_dataset, load_splits

# (index in cheng_224.npz, file name, caption, note shown under the prediction)
CHOSEN = [
    (2040, "glioma_correct", "Glioma — correct",
     "The heatmap lands on the tumour."),
    (406, "meningioma_correct", "Meningioma — correct",
     "The heatmap lands on the tumour."),
    (999, "pituitary_correct", "Pituitary — correct, looking elsewhere",
     "Right answer, but the heatmap misses the tumour. Across the test set, the model's attention landed on "
     "pituitary tumours less often than chance would, which suggests it relies on where the slice sits in the "
     "head rather than on the tumour itself."),
    (560, "meningioma_called_glioma", "Meningioma — called glioma",
     "A confident mistake: all four test slices of this patient were called glioma, at 0.97–1.00 confidence."),
    (827, "glioma_called_meningioma", "Glioma — called meningioma",
     "From the patient with the most mistakes: 8 of the test set's 39."),
    (1759, "pituitary_called_meningioma", "Pituitary — called meningioma",
     "One of six wrong slices out of this patient's seven."),
]

FIELDS = ["name", "index", "patient_id", "true", "caption", "note"]


def main():
    data = load_dataset()
    test = set(indices_for_split(data["patient_ids"], load_splits(SPLITS_PATH), "test").tolist())
    EXAMPLES_DIR.mkdir(exist_ok=True)
    rows = []
    for index, name, caption, note in CHOSEN:
        if index not in test:
            raise SystemExit(f"scan {index} is not in the test split")
        Image.fromarray(data["images"][index]).save(EXAMPLES_DIR / f"{name}.png")
        Image.fromarray((data["masks"][index] * 255).astype(np.uint8)).save(EXAMPLES_DIR / f"{name}_mask.png")
        rows.append({"name": name, "index": index, "patient_id": str(data["patient_ids"][index]),
                     "true": CLASS_NAMES[int(data["labels"][index])], "caption": caption, "note": note})
    with open(EXAMPLES_DIR / "examples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} examples to {EXAMPLES_DIR}")


if __name__ == "__main__":
    main()
