"""Where the final model looks: Grad-CAM heatmaps measured against the tumour outlines.

Usage:
    .\\.venv\\Scripts\\python.exe -m src.explain --name 03-resnet18-finetuned

Runs on the test scans, and only once the test score exists. It never
changes the model: whatever it finds is reported, not fixed.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.data import CLASS_NAMES, DATASET_PATH, SPLITS_PATH, MRIDataset, indices_for_split, load_dataset, load_splits
from src.gradcam import grad_cam
from src.models import build_model
from src.plots import plot_gradcam_examples, plot_mistakes
from src.score import TEST_RESULTS
from src.train import EXPERIMENTS_DIR

TOLERANCE = 8  # pixels of leeway in the pointing game: a quarter of one 32-pixel heatmap cell
CONFIDENT = 0.9  # a wrong answer at or above this confidence is a "confident mistake"
EXAMPLES_PER_TYPE = 4


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


def patient_errors(rows):
    """Mistakes counted per patient, most mistakes first.

    Neighbouring slices of one patient look alike, so many missed slices can
    be one hard patient rather than many separate failures.
    """
    by_patient = {}
    for row in rows:
        by_patient.setdefault(row["patient_id"], []).append(row)
    table = []
    for patient_id, scans in by_patient.items():
        wrong = [scan["predicted"] for scan in scans if not scan["correct"]]
        table.append({
            "patient_id": patient_id,
            "true": scans[0]["true"],
            "slices": len(scans),
            "wrong": len(wrong),
            "most_common_wrong_prediction": Counter(wrong).most_common(1)[0][0] if wrong else "",
        })
    return sorted(table, key=lambda row: (-row["wrong"], row["patient_id"]))


def group_stats(rows):
    """Pointing-game hit rate and in-mask share for a group of scans, beside their luck baselines."""
    if not rows:
        return None
    share = float(np.mean([row["mask_share"] for row in rows]))
    area = float(np.mean([row["mask_area"] for row in rows]))
    return {
        "scans": len(rows),
        "pointing_hit_rate": round(float(np.mean([row["pointing_hit"] for row in rows])), 4),
        "pointing_chance": round(float(np.mean([row["pointing_chance"] for row in rows])), 4),
        "mean_mask_share": round(share, 4),
        "mean_mask_area": round(area, 4),
        "mask_share_times_luck": round(share / area, 1) if area > 0 else None,
    }


def summarise(rows):
    """The headline numbers: overall, correct versus wrong, and per true tumour type."""
    return {
        "overall": group_stats(rows),
        "correct": group_stats([row for row in rows if row["correct"]]),
        "wrong": group_stats([row for row in rows if not row["correct"]]),
        "by_true_type": {name: group_stats([row for row in rows if row["true"] == name]) for name in CLASS_NAMES},
        "confident_mistakes": sum(1 for row in rows if not row["correct"] and row["confidence"] >= CONFIDENT),
    }


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_explanation(name, dataset_path=DATASET_PATH, splits_path=SPLITS_PATH, experiments_dir=EXPERIMENTS_DIR,
                    seed=0):
    """Explain every test prediction of the scored final model and save the results.

    Refuses to run before the test score exists: this stage explains the
    final model's test answers; it never helps choose or change a model.
    Writes test_scans.csv, patient_errors.csv, summary.json and the figures
    into the experiment's explain/ folder.
    """
    folder = Path(experiments_dir) / name
    if not (folder / TEST_RESULTS).exists():
        raise RuntimeError(f"{name} has no {TEST_RESULTS}: only the final model is explained, "
                           "after its one test score.")
    saved = json.loads((folder / TEST_RESULTS).read_text())
    config = json.loads((folder / "config.json").read_text())
    model = build_model(config["model"], n_classes=len(CLASS_NAMES), pretrained=False)
    model.load_state_dict(torch.load(folder / "model.pt", weights_only=True))
    model.eval()

    data = load_dataset(dataset_path)
    test_idx = indices_for_split(data["patient_ids"], load_splits(splits_path), "test")
    dataset = MRIDataset(data["images"], data["labels"], test_idx)

    rows, heatmaps = [], []
    for position, index in enumerate(test_idx):
        image, label = dataset[position]
        heatmap, predicted, confidence = grad_cam(model, image)
        mask = data["masks"][index]
        rows.append({
            "index": int(index),
            "patient_id": str(data["patient_ids"][index]),
            "true": CLASS_NAMES[label],
            "predicted": CLASS_NAMES[predicted],
            "confidence": round(confidence, 4),
            "correct": predicted == label,
            "pointing_hit": pointing_hit(heatmap, mask),
            "pointing_chance": round(pointing_chance(mask), 5),
            "mask_share": round(mask_share(heatmap, mask), 5),
            "mask_area": round(float(mask.mean()), 5),
        })
        heatmaps.append(heatmap)

    accuracy = round(float(np.mean([row["correct"] for row in rows])), 4)
    summary = summarise(rows)
    summary["test_accuracy"] = accuracy
    summary["reproduces_saved_test_score"] = accuracy == saved["accuracy"]
    print(f"Test accuracy {accuracy:.1%} against saved {saved['accuracy']:.1%} -> "
          f"{'MATCH' if summary['reproduces_saved_test_score'] else 'MISMATCH'}")
    if not summary["reproduces_saved_test_score"]:
        raise RuntimeError("these predictions do not reproduce the saved test score, so they would not "
                           "explain it; nothing was written")

    out_dir = folder / "explain"
    out_dir.mkdir(exist_ok=True)
    write_csv(rows, out_dir / "test_scans.csv")
    write_csv(patient_errors(rows), out_dir / "patient_errors.csv")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    def panel(position):
        index = rows[position]["index"]
        return data["images"][index] / 255.0, data["masks"][index], heatmaps[position]

    # Examples are drawn at random with a fixed seed, never hand-picked.
    rng = np.random.default_rng(seed)
    examples = {}
    for class_name in CLASS_NAMES:
        correct = [p for p, row in enumerate(rows) if row["true"] == class_name and row["correct"]]
        if correct:
            chosen = sorted(rng.choice(correct, size=min(EXAMPLES_PER_TYPE, len(correct)), replace=False))
            examples[class_name] = [panel(p) for p in chosen]
    if examples:
        plot_gradcam_examples(examples, out_dir / "gradcam_examples.png")

    for class_name in CLASS_NAMES:
        items = []
        for p, row in enumerate(rows):
            if row["true"] != class_name or row["correct"]:
                continue
            image, mask, heatmap = panel(p)
            confident = row["confidence"] >= CONFIDENT
            label = (f"said {row['predicted']} ({row['confidence']:.3f})"
                     f"{' CONFIDENT' if confident else ''}\npatient {row['patient_id']}")
            items.append({"image": image, "mask": mask, "heatmap": heatmap, "label": label, "confident": confident})
        if items:
            plot_mistakes(items, f"True {class_name}: every test mistake", out_dir / f"mistakes_{class_name}.png")

    overall = summary["overall"]
    print(f"Pointing game: peak on the tumour in {overall['pointing_hit_rate']:.1%} of scans "
          f"(luck: {overall['pointing_chance']:.1%})")
    print(f"In-mask share: {overall['mean_mask_share']:.1%} of the heatmap inside the outline "
          f"(luck: {overall['mean_mask_area']:.1%})")
    print(f"Confident mistakes: {summary['confident_mistakes']}. Saved to {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Explain the final model's test predictions with Grad-CAM.")
    parser.add_argument("--name", required=True, help="experiment folder name, e.g. 03-resnet18-finetuned")
    args = parser.parse_args()
    run_explanation(args.name)


if __name__ == "__main__":
    main()
