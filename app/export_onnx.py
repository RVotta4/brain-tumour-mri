"""Export the final model for the browser demo, and check the export.

Usage:
    .\\.venv\\Scripts\\python.exe -m app.export_onnx

Writes experiments/03-resnet18-finetuned/model.onnx (uploaded to Hugging Face,
never to GitHub), then runs the six example scans through the exported file
and compares its answers with those recorded in the one-shot test run.
"""

import csv

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from app.demo import MODEL_PATH, load_examples, load_model
from src.data import CLASS_NAMES
from src.gradcam import class_scores, feature_maps

ONNX_PATH = MODEL_PATH.with_suffix(".onnx")
TEST_SCANS = MODEL_PATH.parent / "explain" / "test_scans.csv"


class BrowserModel(nn.Module):
    """The final model, returning the three scores and a heatmap for each tumour type.

    For this network Grad-CAM needs no backward pass. Each class score is an
    average of the 512 pattern maps, weighted by the last layer's weights, so
    the gradient Grad-CAM uses for map k is just that weight divided by 49
    (the 7x7 cells). Weighting the maps by the last layer's weights therefore
    gives Grad-CAM's heatmap, once it is scaled so its peak is 1. That lets a
    browser compute it with a single forward pass.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, scan):
        maps = feature_maps(self.model, scan)
        scores = class_scores(self.model, maps)
        weights = self.model.backbone.fc.weight  # (3 tumour types, 512 maps)
        cams = torch.relu(torch.einsum("ck,bkhw->bchw", weights, maps))  # keep evidence for, drop evidence against
        heatmaps = F.interpolate(cams, size=scan.shape[-2:], mode="bilinear", align_corners=False)
        return scores, heatmaps


def export(model, path):
    """Save the model as one ONNX file taking a (1, 1, 224, 224) scan with values in [0, 1].

    verbose=False keeps the exporter quiet; its progress messages include an
    emoji that the Windows console can't print.
    """
    torch.onnx.export(BrowserModel(model).eval(), (torch.zeros(1, 1, 224, 224),), path,
                      input_names=["scan"], output_names=["scores", "heatmaps"],
                      dynamo=True, external_data=False, verbose=False)


def run_onnx(session, pixels):
    """Scores (3,) and heatmaps (3, 224, 224) for one uint8 scan, from the exported model."""
    scan = (pixels.astype(np.float32) / 255.0)[None, None]  # as in training: values in [0, 1]
    scores, heatmaps = session.run(None, {"scan": scan})
    return scores[0], heatmaps[0]


def check_examples(path=ONNX_PATH):
    """Compare the exported model's answers on the examples with the recorded test run."""
    import onnxruntime  # only needed for this check

    session = onnxruntime.InferenceSession(str(path))
    with open(TEST_SCANS, newline="") as f:
        recorded = {int(row["index"]): row for row in csv.DictReader(f)}
    all_match = True
    for example in load_examples():
        scores, _ = run_onnx(session, example.scan)
        probabilities = np.exp(scores - scores.max())
        probabilities /= probabilities.sum()
        predicted = CLASS_NAMES[int(probabilities.argmax())]
        confidence = float(probabilities.max())
        row = recorded[example.index]
        same = predicted == row["predicted"] and abs(confidence - float(row["confidence"])) < 1e-3
        all_match &= same
        print(f"{example.name:30} exported {predicted:10} {confidence:.4f} | test run {row['predicted']:10} "
              f"{row['confidence']} | {'MATCH' if same else 'MISMATCH'}")
    return all_match


def main():
    export(load_model(), ONNX_PATH)
    print(f"Wrote {ONNX_PATH} ({ONNX_PATH.stat().st_size / 1e6:.1f} MB)")
    if not check_examples():
        raise SystemExit("The exported model does not reproduce the test run. Don't deploy it.")


if __name__ == "__main__":
    main()
