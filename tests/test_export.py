import numpy as np
import pytest
import torch

from app.export_onnx import BrowserModel, export, run_onnx
from src.gradcam import grad_cam
from src.models import ResNet18Grey


def random_model():
    torch.manual_seed(0)
    return ResNet18Grey(weights=None).eval()


def scaled(heatmap):
    peak = heatmap.max()
    return heatmap / peak if peak > 0 else heatmap


def test_the_browser_heatmap_equals_grad_cam():
    model = random_model()
    scan = torch.rand(1, 224, 224)

    expected, predicted, _ = grad_cam(model, scan)
    with torch.no_grad():
        scores, heatmaps = BrowserModel(model)(scan.unsqueeze(0))

    assert int(scores.argmax()) == predicted
    np.testing.assert_allclose(scaled(heatmaps[0, predicted].numpy()), expected, atol=1e-4)


def test_the_exported_file_gives_the_same_answers(tmp_path):
    onnxruntime = pytest.importorskip("onnxruntime")
    model = random_model()

    export(model, tmp_path / "model.onnx")

    assert not (tmp_path / "model.onnx.data").exists()  # one file, weights inside
    pixels = np.random.default_rng(3).integers(0, 256, size=(224, 224), dtype=np.uint8)
    scores, heatmaps = run_onnx(onnxruntime.InferenceSession(str(tmp_path / "model.onnx")), pixels)
    with torch.no_grad():
        expected_scores, expected_heatmaps = BrowserModel(model)(torch.from_numpy(pixels / 255.0).float()[None, None])
    np.testing.assert_allclose(scores, expected_scores[0].numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(heatmaps, expected_heatmaps[0].numpy(), rtol=1e-4, atol=1e-4)
