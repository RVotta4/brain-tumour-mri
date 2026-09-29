import subprocess
import sys
from pathlib import Path

import pytest
import torch

from src.gradcam import class_scores, feature_maps, grad_cam
from src.models import ResNet18Grey, SmallCNN

ROOT = Path(__file__).resolve().parent.parent


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


def test_gradcam_needs_only_the_model_code():
    # A fresh Python process, so modules loaded by other tests don't count.
    listing = "import sys, src.gradcam; print(sorted(m for m in sys.modules if m.startswith('src.')))"
    loaded = subprocess.run([sys.executable, "-c", listing], capture_output=True, text=True, check=True, cwd=ROOT)

    assert loaded.stdout.strip() == "['src.gradcam', 'src.models']"
