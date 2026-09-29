"""Checks the browser demo's image steps (space/pipeline.js) against the Python code.

The Python functions write their answers to a file; the Node tests compare
the JavaScript's answers with it.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from matplotlib import colormaps
from PIL import Image

from app.demo import load_examples, overlay
from src.data import CLASS_NAMES, preprocess_image

ROOT = Path(__file__).resolve().parent.parent
SIZES = [(300, 200), (100, 150), (37, 99), (224, 224), (400, 333)]  # (height, width): shrink, enlarge, odd, same


def reference_cases():
    rng = np.random.default_rng(0)
    for height, width in SIZES:
        rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        rgb[: height // 3] //= 3  # a darker band, so it isn't pure noise
        grey = np.asarray(Image.fromarray(rgb).convert("L"))
        yield {
            "width": width,
            "height": height,
            "rgba": np.dstack([rgb, np.full((height, width), 255, np.uint8)]).ravel().tolist(),
            "grey": grey.ravel().tolist(),
            "pixels": preprocess_image(grey.astype(np.float32)).ravel().tolist(),
        }


def write_fixtures(path):
    example = load_examples()[2]
    rows, cols = np.mgrid[0:224, 0:224]
    heatmap = np.exp(-((rows - 90) ** 2 + (cols - 140) ** 2) / 900.0).astype(np.float32)
    heatmap /= heatmap.max()
    fixtures = {
        "class_names": CLASS_NAMES,
        "cases": list(reference_cases()),
        "jet": colormaps["jet"](np.arange(256) / 255.0)[:, :3].tolist(),
        "overlay": {
            "pixels": example.scan.ravel().tolist(),
            "mask": example.mask.ravel().tolist(),
            "heatmap": heatmap.ravel().tolist(),
            "with_outline": overlay(example.scan, heatmap, example.mask).ravel().tolist(),
            "without_outline": overlay(example.scan, heatmap).ravel().tolist(),
        },
    }
    path.write_text(json.dumps(fixtures), encoding="utf-8")


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_the_browser_pipeline_matches_python(tmp_path):
    fixtures = tmp_path / "fixtures.json"
    write_fixtures(fixtures)

    result = subprocess.run(
        ["node", "--test", "--test-reporter=tap", "space/tests/pipeline.test.mjs"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", env={**os.environ, "FIXTURES": str(fixtures)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "# skipped 0" in result.stdout, "some browser tests were skipped"  # Node 24's TAP wording
