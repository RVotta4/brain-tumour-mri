import subprocess
import sys
from pathlib import Path

import pytest

from app.deploy import CODE, RENAMED, files_to_upload

ROOT = Path(__file__).resolve().parent.parent


def make_fake_repo(root):
    for path in CODE + list(RENAMED):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("x")
    (root / "app" / "examples").mkdir(exist_ok=True)
    for name in ["a.png", "a_mask.png", "examples.csv"]:
        (root / "app" / "examples" / name).write_text("x")


def test_files_to_upload_mirrors_the_repo_layout(tmp_path):
    make_fake_repo(tmp_path)

    files = files_to_upload(tmp_path)

    assert sorted(files.values()) == sorted(
        CODE + ["README.md", "requirements.txt", "app/examples/a.png", "app/examples/a_mask.png",
                "app/examples/examples.csv"]
    )
    assert files[tmp_path / "app" / "README.md"] == "README.md"
    assert files[tmp_path / "experiments" / "03-resnet18-finetuned" / "model.pt"] == "experiments/03-resnet18-finetuned/model.pt"


def test_files_to_upload_refuses_when_something_is_missing(tmp_path):
    make_fake_repo(tmp_path)
    (tmp_path / "experiments" / "03-resnet18-finetuned" / "model.pt").unlink()

    with pytest.raises(FileNotFoundError, match="model.pt"):
        files_to_upload(tmp_path)


def test_the_demo_needs_only_the_code_that_is_uploaded():
    # A fresh Python process, so modules loaded by other tests don't count.
    listing = "import sys, app.demo; print(' '.join(sorted(m for m in sys.modules if m.startswith(('src.', 'app.')))))"
    loaded = subprocess.run([sys.executable, "-c", listing], capture_output=True, text=True, check=True, cwd=ROOT)

    for module in loaded.stdout.split():
        assert module.replace(".", "/") + ".py" in CODE, f"{module} is imported by the demo but not uploaded"
