import csv
import json

import pytest

from app.deploy import MODEL, PAGE, preview, site_files


def make_fake_repo(root):
    for name in PAGE + ["README.md"]:
        (root / "space").mkdir(exist_ok=True)
        (root / "space" / name).write_text("x")
    (root / MODEL).parent.mkdir(parents=True)
    (root / MODEL).write_text("x")
    examples = root / "app" / "examples"
    examples.mkdir(parents=True)
    for name in ["a.png", "a_mask.png"]:
        (examples / name).write_text("x")
    with open(examples / "examples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "index", "patient_id", "true", "caption", "note"])
        writer.writeheader()
        writer.writerow({"name": "a", "index": 7, "patient_id": "P1", "true": "glioma",
                         "caption": "Glioma — correct", "note": "Hi, there."})


def test_site_files_lists_the_page_model_and_examples(tmp_path):
    make_fake_repo(tmp_path)

    files = site_files(tmp_path)

    assert sorted(files) == sorted(PAGE + ["README.md", "model.onnx", "examples/a.png", "examples/a_mask.png",
                                           "examples.json"])
    assert files["model.onnx"] == tmp_path / MODEL


def test_examples_json_carries_what_the_page_shows(tmp_path):
    make_fake_repo(tmp_path)

    examples = json.loads(site_files(tmp_path)["examples.json"])

    assert examples == [{"name": "a", "patient_id": "P1", "true": "glioma", "caption": "Glioma — correct",
                         "note": "Hi, there."}]


def test_site_files_refuses_when_something_is_missing(tmp_path):
    make_fake_repo(tmp_path)
    (tmp_path / MODEL).unlink()

    with pytest.raises(FileNotFoundError, match="model.onnx"):
        site_files(tmp_path)


def test_preview_writes_the_site_to_a_folder(tmp_path):
    make_fake_repo(tmp_path)

    preview(tmp_path / "out", root=tmp_path)

    assert (tmp_path / "out" / "index.html").read_text() == "x"
    assert (tmp_path / "out" / "examples" / "a_mask.png").exists()
    assert json.loads((tmp_path / "out" / "examples.json").read_text(encoding="utf-8"))[0]["name"] == "a"
