"""Put the live demo on a free static Hugging Face Space, or preview it locally.

Usage:
    .\\.venv\\Scripts\\python.exe -m app.deploy --preview space_preview
    .\\.venv\\Scripts\\python.exe -m app.deploy --space <username>/brain-tumour-mri   (after `hf auth login`)

A static Space only hands files to the visitor's browser, which runs the model
itself. The model goes to Hugging Face only, never to GitHub. Running --space
again uploads the current files as a new commit on the Space.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ["index.html", "style.css", "pipeline.js", "app.js"]
MODEL = "experiments/03-resnet18-finetuned/model.onnx"
EXAMPLE_FIELDS = ["name", "patient_id", "true", "caption", "note"]  # what the page shows about each example


def examples_json(examples_dir):
    """The example list the page reads, made from app/examples/examples.csv so the text lives in one place."""
    with open(examples_dir / "examples.csv", newline="", encoding="utf-8") as f:
        rows = [{field: row[field] for field in EXAMPLE_FIELDS} for row in csv.DictReader(f)]
    return json.dumps(rows, indent=2, ensure_ascii=False).encode("utf-8")


def site_files(root=ROOT):
    """Everything the Space serves: its path there -> a local file, or the bytes to write."""
    root = Path(root)
    files = {name: root / "space" / name for name in PAGE}
    files["README.md"] = root / "space" / "README.md"
    files["model.onnx"] = root / MODEL
    examples_dir = root / "app" / "examples"
    for path in sorted(examples_dir.glob("*.png")):
        files[f"examples/{path.name}"] = path
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing: " + ", ".join(missing))
    files["examples.json"] = examples_json(examples_dir)
    return files


def preview(folder, root=ROOT):
    """Write the site into a local folder, exactly as it will be uploaded."""
    folder = Path(folder)
    for remote, source in site_files(root).items():
        target = folder / remote
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source if isinstance(source, bytes) else source.read_bytes())
    print(f"Wrote the site to {folder}. To view it, run\n"
          f"    .\\.venv\\Scripts\\python.exe -m http.server 8000 --directory {folder}\n"
          "then open http://localhost:8000 (Ctrl+C stops the server).")


def deploy(space):
    """Create the static Space if needed (public, free), then upload everything in one commit."""
    from huggingface_hub import CommitOperationAdd, HfApi  # only needed for uploading; listed in requirements.txt

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="static", private=False, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=remote,
                                     path_or_fileobj=source if isinstance(source, bytes) else str(source))
                  for remote, source in site_files().items()]
    api.create_commit(repo_id=space, repo_type="space", operations=operations,
                      commit_message="Deploy the demo from github.com/RVotta4/brain-tumour-mri")
    print(f"Uploaded {len(operations)} files. It should be live within a minute: "
          f"https://huggingface.co/spaces/{space}")


def main():
    parser = argparse.ArgumentParser(description="Put the live demo on a static Hugging Face Space.")
    where = parser.add_mutually_exclusive_group(required=True)
    where.add_argument("--space", help="<username>/<space-name> on Hugging Face")
    where.add_argument("--preview", help="a local folder to write the site into")
    args = parser.parse_args()
    if args.preview:
        preview(args.preview)
    else:
        deploy(args.space)


if __name__ == "__main__":
    main()
