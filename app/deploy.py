"""Upload the live demo to a Hugging Face Space.

Usage (after `hf auth login`):
    .\\.venv\\Scripts\\python.exe -m app.deploy --space <username>/brain-tumour-mri

The Space mirrors this repo's folders, so the app finds everything at the same
paths as on the laptop. The model goes to Hugging Face only, never to GitHub.
Running it again uploads the current files as a new commit on the Space.
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Uploaded at the same path as in this repo.
CODE = [
    "app/__init__.py",
    "app/app.py",
    "app/demo.py",
    "src/__init__.py",
    "src/data.py",
    "src/gradcam.py",
    "src/models.py",
    "experiments/03-resnet18-finetuned/model.pt",
]
# Uploaded to where Hugging Face looks for them: the Space's top folder.
RENAMED = {"app/README.md": "README.md", "app/requirements.txt": "requirements.txt"}


def files_to_upload(root=ROOT):
    """Map each local file to its path on the Space, refusing if any is missing."""
    root = Path(root)
    files = {root / path: path for path in CODE}
    files.update({root / local: remote for local, remote in RENAMED.items()})
    for path in sorted((root / "app" / "examples").iterdir()):
        files[path] = f"app/examples/{path.name}"
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing: " + ", ".join(missing))
    return files


def deploy(space):
    """Create the Space if needed (public, Gradio, free CPU), then upload everything in one commit."""
    from huggingface_hub import CommitOperationAdd, HfApi  # comes with Gradio; not needed by the tests

    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="gradio", private=False, exist_ok=True)
    operations = [CommitOperationAdd(path_in_repo=remote, path_or_fileobj=str(local))
                  for local, remote in files_to_upload().items()]
    api.create_commit(repo_id=space, repo_type="space", operations=operations,
                      commit_message="Deploy the demo from github.com/RVotta4/brain-tumour-mri")
    print(f"Uploaded {len(operations)} files. The Space builds in a few minutes: "
          f"https://huggingface.co/spaces/{space}")


def main():
    parser = argparse.ArgumentParser(description="Upload the live demo to a Hugging Face Space.")
    parser.add_argument("--space", required=True, help="<username>/<space-name>")
    deploy(parser.parse_args().space)


if __name__ == "__main__":
    main()
