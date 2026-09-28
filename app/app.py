"""The live demo page: upload a brain MRI slice, see what the final model says and where it looked.

Run locally from the repo root:
    .\\.venv\\Scripts\\python.exe app\\app.py
then open http://localhost:7860. Hugging Face Spaces runs the same file.
"""

import sys
from pathlib import Path

# Started as `python app/app.py`, Python puts the app/ folder first on its import
# path, where this file would hide the `app` package. Putting the repo root first
# makes `app.demo` and `src` import the same way here and on the Space.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from app.demo import analyse, describe, load_examples, load_model  # noqa: E402

TITLE = "Brain tumour MRI classifier"

WARNING = f"""# {TITLE}

**Educational project — not a medical device, not for diagnosis.**

This model only knows three tumour types: glioma, meningioma and pituitary. Given a healthy scan or a
non-MRI image, it will still pick one of the three.

Uploads are not saved; temporary copies are deleted within an hour.
"""

CREDIT = """ResNet-18 fine-tuned on 2D contrast-enhanced T1-weighted MRI slices: 90.9% accuracy on 430 held-out
test scans from 34 patients. How it was built, and where it fails:
[github.com/RVotta4/brain-tumour-mri](https://github.com/RVotta4/brain-tumour-mri).

Example scans: Cheng, Jun (2017). *brain tumor dataset*. figshare.
[doi:10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427) (CC BY 4.0).
"""

model = load_model()
examples = load_examples()


def run(image):
    """Called whenever the image changes; clearing it clears the results."""
    if image is None:
        return None, None, ""
    analysis = analyse(model, image, examples)
    return analysis.probabilities, analysis.image, describe(analysis)


# delete_cache=(3600, 3600): every hour, delete temporary files older than an hour.
with gr.Blocks(title=TITLE, analytics_enabled=False, delete_cache=(3600, 3600)) as page:
    gr.Markdown(WARNING)
    with gr.Row():
        with gr.Column():
            scan = gr.Image(label="MRI slice (PNG or JPG; DICOM files are not supported)", type="pil",
                            image_mode=None, sources=["upload"])
            gr.Examples(examples=[str(example.path) for example in examples], inputs=scan,
                        example_labels=[example.caption for example in examples], label="Example test scans")
        with gr.Column():
            summary = gr.Markdown()
            bars = gr.Label(label="Confidence", num_top_classes=3)
            heatmap = gr.Image(label="Grad-CAM: where the evidence came from (red = most)", interactive=False)
    gr.Markdown(CREDIT)
    scan.change(run, inputs=scan, outputs=[bars, heatmap, summary])


if __name__ == "__main__":
    page.launch()
