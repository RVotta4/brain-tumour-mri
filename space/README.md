---
title: Brain Tumour MRI Classifier
emoji: 🧠
colorFrom: blue
colorTo: gray
sdk: static
pinned: false
short_description: Brain tumour MRI classifier that runs in your browser
---

# Brain tumour MRI classifier

**Educational project, not a medical device, not for diagnosis.**

Upload a 2D brain MRI slice (PNG or JPG), or click one of six test-set examples. The page shows the model's
predicted tumour type (glioma, meningioma or pituitary), its confidence for all three, and a Grad-CAM
heatmap of where the evidence for its answer came from. For the examples, the clinicians' tumour outline
is drawn in green.

The model runs entirely in your browser with ONNX Runtime Web: images you upload never leave your device.

The model is a ResNet-18 fine-tuned on the Cheng et al. dataset, scoring 90.9% on 430 held-out test scans
from 34 patients. The full report, including where it fails, is at
[github.com/RVotta4/brain-tumour-mri](https://github.com/RVotta4/brain-tumour-mri).

Example scans: Cheng, Jun (2017). *brain tumor dataset*. figshare.
[doi:10.6084/m9.figshare.1512427](https://doi.org/10.6084/m9.figshare.1512427) (CC BY 4.0).
