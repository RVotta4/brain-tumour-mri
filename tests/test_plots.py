import numpy as np

from src.plots import plot_augmentation, plot_confusion_matrix, plot_gradcam_examples, plot_history, plot_mistakes


def test_plot_history_writes_png(tmp_path):
    history = [
        {"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "train_accuracy": 0.4, "val_accuracy": 0.35},
        {"epoch": 2, "train_loss": 0.7, "val_loss": 0.9, "train_accuracy": 0.6, "val_accuracy": 0.5},
    ]
    path = tmp_path / "curves.png"

    plot_history(history, path)

    assert path.read_bytes()[:4] == b"\x89PNG"


def test_plot_confusion_matrix_writes_png(tmp_path):
    path = tmp_path / "cm.png"

    plot_confusion_matrix([[5, 1, 0], [2, 7, 1], [0, 0, 4]], ["a", "b", "c"], path)

    assert path.read_bytes()[:4] == b"\x89PNG"


def test_plot_augmentation_writes_png(tmp_path):
    import numpy as np

    images = [np.random.default_rng(i).random((16, 16)) for i in range(8)]
    path = tmp_path / "augmentation.png"

    plot_augmentation(images, path)

    assert path.read_bytes()[:4] == b"\x89PNG"


def tiny_panel(seed):
    rng = np.random.default_rng(seed)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = 1
    return rng.random((16, 16)), mask, rng.random((16, 16))


def test_plot_gradcam_examples_writes_png(tmp_path):
    examples = {"glioma": [tiny_panel(0), tiny_panel(1)], "pituitary": [tiny_panel(2)]}
    path = tmp_path / "examples.png"

    plot_gradcam_examples(examples, path)

    assert path.read_bytes()[:4] == b"\x89PNG"


def test_plot_mistakes_writes_png(tmp_path):
    items = []
    for i in range(3):
        image, mask, heatmap = tiny_panel(i)
        items.append({"image": image, "mask": mask, "heatmap": heatmap,
                      "label": f"said glioma (0.9{i})", "confident": i == 0})
    path = tmp_path / "mistakes.png"

    plot_mistakes(items, "True meningioma: every test mistake", path)

    assert path.read_bytes()[:4] == b"\x89PNG"
