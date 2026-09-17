from src.plots import plot_confusion_matrix, plot_history


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
