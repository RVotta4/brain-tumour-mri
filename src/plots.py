"""Charts saved alongside each experiment."""

import matplotlib

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt


def plot_history(history, path):
    """Loss and accuracy per epoch, training vs validation, side by side.

    If the training line keeps improving while validation flattens or gets
    worse, the model is memorising the training scans: overfitting.
    """
    epochs = [row["epoch"] for row in history]
    fig, (ax_loss, ax_accuracy) = plt.subplots(1, 2, figsize=(10, 4))

    ax_loss.plot(epochs, [row["train_loss"] for row in history], marker="o", label="training")
    ax_loss.plot(epochs, [row["val_loss"] for row in history], marker="o", label="validation")
    ax_loss.set_title("Loss (lower is better)")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()

    ax_accuracy.plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="training")
    ax_accuracy.plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="validation")
    ax_accuracy.set_title("Accuracy (higher is better)")
    ax_accuracy.set_xlabel("Epoch")
    ax_accuracy.set_ylim(0, 1)
    ax_accuracy.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_confusion_matrix(matrix, class_names, path):
    """Grid of true type (rows) vs predicted type (columns), with counts."""
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(n), labels=class_names)
    ax.set_yticks(range(n), labels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    largest = max(max(row) for row in matrix) or 1
    for row in range(n):
        for col in range(n):
            value = matrix[row][col]
            colour = "white" if value > largest / 2 else "black"
            ax.text(col, row, value, ha="center", va="center", color=colour)

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
