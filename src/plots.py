"""Charts saved alongside each experiment."""

import math

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


def draw_outline(ax, mask):
    """Draw the clinician's tumour outline in green."""
    ax.contour(mask, levels=[0.5], colors="lime", linewidths=0.8)


def plot_gradcam_examples(examples, path):
    """Correctly classified scans, one row per tumour type.

    examples maps type name -> list of (image, mask, heatmap). Each example is
    a pair of panels: the scan with the tumour outline, then the Grad-CAM
    heatmap over the scan (red = the regions that drove the decision).
    """
    columns = 2 * max(len(items) for items in examples.values())
    fig, axes = plt.subplots(len(examples), columns, figsize=(1.9 * columns, 2.1 * len(examples)), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for row, (class_name, items) in enumerate(examples.items()):
        for k, (image, mask, heatmap) in enumerate(items):
            scan_ax, heat_ax = axes[row, 2 * k], axes[row, 2 * k + 1]
            scan_ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            draw_outline(scan_ax, mask)
            scan_ax.set_title(class_name, fontsize=9)
            heat_ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            heat_ax.imshow(heatmap, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            heat_ax.set_title("heatmap", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_mistakes(items, title, path, columns=5):
    """Every mistake for one true tumour type: heatmap over the scan, with the tumour outline.

    items: dicts with image, mask, heatmap, label and confident.
    Confident mistakes are titled in red.
    """
    rows = math.ceil(len(items) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.3 * columns, 2.5 * rows + 0.5), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, item in zip(axes.flat, items):
        ax.imshow(item["image"], cmap="gray", vmin=0, vmax=1)
        ax.imshow(item["heatmap"], cmap="jet", alpha=0.45, vmin=0, vmax=1)
        draw_outline(ax, item["mask"])
        ax.set_title(item["label"], fontsize=7, color="red" if item["confident"] else "black")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_augmentation(images, path):
    """One scan and its augmented versions in a 2x4 grid.

    images[0] is the original; the rest are the same scan after the random
    training changes. Used to check by eye that augmentation looks realistic.
    """
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.5))
    for index, (ax, image) in enumerate(zip(axes.flat, images)):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1)
        ax.set_title("original" if index == 0 else f"augmented {index}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
