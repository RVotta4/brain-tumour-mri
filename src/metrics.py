"""Scoring functions for classification results.

Plain Python with no PyTorch, so every function can be checked against
examples worked out by hand.
"""


def confusion_matrix(y_true, y_pred, n_classes):
    """Count how often each true class was predicted as each class.

    Row = true class, column = predicted class. matrix[1][0] is the number
    of class-1 scans the model called class 0.
    """
    matrix = [[0] * n_classes for _ in range(n_classes)]
    for true, predicted in zip(y_true, y_pred, strict=True):
        matrix[true][predicted] += 1
    return matrix


def accuracy(y_true, y_pred):
    """Fraction of predictions that are correct."""
    if len(y_true) == 0:
        raise ValueError("no predictions to score")
    correct = sum(1 for true, predicted in zip(y_true, y_pred, strict=True) if true == predicted)
    return correct / len(y_true)


def precision_recall(matrix):
    """Per-class precision and recall from a confusion matrix.

    precision: of the scans predicted as this class, the fraction that really were.
    recall:    of the scans that really are this class, the fraction the model caught.

    A class with nothing to divide by scores 0.0 rather than crashing.
    """
    n = len(matrix)
    scores = []
    for c in range(n):
        true_positive = matrix[c][c]
        predicted_as_c = sum(matrix[row][c] for row in range(n))
        actually_c = sum(matrix[c])
        scores.append({
            "precision": true_positive / predicted_as_c if predicted_as_c else 0.0,
            "recall": true_positive / actually_c if actually_c else 0.0,
        })
    return scores
