import pytest

from src.metrics import accuracy, confusion_matrix, precision_recall

Y_TRUE = [0, 0, 0, 1, 1, 2]
Y_PRED = [0, 0, 1, 1, 1, 0]


def test_confusion_matrix_rows_are_true_columns_are_predicted():
    assert confusion_matrix(Y_TRUE, Y_PRED, n_classes=3) == [
        [2, 1, 0],
        [0, 2, 0],
        [1, 0, 0],
    ]


def test_accuracy_is_fraction_correct():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)


def test_accuracy_rejects_empty_input():
    with pytest.raises(ValueError):
        accuracy([], [])


def test_precision_and_recall_match_hand_calculation():
    matrix = confusion_matrix(Y_TRUE, Y_PRED, n_classes=3)
    scores = precision_recall(matrix)

    assert scores[0]["precision"] == pytest.approx(2 / 3)
    assert scores[0]["recall"] == pytest.approx(2 / 3)
    assert scores[1]["precision"] == pytest.approx(2 / 3)
    assert scores[1]["recall"] == pytest.approx(1.0)


def test_class_never_predicted_scores_zero_instead_of_crashing():
    matrix = confusion_matrix(Y_TRUE, Y_PRED, n_classes=3)
    scores = precision_recall(matrix)

    assert scores[2] == {"precision": 0.0, "recall": 0.0}


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1], [0], n_classes=2)
