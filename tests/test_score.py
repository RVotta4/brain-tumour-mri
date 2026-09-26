import json

import pytest

from src.score import TEST_RESULTS, check_test_set_unused, score_experiment
from src.train import run_experiment
from tests.test_train import write_tiny_dataset


def trained_tiny_experiment(tmp_path):
    dataset_path, splits_path = write_tiny_dataset(tmp_path)
    experiments_dir = tmp_path / "experiments"
    metrics = run_experiment(
        name="tiny", model_name="small_cnn", epochs=1, batch_size=4, lr=1e-3, seed=42,
        dataset_path=dataset_path, splits_path=splits_path, experiments_dir=experiments_dir,
    )
    paths = {"dataset_path": dataset_path, "splits_path": splits_path, "experiments_dir": experiments_dir}
    return metrics, paths


def test_guard_allows_an_unused_test_set(tmp_path):
    (tmp_path / "01-anything").mkdir()

    check_test_set_unused(tmp_path)


def test_guard_refuses_once_any_experiment_has_test_results(tmp_path):
    (tmp_path / "01-anything").mkdir()
    (tmp_path / "01-anything" / TEST_RESULTS).write_text("{}")

    with pytest.raises(RuntimeError, match="already been used"):
        check_test_set_unused(tmp_path)


def test_scoring_on_test_writes_results_and_then_locks(tmp_path):
    _, paths = trained_tiny_experiment(tmp_path)

    metrics = score_experiment("tiny", "test", **paths)

    folder = paths["experiments_dir"] / "tiny"
    assert metrics["split"] == "test"
    assert metrics["n_scans"] == 9  # 1 test patient per class x 3 slices
    assert metrics["best_epoch"] == 1
    assert json.loads((folder / TEST_RESULTS).read_text())["split"] == "test"
    assert (folder / "test_confusion_matrix.png").exists()
    with pytest.raises(RuntimeError, match="already been used"):
        score_experiment("tiny", "test", **paths)


def test_rescoring_validation_reproduces_training_result_and_writes_nothing(tmp_path):
    trained, paths = trained_tiny_experiment(tmp_path)

    rescored = score_experiment("tiny", "validation", **paths)

    assert rescored["accuracy"] == trained["accuracy"]
    assert rescored["loss"] == trained["loss"]
    assert rescored["confusion_matrix"] == trained["confusion_matrix"]
    assert not (paths["experiments_dir"] / "tiny" / TEST_RESULTS).exists()


def test_unknown_split_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="split"):
        score_experiment("tiny", "train", experiments_dir=tmp_path)


def test_a_quick_subset_run_cannot_spend_the_test_set(tmp_path):
    folder = tmp_path / "smoke"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"model": "small_cnn", "subset": 64}))

    with pytest.raises(ValueError, match="subset"):
        score_experiment("smoke", "test", experiments_dir=tmp_path)

    assert not (folder / TEST_RESULTS).exists()
