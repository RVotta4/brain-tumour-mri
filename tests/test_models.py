import pytest
import torch

from src.models import SmallCNN, build_model


def test_small_cnn_gives_three_scores_per_image():
    model = SmallCNN()

    output = model(torch.zeros(2, 1, 224, 224))

    assert output.shape == (2, 3)


def test_small_cnn_works_on_smaller_images_too():
    output = SmallCNN()(torch.zeros(4, 1, 32, 32))

    assert output.shape == (4, 3)


def test_build_model_by_name():
    assert isinstance(build_model("small_cnn"), SmallCNN)


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model"):
        build_model("not_a_model")
