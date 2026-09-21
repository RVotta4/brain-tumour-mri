import pytest
import torch

from src.models import ResNet18Grey, SmallCNN, build_model


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


def test_resnet18_gives_three_scores_per_image():
    model = ResNet18Grey(weights=None)

    output = model(torch.zeros(2, 1, 224, 224))

    assert output.shape == (2, 3)


def test_resnet18_repeats_the_grey_channel():
    model = ResNet18Grey(weights=None)

    prepared = model.prepare(torch.zeros(2, 1, 8, 8))

    assert prepared.shape == (2, 3, 8, 8)


def test_resnet18_applies_imagenet_normalisation():
    model = ResNet18Grey(weights=None)

    prepared = model.prepare(torch.full((1, 1, 2, 2), 0.485))

    assert prepared[0, 0].mean().item() == pytest.approx(0.0, abs=1e-6)
    assert prepared[0, 1].mean().item() == pytest.approx((0.485 - 0.456) / 0.224, abs=1e-5)
    assert prepared[0, 2].mean().item() == pytest.approx((0.485 - 0.406) / 0.225, abs=1e-5)
