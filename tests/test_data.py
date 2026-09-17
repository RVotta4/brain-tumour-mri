import numpy as np

from src.data import IMAGE_SIZE, preprocess_image, preprocess_mask


def test_preprocess_image_resizes_to_224_uint8():
    raw = np.random.default_rng(0).integers(0, 3000, size=(512, 512)).astype(np.int16)

    out = preprocess_image(raw)

    assert out.shape == (IMAGE_SIZE, IMAGE_SIZE) == (224, 224)
    assert out.dtype == np.uint8


def test_preprocess_image_stretches_to_full_brightness_range():
    raw = np.zeros((512, 512), dtype=np.int16)
    raw[:256, :] = 1000
    raw[256:, :] = 3000

    out = preprocess_image(raw)

    assert out.min() == 0
    assert out.max() == 255


def test_preprocess_image_handles_blank_slice_without_dividing_by_zero():
    raw = np.full((512, 512), 7, dtype=np.int16)

    out = preprocess_image(raw)

    assert (out == 0).all()


def test_preprocess_mask_stays_binary_and_keeps_tumour():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1

    out = preprocess_mask(mask)

    assert out.shape == (224, 224)
    assert set(np.unique(out)) == {0, 1}
    assert out.sum() > 0
