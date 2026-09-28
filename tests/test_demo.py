import csv
from pathlib import Path

import numpy as np
from PIL import Image

from app.demo import Example, OUTLINE_COLOUR, find_example, load_examples, outline, overlay, scan_pixels, to_grey


def fake_example(seed=0):
    """A stand-in example whose pixels stay within 10-199, so re-stretching would change them."""
    rng = np.random.default_rng(seed)
    scan = rng.integers(10, 200, size=(224, 224), dtype=np.uint8)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[100:130, 90:120] = 1
    return Example(name="fake", index=0, path=Path("fake.png"), scan=scan, mask=mask, patient_id="P1",
                   true="glioma", caption="Glioma — correct", note="A note.")


def test_to_grey_handles_colour_transparency_and_16_bit():
    rgb = Image.new("RGB", (5, 4), (77, 77, 77))
    rgba = Image.new("RGBA", (5, 4), (77, 77, 77, 128))
    deep = Image.fromarray(np.full((4, 5), 4000, dtype=np.uint16))  # 16-bit greyscale keeps its range

    for image, level in [(rgb, 77), (rgba, 77), (deep, 4000)]:
        grey = to_grey(image)
        assert grey.shape == (4, 5)
        assert grey.dtype == np.float32
        assert np.all(grey == level)


def test_an_upload_is_stretched_and_resized_to_224():
    rng = np.random.default_rng(1)
    upload = Image.fromarray(rng.integers(50, 150, size=(300, 200, 3), dtype=np.uint8))  # colour, not square

    pixels, example = scan_pixels(upload, [fake_example()])

    assert example is None
    assert pixels.shape == (224, 224)
    assert pixels.dtype == np.uint8
    # Stretched beyond the input's 50-149 range. (Not exactly 0-255: resizing after
    # the stretch blends neighbouring pixels, which pulls the extremes in.)
    assert pixels.min() < 50 and pixels.max() > 150


def test_a_blank_upload_does_not_crash():
    pixels, example = scan_pixels(Image.new("L", (64, 64), 128), [])

    assert example is None
    assert pixels.shape == (224, 224)
    assert pixels.max() == 0


def test_an_example_is_recognised_and_its_stored_pixels_are_used():
    example = fake_example()

    pixels, found = scan_pixels(Image.fromarray(example.scan), [fake_example(seed=5), example])

    assert found is example
    assert np.array_equal(pixels, example.scan)


def test_one_changed_pixel_means_it_is_not_the_example():
    example = fake_example()
    altered = example.scan.copy()
    altered[0, 0] += 1

    assert find_example(to_grey(Image.fromarray(altered)), [example]) is None


def test_load_examples_reads_the_folder(tmp_path):
    scan = (np.arange(224 * 224) % 256).astype(np.uint8).reshape(224, 224)
    mask = np.zeros((224, 224), dtype=np.uint8)
    mask[10:20, 10:20] = 1
    Image.fromarray(scan).save(tmp_path / "one.png")
    Image.fromarray(mask * 255).save(tmp_path / "one_mask.png")
    with open(tmp_path / "examples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "index", "patient_id", "true", "caption", "note"])
        writer.writeheader()
        writer.writerow({"name": "one", "index": 42, "patient_id": "P9", "true": "pituitary",
                         "caption": "Pituitary — correct", "note": "Hello."})

    (example,) = load_examples(tmp_path)

    assert example.name == "one"
    assert example.index == 42
    assert example.path == tmp_path / "one.png"
    assert np.array_equal(example.scan, scan)
    assert np.array_equal(example.mask, mask)
    assert (example.patient_id, example.true, example.caption, example.note) == ("P9", "pituitary", "Pituitary — correct", "Hello.")


def square_mask(size=64, top=20, left=20, width=10):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[top:top + width, left:left + width] = 1
    return mask


def test_outline_is_the_border_of_the_tumour():
    mask = square_mask()  # a 10x10 square

    assert outline(mask, width=1).sum() == 36  # 100 pixels minus the 8x8 inside
    assert outline(mask, width=2).sum() == 64  # 100 pixels minus the 6x6 inside
    assert not outline(mask, width=2)[25, 25]  # the middle is not border
    assert not outline(mask, width=2)[mask == 0].any()  # nothing outside the tumour
    assert not outline(np.zeros((64, 64), dtype=np.uint8)).any()


def is_outline_colour(image):
    return (image == OUTLINE_COLOUR).all(axis=2)


def test_overlay_draws_the_outline_only_on_the_border():
    mask = square_mask()
    black = np.zeros((64, 64), dtype=np.uint8)
    cold = np.zeros((64, 64), dtype=np.float32)

    with_outline = overlay(black, cold, mask)
    without = overlay(black, cold)

    assert with_outline.shape == (64, 64, 3)
    assert with_outline.dtype == np.uint8
    assert np.array_equal(is_outline_colour(with_outline), outline(mask))
    assert not is_outline_colour(without).any()


def test_overlay_blends_scan_and_heatmap_like_the_stage_4_figures():
    white = np.full((4, 4), 255, dtype=np.uint8)
    cold = np.zeros((4, 4), dtype=np.float32)  # jet at 0 is dark blue: (0, 0, 0.5)

    # 55% white scan + 45% dark blue: red and green 0.55 * 255 = 140.25; blue (0.55 + 0.45 * 0.5) * 255 = 197.6
    assert overlay(white, cold)[0, 0].tolist() == [140, 140, 198]
