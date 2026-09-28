import csv
from pathlib import Path

import numpy as np
from PIL import Image

from app.demo import Example, find_example, load_examples, scan_pixels, to_grey


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
