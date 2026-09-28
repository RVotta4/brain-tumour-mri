// Tests for pipeline.js. Run them through pytest (tests/test_space.py), which first
// writes the Python code's answers to a file and passes its path in FIXTURES.

import assert from "node:assert/strict";
import fs from "node:fs";
import { test } from "node:test";

import * as pipeline from "../pipeline.js";

const fixtures = process.env.FIXTURES ? JSON.parse(fs.readFileSync(process.env.FIXTURES, "utf8")) : null;
const needsFixtures = { skip: fixtures ? false : "run through pytest, which writes the Python answers" };

test("halves round to the nearest even number, like NumPy", () => {
  assert.deepEqual([0.5, 1.5, 2.5, 3.5, 2.4, 2.6].map(pipeline.roundHalfEven), [0, 2, 2, 4, 2, 3]);
});

test("the outline is the tumour's border", () => {
  const mask = new Uint8Array(64 * 64);
  for (let y = 20; y < 30; y++) for (let x = 20; x < 30; x++) mask[y * 64 + x] = 1; // a 10x10 square
  const count = (pixels) => pixels.reduce((total, v) => total + v, 0);

  assert.equal(count(pipeline.outline(mask, 64, 64, 1)), 36); // 100 minus the 8x8 inside
  assert.equal(count(pipeline.outline(mask, 64, 64, 2)), 64); // 100 minus the 6x6 inside
  assert.equal(count(pipeline.outline(new Uint8Array(64 * 64), 64, 64)), 0);
});

test("softmax gives probabilities that sum to 1, in score order", () => {
  const probabilities = pipeline.softmax([2, 1, 0]);

  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(probabilities[0] > probabilities[1] && probabilities[1] > probabilities[2]);
});

test("an example is recognised only when every pixel matches", () => {
  const scan = Uint8Array.from({ length: 224 * 224 }, (_, i) => i % 251);
  const example = { name: "e", scan };
  const altered = scan.slice();
  altered[0] += 1;

  assert.equal(pipeline.findExample(scan.slice(), 224, 224, [example]), example);
  assert.equal(pipeline.findExample(altered, 224, 224, [example]), null);
  assert.equal(pipeline.findExample(scan.slice(0, 100), 10, 10, [example]), null);
});

test("the predicted type's heatmap is scaled so its peak is 1", () => {
  const heatmaps = new Float32Array(3 * 4);
  heatmaps.set([0, 1, 2, 4], 4); // the second type's map

  assert.deepEqual(Array.from(pipeline.normaliseHeatmap(heatmaps, 1, 4)), [0, 0.25, 0.5, 1]);
  assert.deepEqual(Array.from(pipeline.normaliseHeatmap(new Float32Array(12), 0, 4)), [0, 0, 0, 0]);
});

test("the summary says whether an example was right", () => {
  const example = { true: "glioma", patient_id: "P1", note: "A note." };

  const right = pipeline.describe("glioma", 0.93, example);
  assert.equal(right.headline, "Prediction: glioma (93.0% confidence)");
  assert.match(right.lines[0], /correct/);

  const wrong = pipeline.describe("meningioma", 0.99, example);
  assert.match(wrong.lines[0], /mistake/);
  assert.equal(wrong.lines[1], "A note.");

  const upload = pipeline.describe("pituitary", 0.5, null);
  assert.match(upload.lines[0], /outline/);
  assert.doesNotMatch(upload.lines.join(" "), /test patient/);
});

test("the same tumour types, in the same order, as the Python code", needsFixtures, () => {
  assert.deepEqual(pipeline.CLASS_NAMES, fixtures.class_names);
});

test("greyscale and preparation match Python pixel for pixel", needsFixtures, () => {
  for (const c of fixtures.cases) {
    const grey = pipeline.toGrey(c.rgba, c.width, c.height);
    assert.deepEqual(Array.from(grey), c.grey, `greyscale, ${c.width}x${c.height}`);
    assert.deepEqual(Array.from(pipeline.preprocess(grey, c.width, c.height)), c.pixels, `prepared, ${c.width}x${c.height}`);
  }
});

test("the jet colours match Matplotlib's table", needsFixtures, () => {
  fixtures.jet.forEach((rgb, i) => rgb.forEach((value, channel) => {
    assert.ok(Math.abs(value - pipeline.JET_TABLE[channel][i]) < 1e-12, `entry ${i}, channel ${channel}`);
  }));
});

test("the overlay matches Python pixel for pixel", needsFixtures, () => {
  const o = fixtures.overlay;
  const pixels = Uint8Array.from(o.pixels);
  const heatmap = Float32Array.from(o.heatmap);
  for (const [mask, expected] of [[Uint8Array.from(o.mask), o.with_outline], [null, o.without_outline]]) {
    const rgb = Array.from(pipeline.overlay(pixels, heatmap, mask)).filter((_, i) => i % 4 !== 3); // drop alpha
    assert.deepEqual(rgb, expected);
  }
});
