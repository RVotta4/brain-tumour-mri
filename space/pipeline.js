// The browser demo's image steps: the same arithmetic as app/demo.py and the
// libraries it uses (Pillow, NumPy, Matplotlib), so the model sees exactly the
// pixels it saw in Python. No browser APIs here, so Node can test every function.

export const CLASS_NAMES = ["meningioma", "glioma", "pituitary"]; // src/data.py's order
export const SIZE = 224;
export const HEAT_ALPHA = 0.45; // how strongly the heatmap tints the scan
export const OUTLINE_COLOUR = [0, 255, 0]; // lime
export const OUTLINE_WIDTH = 2; // pixels

const f32 = Math.fround; // round to a 32-bit float, as NumPy's float32 and Pillow's "F" images do

/** Pillow's convert("L"): grey levels from RGBA pixels, with Pillow's integer rounding. */
export function toGrey(rgba, width, height) {
  const grey = new Uint8Array(width * height);
  for (let i = 0; i < grey.length; i++) {
    const r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2];
    grey[i] = (r * 19595 + g * 38470 + b * 7471 + 0x8000) >> 16;
  }
  return grey;
}

/** NumPy's np.round: exact halves go to the nearest even number. */
export function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Pillow's bilinear weights for one axis. When shrinking, the filter widens to smooth. */
function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterScale = Math.max(scale, 1);
  const support = filterScale; // bilinear reaches 1 pixel, times the widening
  const inverse = 1 / filterScale;
  const rows = [];
  for (let out = 0; out < outSize; out++) {
    const centre = (out + 0.5) * scale;
    const first = Math.max(0, Math.trunc(centre - support + 0.5));
    const last = Math.min(inSize, Math.trunc(centre + support + 0.5));
    const weights = [];
    let total = 0;
    for (let x = first; x < last; x++) {
      const w = Math.max(0, 1 - Math.abs((x - centre + 0.5) * inverse));
      weights.push(w);
      total += w;
    }
    rows.push({ first, weights: weights.map((w) => (total !== 0 ? w / total : w)) });
  }
  return rows;
}

/** Pillow's resize(BILINEAR) of a float image: across, then down, each result rounded to 32 bits. */
function resize(image, width, height, outWidth, outHeight) {
  let current = image;
  let rowLength = width;
  if (outWidth !== width) {
    const columns = coefficients(width, outWidth);
    const next = new Float32Array(outWidth * height);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < outWidth; x++) {
        const { first, weights } = columns[x];
        let sum = 0;
        for (let k = 0; k < weights.length; k++) sum += current[y * rowLength + first + k] * weights[k];
        next[y * outWidth + x] = sum;
      }
    }
    current = next;
    rowLength = outWidth;
  }
  if (outHeight !== height) {
    const rows = coefficients(height, outHeight);
    const next = new Float32Array(rowLength * outHeight);
    for (let y = 0; y < outHeight; y++) {
      const { first, weights } = rows[y];
      for (let x = 0; x < rowLength; x++) {
        let sum = 0;
        for (let k = 0; k < weights.length; k++) sum += current[(first + k) * rowLength + x] * weights[k];
        next[y * rowLength + x] = sum;
      }
    }
    current = next;
  }
  return current;
}

/** src/data.py's preprocess_image: stretch to [0, 1], resize to 224x224, back to 0-255. */
export function preprocess(grey, width, height) {
  let low = Infinity;
  let high = -Infinity;
  for (const v of grey) {
    if (v < low) low = v;
    if (v > high) high = v;
  }
  const stretched = new Float32Array(grey.length); // stays all zero for a blank image
  if (high > low) {
    const range = f32(high - low);
    for (let i = 0; i < grey.length; i++) stretched[i] = f32(f32(grey[i] - low) / range);
  }
  const resized = resize(stretched, width, height, SIZE, SIZE);
  const pixels = new Uint8Array(SIZE * SIZE);
  for (let i = 0; i < pixels.length; i++) pixels[i] = Math.min(255, Math.max(0, roundHalfEven(f32(resized[i] * 255))));
  return pixels;
}

/** The example whose 224x224 scan equals these grey pixels exactly, or null. */
export function findExample(grey, width, height, examples) {
  if (width !== SIZE || height !== SIZE) return null;
  return examples.find((example) => example.scan.length === grey.length && example.scan.every((v, i) => v === grey[i])) ?? null;
}

/** Scores to probabilities that sum to 1. */
export function softmax(scores) {
  const peak = Math.max(...scores);
  const exps = scores.map((s) => Math.exp(s - peak));
  const total = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / total);
}

/** One tumour type's heatmap from the model's (3, 224, 224) output, scaled so its peak is 1. */
export function normaliseHeatmap(heatmaps, type, pixels = SIZE * SIZE) {
  const heatmap = heatmaps.slice(type * pixels, (type + 1) * pixels);
  let peak = 0;
  for (const v of heatmap) if (v > peak) peak = v;
  return peak > 0 ? heatmap.map((v) => v / peak) : heatmap;
}

// Matplotlib's "jet" colour map, as the points it is defined by: [position, value below, value above].
const JET_POINTS = {
  red: [[0, 0, 0], [0.35, 0, 0], [0.66, 1, 1], [0.89, 1, 1], [1, 0.5, 0.5]],
  green: [[0, 0, 0], [0.125, 0, 0], [0.375, 1, 1], [0.64, 1, 1], [0.91, 0, 0], [1, 0, 0]],
  blue: [[0, 0.5, 0.5], [0.11, 1, 1], [0.34, 1, 1], [0.65, 0, 0], [1, 0, 0]],
};

/** A 256-entry table for one colour channel, built as Matplotlib builds it. */
function channelTable(points, entries = 256) {
  const table = new Float64Array(entries);
  for (let i = 0; i < entries; i++) {
    const x = i / (entries - 1);
    let j = 1;
    while (j < points.length - 1 && points[j][0] < x) j++;
    const [x0, , y0] = points[j - 1];
    const [x1, y1] = points[j];
    table[i] = y0 + ((x - x0) / (x1 - x0)) * (y1 - y0);
  }
  table[0] = points[0][2];
  table[entries - 1] = points[points.length - 1][1];
  return table;
}

export const JET_TABLE = ["red", "green", "blue"].map((channel) => channelTable(JET_POINTS[channel]));

/** The jet colour (red, green, blue in [0, 1]) for a heatmap value in [0, 1]. */
export function jet(value) {
  const i = Math.min(255, Math.max(0, Math.trunc(f32(value * 256))));
  return [JET_TABLE[0][i], JET_TABLE[1][i], JET_TABLE[2][i]];
}

/** The tumour's border: tumour pixels within `rounds` pixels of non-tumour (repeated erosion). */
export function outline(mask, width, height, rounds = OUTLINE_WIDTH) {
  const inside = Uint8Array.from(mask, (v) => (v > 0 ? 1 : 0));
  let core = inside;
  for (let r = 0; r < rounds; r++) {
    const next = new Uint8Array(core.length);
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const i = y * width + x;
        // A pixel stays only if it and its four neighbours are all tumour (off the image counts as not).
        next[i] = core[i] && y > 0 && core[i - width] && y < height - 1 && core[i + width]
          && x > 0 && core[i - 1] && x < width - 1 && core[i + 1] ? 1 : 0;
      }
    }
    core = next;
  }
  return inside.map((v, i) => (v && !core[i] ? 1 : 0));
}

/** app/demo.py's overlay: 55% grey scan + 45% jet heatmap, lime outline on top. RGBA, ready for a canvas. */
export function overlay(pixels, heatmap, mask = null) {
  const rgba = new Uint8ClampedArray(pixels.length * 4);
  const border = mask ? outline(mask, SIZE, SIZE) : null;
  const scanShare = f32(1 - HEAT_ALPHA); // NumPy does this part in 32-bit floats
  for (let i = 0; i < pixels.length; i++) {
    const grey = f32(scanShare * f32(pixels[i] / 255));
    const colour = border && border[i]
      ? OUTLINE_COLOUR
      : jet(heatmap[i]).map((h) => roundHalfEven((grey + HEAT_ALPHA * h) * 255));
    rgba.set([...colour, 255], 4 * i);
  }
  return rgba;
}

/** The text above the confidence bars (app/demo.py's describe). */
export function describe(predicted, confidence, example) {
  const headline = `Prediction: ${predicted} (${(confidence * 100).toFixed(1)}% confidence)`;
  if (!example) {
    return { headline, lines: ["Uploaded image: no tumour outline is available, so only the heatmap is shown."] };
  }
  const verdict = predicted === example.true ? "correct" : "a mistake";
  return {
    headline,
    lines: [
      `Example from test patient ${example.patient_id}. True type: ${example.true}, so this answer is ${verdict}. `
        + "The green line is the clinicians' tumour outline.",
      example.note,
    ],
  };
}
