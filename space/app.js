// The page's wiring: read images, run the model, show the results.
// All the image arithmetic lives in pipeline.js, which is tested against the Python code.

import {
  CLASS_NAMES, SIZE, describe, findExample, normaliseHeatmap, overlay, preprocess, softmax, toGrey,
} from "./pipeline.js";

// ort (ONNX Runtime Web) is loaded by index.html. Its WebAssembly files come from the same CDN.
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1; // a static Space can't send the headers that multi-threading needs

const page = {
  status: document.getElementById("status"),
  upload: document.getElementById("upload"),
  examples: document.getElementById("examples"),
  summary: document.getElementById("summary"),
  bars: document.getElementById("bars"),
  heatmap: document.getElementById("heatmap"),
  heatmapCaption: document.getElementById("heatmap-caption"),
};

/** Decode an image file to grey levels, exactly as stored: no colour-profile conversion. */
async function readImage(blob) {
  const bitmap = await createImageBitmap(blob, { colorSpaceConversion: "none", premultiplyAlpha: "none" });
  const canvas = document.createElement("canvas");
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, bitmap.width, bitmap.height);
  return { grey: toGrey(data, bitmap.width, bitmap.height), width: bitmap.width, height: bitmap.height };
}

async function fetchBlob(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url} (${response.status})`);
  return response.blob();
}

/** The six example scans, with their outlines, from examples.json. */
async function loadExamples() {
  const list = await (await fetch("examples.json")).json();
  return Promise.all(list.map(async (example) => {
    const url = `examples/${example.name}.png`;
    const scan = await readImage(await fetchBlob(url));
    const mask = await readImage(await fetchBlob(`examples/${example.name}_mask.png`));
    return { ...example, url, scan: scan.grey, mask: mask.grey.map((v) => (v > 127 ? 1 : 0)) };
  }));
}

function showExamples(examples, onPick) {
  for (const example of examples) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "example";
    const image = document.createElement("img");
    image.src = example.url;
    image.alt = example.caption;
    const caption = document.createElement("span");
    caption.textContent = example.caption;
    button.append(image, caption);
    button.addEventListener("click", () => onPick(example));
    page.examples.append(button);
  }
}

function paragraph(text) {
  const p = document.createElement("p");
  p.textContent = text;
  return p;
}

function bar(name, probability) {
  const row = document.createElement("div");
  row.className = "bar";
  const track = document.createElement("span");
  track.className = "track";
  const fill = document.createElement("span");
  fill.className = "fill";
  fill.style.width = `${(probability * 100).toFixed(1)}%`;
  track.append(fill);
  const label = document.createElement("span");
  label.textContent = name;
  const value = document.createElement("span");
  value.textContent = `${Math.round(probability * 100)}%`;
  row.append(label, track, value);
  return row;
}

function showResult(probabilities, text, picture) {
  const headline = paragraph("");
  const strong = document.createElement("strong");
  strong.textContent = text.headline;
  headline.append(strong);
  page.summary.replaceChildren(headline, ...text.lines.map(paragraph));

  const ranked = CLASS_NAMES.map((name, i) => [name, probabilities[i]]).sort((a, b) => b[1] - a[1]);
  page.bars.replaceChildren(...ranked.map(([name, probability]) => bar(name, probability)));

  page.heatmap.getContext("2d").putImageData(new ImageData(picture, SIZE, SIZE), 0, 0);
  page.heatmap.hidden = false;
  page.heatmapCaption.hidden = false;
}

async function main() {
  const [session, examples] = await Promise.all([ort.InferenceSession.create("model.onnx"), loadExamples()]);

  // One model run at a time, in the order they were asked for.
  let queue = Promise.resolve();
  function runModel(scan) {
    const result = queue.then(() => session.run({ scan }));
    queue = result.catch(() => {});
    return result;
  }

  // Each click or upload gets a number; only the newest may show its result, so a
  // slow earlier run can't overwrite a later one.
  let latest = 0;

  async function analyse(getImage) {
    const request = ++latest;
    page.status.textContent = "Running the model…";
    try {
      const { grey, width, height } = await readImage(await getImage());
      const example = findExample(grey, width, height, examples);
      const pixels = example ? example.scan : preprocess(grey, width, height);
      const scan = new ort.Tensor("float32", Float32Array.from(pixels, (p) => p / 255), [1, 1, SIZE, SIZE]);
      const { scores, heatmaps } = await runModel(scan);
      if (request !== latest) return; // a newer click or upload has taken over
      const probabilities = softmax(Array.from(scores.data));
      const predicted = probabilities.indexOf(Math.max(...probabilities));
      const heatmap = normaliseHeatmap(heatmaps.data, predicted);
      showResult(probabilities, describe(CLASS_NAMES[predicted], probabilities[predicted], example),
        overlay(pixels, heatmap, example ? example.mask : null));
      page.status.textContent = "";
    } catch (error) {
      if (request === latest) page.status.textContent = `Something went wrong: ${error.message}`;
    }
  }

  showExamples(examples, (example) => analyse(() => fetchBlob(example.url)));
  page.upload.addEventListener("change", () => {
    const file = page.upload.files[0];
    if (file) analyse(async () => file);
  });
  page.upload.disabled = false;
  page.status.textContent = "Ready. Upload a slice or click an example.";
}

main().catch((error) => {
  page.status.textContent = `Could not start the demo: ${error.message}`;
});
