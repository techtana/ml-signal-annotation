# ml-cnn-annotation-tool

A **pure static web app** for annotating multi-channel time-series traces and training a CNN regressor — no server, no Python, no install required.

Open any of the HTML files directly in a browser, or serve the repo root with any static host (GitHub Pages, Netlify, Cloudflare Pages, `python -m http.server`, etc.).

---

## Workflow

1. **Source** (`source.html`) — select a bundled sample CSV or drop your own trace file.
2. **Annotate** (`annotate.html`) — click on the Plotly chart to label the event position for each sample. Labels are saved to `localStorage`; the trace CSV is never modified.
3. **Train** (`train.html`) — runs the CNN regressor in your browser via TensorFlow.js (WebGL). The trained model is stored in IndexedDB.
4. **Predict** (`predict.html`) — drop the downloaded `model.json` + `model.weights.bin` (or use the browser model directly) and run inference on any trace CSV.
5. **Settings** (`settings.html`) — configure column names and training hyperparameters. Saved to `localStorage`.

---

## File layout

```
index.html          landing page
source.html         trace file selection
annotate.html       interactive labeling (SPA — no page reloads between samples)
train.html          browser-side CNN training
predict.html        browser-side inference
settings.html       configuration

css/
  app.css           shared styles

js/
  store.js          localStorage + IndexedDB helpers  (window.CnnStore)
  csv-parser.js     client-side CSV parser            (window.CsvParser)
  annotate.js       SPA annotation logic
  train.js          TF.js training pipeline
  predict.js        TF.js inference pipeline

data/
  traces/           sample trace CSVs (served as static assets)
  prediction_template.csv
```

---

## Storage

| Data | Where |
|---|---|
| Active trace CSV | IndexedDB (`cnn-db / traces / 'active'`) |
| Annotations | `localStorage['cnn-annotations']` — `{ [filename]: { [sampleKey]: label } }` |
| Annotation state (current sample index) | `localStorage['cnn-annotation-state']` |
| Trained model | IndexedDB via TF.js (`indexeddb://cnn-latest`) |
| Model metadata + loss history | `localStorage['cnn-model-meta']`, `localStorage['cnn-train-loss']` |
| Config | `localStorage['cnn-config']` |

Clearing browser storage (DevTools → Application → Clear site data) resets everything to defaults.

---

## Trace CSV format

One row per time step:

```
run_id,elapsed_time,channel_1,channel_2
1,0.0,0.44,0.49
1,0.5,0.46,0.51
2,0.0,0.38,0.41
```

- **Group-by column** (default `run_id`) — groups rows into individual samples.
- **Time column** (default `elapsed_time`) — used as the x-axis; rows are sorted by this column.
- **Channel columns** — all remaining columns (or configured explicitly in Settings).

The column names are configurable in Settings and default to the values above.

---

## CNN model

Input shape: `(time_steps, num_channels, 1)`

```
Input → Conv2D(32, kernel=(5,1)) → MaxPool(2,1)
      → Conv2D(64, kernel=(5,1)) → MaxPool(2,1)
      → Flatten → Dense(max_len, relu) → Dense(1, linear)
```

- Kernels are `(5, 1)` — convolve along the time axis only.
- Loss: MSE. Output: normalized scalar scaled back to physical time units.
- Preprocessing: trim `trim_ratio` fraction from each end, then per-sample min-max normalization, then pad/clip to `max_len`.

---

## Sample data

Five generated datasets in `data/traces/`:

| File | Runs | Steps | Channels |
|---|---|---|---|
| `sample_a_traces.csv` | 10 | 100 | 3 |
| `sample_b_traces.csv` | 25 | 150 | 5 |
| `sample_c_traces.csv` | 40 | 200 | 2 |
| `sample_d_traces.csv` | 15 | 80  | 7 |
| `sample_e_traces.csv` | 60 | 120 | 4 |

Each trace has a sigmoid-shaped event at a random position (30–70 % through the trace) with Gaussian noise per channel.

---

## Deployment

No build step needed. Push the repo root to any static host:

```bash
# GitHub Pages — push to gh-pages branch or configure /root in repo settings

# Netlify / Cloudflare Pages — connect repo, set publish directory to "."

# Local dev
python -m http.server 8000
# then open http://localhost:8000
```

---

## Dependencies (all CDN, no install)

| Library | Version | Used by |
|---|---|---|
| Bootstrap | 5.3.3 | All pages |
| TensorFlow.js | 4.20.0 | train.html, predict.html |
| Plotly.js | 2.32.0 | annotate.html, train.html, predict.html |
