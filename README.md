# ML Annotation & Regression Pipeline

A Flask web app for interactive time-series annotation and CNN regression modeling. Annotate multi-channel time-series samples by clicking on a Plotly chart, then train a CNN to predict the labeled position automatically.

---

## Overview

**Interactive Annotation + CNN Regressor**

Annotate multi-channel time-series samples by clicking on a plot, then train a CNN to predict the labeled position automatically.

---

## First-time setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd ml-cnn-annotation-tool
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support on **native Windows** (TensorFlow ≥ 2.11 dropped CUDA on Windows):

```bash
pip install tensorflow-directml-plugin
```

This installs Microsoft's DirectML backend, which works on NVIDIA and AMD GPUs without WSL2. On Linux or WSL2, standard CUDA/TensorFlow GPU works out of the box.

### 3. Start the Flask web app

```bash
# Windows (PowerShell)
$env:FLASK_APP = "app/app.py"
flask run --host 0.0.0.0 --port 8972

# macOS / Linux
FLASK_APP=app/app.py flask run --host 0.0.0.0 --port 8972

# Or run directly
python app/app.py
```

Open **http://localhost:8972** in your browser.

### 4. Walk through the workflow

1. **Choose trace file** (`/cnn/source`) — select one of the bundled sample CSVs from `data/traces/` or upload your own.
2. **Annotate** (`/cnn/annotate`) — click on the Plotly chart to label the event position for each sample. Labels are saved to `data/annotations/` as a companion file; the original trace CSV is never modified.
3. **Train** (`/cnn/train`) — runs the CNN regressor in-process. Predictions are written to `data/predictions/`.
4. **Predict** (`/cnn/predict`) — load a saved model and run inference on any trace file. Toggle between a table and a per-sample trace plot with the predicted position overlaid.

---

## Project structure

```
├── app/                         # Flask web app (Component 1 UI)
│   ├── __init__.py              # create_app() factory
│   ├── app.py                   # entrypoint (flask run / python app.py)
│   ├── routes/
│   │   ├── cnn.py               # CNN workflow routes (/cnn/*)
│   │   ├── pipeline_dashboard.py # Pipeline jobs dashboard (/pipeline/*)
│   │   └── core.py              # Root redirect
│   ├── services/
│   │   ├── cnn_pipeline.py      # load_and_group, build_model, train_and_predict, predict_only
│   │   ├── annotations.py       # read/write annotation files (separate from trace CSVs)
│   │   ├── config_store.py      # persist CnnConfig to artifacts/web/cnn_config.json
│   │   ├── state.py             # annotation session state (current sample index)
│   │   └── trace_files.py       # temp-upload helpers
│   ├── templates/
│   │   ├── base.html            # Bootstrap 5 shell + navbar
│   │   ├── cnn/                 # source, annotate, train, predict, settings pages
│   │   └── pipeline/            # jobs, job detail pages
│   └── static/
│       ├── app.css
│       └── prediction_template.csv
│
├── utils/
│   └── preprocess.py            # normalize, equalize_length
│
├── data/
│   ├── traces/                  # input trace CSVs (read-only during the workflow)
│   ├── annotations/             # one annotation CSV per trace, never merged back
│   └── predictions/             # model output CSVs
│
├── artifacts/                   # saved Keras models + app config
├── requirements.txt
```

---

### Architecture

The Flask app is structured as a single-page-per-step workflow with three layers:

```
Browser (Plotly.js + Bootstrap 5)
    │  click events → POST /cnn/annotate/label (JSON)
    │  sample data  ← GET  /cnn/predict/sample-data (JSON)
    ▼
Routes (app/routes/cnn.py)
    │  thin controllers: validate input, call services, render templates
    ▼
Services (app/services/)
    ├── cnn_pipeline.py   — TensorFlow/Keras model logic, data loading
    ├── annotations.py    — annotation file I/O
    ├── config_store.py   — persistent settings (artifacts/web/cnn_config.json)
    └── state.py          — per-session annotation position (which sample is next)
```

**Data is kept in separate files by type:**

| Folder | Contents | Written by |
|---|---|---|
| `data/traces/` | Input CSVs — never modified | User / upload |
| `data/annotations/` | `{trace_stem}_annotations.csv` | Annotation step |
| `data/predictions/` | `{name}_predictions.csv` | Train / Predict step |
| `artifacts/` | `cnn_regressor_YYYYMMDD_HHMMSS.keras`, app config | Training step |

### CNN model

Input shape: `(time_steps, num_channels, 1)`

```
Input → Conv2D(32, kernel=(5,1)) → MaxPool(2,1)
      → Conv2D(64, kernel=(5,1)) → MaxPool(2,1)
      → Flatten → Dense(max_len, relu) → Dense(1, linear)
```

- Kernels are `(5, 1)` — they convolve along the time axis only, treating each channel independently.
- Pooling is `(2, 1)` for the same reason: time is downsampled, channel count is preserved.
- Loss: MSE. Output: normalized scalar scaled back to physical time units.

### Web app pages

| URL | Purpose |
|---|---|
| `/cnn/source` | Select or upload a trace CSV |
| `/cnn/annotate` | Interactive Plotly chart; click to label; Prev/Next/Skip navigation |
| `/cnn/train` | Run CNN training in-process; shows annotated count + training summary; button disabled while running to prevent double-submit |
| `/cnn/predict` | Run inference with a saved model; data path defaults to the CSV used during training; live compatibility badge checks channel count and column names before running; toggle between result table and per-sample trace plot |
| `/cnn/settings` | Edit all pipeline parameters (paths, hyperparameters, GPU options) |
| `/cnn/download-sample` | Download a ZIP of sample trace + annotation CSVs |

### GPU configuration

Settings page exposes two fields:

| Setting | Default | Behaviour |
|---|---|---|
| `use_gpu` | `True` | If `False`, TensorFlow hides all GPUs; trains on CPU |
| `gpu_memory_fraction` | `None` (blank) | `None` = memory growth on demand; `0.8` = reserve 80 % of VRAM upfront |

On **native Windows**, install `tensorflow-directml-plugin` (see setup above). On Linux/WSL2, standard CUDA works with no extra steps.

### Sample data

Five generated datasets in `data/traces/`:

| File | Runs | Steps | Channels | Pre-annotated |
|---|---|---|---|---|
| `sample_a_traces.csv` | 10 | 100 | 3 | No |
| `sample_b_traces.csv` | 25 | 150 | 5 | Yes (15 samples) |
| `sample_c_traces.csv` | 40 | 200 | 2 | No |
| `sample_d_traces.csv` | 15 | 80 | 7 | Yes (10 samples) |
| `sample_e_traces.csv` | 60 | 120 | 4 | No |

Pre-annotated companion files are in `data/annotations/`. Each trace has a sigmoid-shaped event at a random position (30–70 % through the trace) with Gaussian noise per channel.


## Dependencies

```
pip install -r requirements.txt
```

| Package | Used by |
|---|---|
| `tensorflow >= 2.13` | CNN model (includes Keras) |
| `scikit-learn` | normalization, train/test split |
| `pandas`, `numpy` | data handling throughout |
| `flask >= 3.0` | web app |

Optional:

```bash
pip install tensorflow-directml-plugin   # GPU on native Windows
```
