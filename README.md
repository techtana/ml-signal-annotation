# ML Annotation & Regression Pipeline

A toolkit for interactive time-series annotation and multi-source tabular regression modeling. Designed for workflows where sensor or process data needs to be labeled and then used to train predictive models.

---

## Overview

Two loosely coupled components:

**Component 1 — Interactive Annotation + CNN Regressor**  
Annotate multi-channel time-series samples by clicking on a plot, then train a CNN to predict the labeled position automatically. Available as both a Flask web app (browser-based, no matplotlib required) and as standalone CLI scripts.

**Component 2 — Tabular Regression Pipeline**  
Query data from multiple sources, merge and engineer features, train PLS regression models per job, and browse results via a Flask dashboard.

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
├── annotation/
│   ├── interactive.py           # InteractiveAnnotation_2dplot / _heatmap (matplotlib)
│   └── annotate.py              # standalone annotation CLI (no training)
│
├── modeling/
│   ├── pipeline.py              # tabular regression pipeline (Component 2)
│   └── import_config.py         # bulk-import job definitions from CSV
│
├── utils/
│   ├── preprocess.py            # normalize, equalize_length, plot_training_history
│   ├── io.py                    # JSON config helpers, section printer
│   └── spark.py                 # PySpark / Hive query utilities
│
├── data/
│   ├── traces/                  # input trace CSVs (read-only during the workflow)
│   ├── annotations/             # one annotation CSV per trace, never merged back
│   └── predictions/             # model output CSVs
│
├── artifacts/                   # saved Keras models + app config
├── train.py                     # CLI: annotate + train in one pass
├── requirements.txt
└── config.json                  # Component 2 job definitions
```

---

## Component 1: Interactive Annotation & CNN Regressor

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
| `/cnn/train` | Run CNN training in-process; shows annotated count + training summary |
| `/cnn/predict` | Run inference with a saved model; toggle between result table and trace plot |
| `/cnn/settings` | Edit all pipeline parameters (paths, hyperparameters, GPU options) |
| `/cnn/download-sample` | Download a ZIP of sample trace + annotation CSVs |

### GPU configuration

Settings page exposes two fields:

| Setting | Default | Behaviour |
|---|---|---|
| `use_gpu` | `True` | If `False`, TensorFlow hides all GPUs; trains on CPU |
| `gpu_memory_fraction` | `None` (blank) | `None` = memory growth on demand; `0.8` = reserve 80 % of VRAM upfront |

On **native Windows**, install `tensorflow-directml-plugin` (see setup above). On Linux/WSL2, standard CUDA works with no extra steps.

### Standalone CLI (no browser needed)

**Annotation only:**

```bash
python -m annotation.annotate
```

A single matplotlib window opens and reuses the same figure for every sample. A dashed vertical line follows the mouse; click anywhere on the x-axis to record the label and advance to the next sample. Edit parameters at the top of [annotation/annotate.py](annotation/annotate.py).

**Annotate + train in one pass:**

```bash
python train.py
```

Edit parameters at the top of [train.py](train.py):

| Parameter | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/traces.csv` | Input time-series CSV |
| `ANNOTATION_PATH` | `data/annotations.csv` | Existing annotation file to load |
| `OUTPUT_PATH` | `data/output.csv` | Unified output (traces + annotations + predictions) |
| `CACHE_PATH` | `data/.session_cache.pkl` | Session cache — delete to reset |
| `GROUP_BY_COL` | `run_id` | Column used to group samples |
| `TIME_INDEX_COL` | `elapsed_time` | Time / x-axis column |
| `CHANNEL_COLS` | `None` | Channel columns; `None` = all non-index columns |
| `ANNOTATE_COUNT` | `100` | Samples to annotate; `None` = all |
| `TRIM_RATIO` | `0.1` | Fraction trimmed from each end before display |
| `TEST_SIZE` | `0.20` | Validation split fraction |
| `BATCH_SIZE` | `200` | Training batch size |
| `NUM_EPOCHS` | `100` | Training epochs |
| `USE_GPU` | `True` | Use GPU if available; falls back to CPU automatically |
| `GPU_MEMORY_FRACTION` | `None` | VRAM fraction to reserve; `None` = dynamic growth |

The CLI keeps everything in memory and writes a single unified output CSV (`OUTPUT_PATH`) containing the original trace rows plus `annotation` and `predicted_position` columns. A pickle cache (`CACHE_PATH`) preserves annotations and predictions between runs — delete it to start fresh.

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

---

## Component 2: Tabular Regression Pipeline

### What it does

1. Reads `config.json` — a list of modeling job definitions
2. Validates that specified data items exist in the configured databases
3. Queries and merges data from multiple sources (target measurements, features, run parameters, sensor summaries)
4. Engineers features, pivots multi-step sensor data into a wide feature matrix, filters noisy columns
5. Trains a PLS regression model per job (optionally split by a grouping column)
6. Exports models, metrics, and plots; writes file paths and status back into `config.json` for resumable runs

### Running the pipeline

```bash
python -m modeling.pipeline [config.json] [--verbose] [--reset] [--reset-model]
```

| Flag | Effect |
|---|---|
| `--verbose` | Print detailed progress to stdout |
| `--reset` | Re-query all data and retrain from scratch |
| `--reset-model` | Retrain using cached data (skip re-query) |

### Config reference (`config.json`)

Each object in the JSON array defines one job.

| Field | Type | Description |
|---|---|---|
| `status` | str | Lifecycle state: `""` (new), `"success"`, `"reset"`, `"skip"`, `"failed ..."` |
| `start_date` | str | Earliest data date to query (`YYYY-MM-DD`) |
| `source_tool` | str | Tool or equipment identifier (regex supported) |
| `source_name` | str | Recipe or process name pattern (regex supported) |
| `source_step` | str | Workflow step identifier (regex supported) |
| `sensor_aggregation` | list | Aggregation types: `["mean", "max", "min", "stddev", "range"]` |
| `target_items` | list | Target (Y) items — each `{source_step, name, limits}` |
| `feature_items` | list | Additional tabular X columns to include |
| `run_params_enable` | int | `1` to include run-level recipe parameters as features |
| `modelgroup` | str | Column to split modeling into subgroups (`""` = one model) |
| `data_conversion` | dict | Regex → conversion type (e.g. `{"^run_param::(.*)Time": "hms_to_seconds"}`) |
| `position_type` | str | How to interpret position data: `"chamber"` or `"slot"` |
| `position_column` | str | Column name that holds the position identifier |

After a successful run, the pipeline writes back parquet paths, model path, and metrics (`R2`, `RMSE`, `N`, `Y-Range`) directly into `config.json`.

### Flask dashboard

```bash
FLASK_APP=app/app.py flask run --host 0.0.0.0 --port 8972
```

- `/pipeline/jobs` — table of all jobs with status
- `/pipeline/job?id=<N>` — full config JSON for job N

### PySpark / database setup

Set environment variables before running:

```bash
export MEASUREMENT_DB=your_measurement_database
export SENSOR_DB=your_sensor_database
```

Create `hive_schema.json` at the project root mapping `{db: {table: [columns]}}`. Adapt table and column names in [utils/spark.py](utils/spark.py) to match your actual schema.

---

## Data flow (Component 2)

```
[Target measurements]     ──┐
[Run-level parameters]    ──┼──► merge ──► feature engineering ──► pivot ──► filter ──► PLS model
[Wafer/item features]     ──┤
[Step-level sensor data]  ──┘
```

Intermediate DataFrames are cached as Parquet. File paths are stored back in `config.json` so subsequent runs skip re-querying unchanged data.

---

## Dependencies

```
pip install -r requirements.txt
```

| Package | Used by |
|---|---|
| `tensorflow >= 2.13` | CNN model (train.py, app) |
| `keras` | included with TensorFlow |
| `scikit-learn` | normalization, train/test split |
| `pandas`, `numpy` | data handling throughout |
| `matplotlib` | CLI annotation UI (annotate.py, train.py) |
| `flask >= 3.0` | web app |
| `pyarrow >= 13` | Parquet I/O in the regression pipeline |
| `pyspark >= 3.4` | Hive/Hadoop queries (Component 2 only) |

Optional:

```bash
pip install tensorflow-directml-plugin   # GPU on native Windows
```
