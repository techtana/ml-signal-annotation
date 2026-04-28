# ML Annotation & Regression Pipeline

A toolkit for interactive time-series annotation and multi-source tabular regression modeling. Designed for workflows where sensor or process data needs to be labeled and then used to train predictive models.

---

## Overview

Two loosely coupled components:

**Component 1 — Interactive Annotation + CNN Regressor**  
Annotate multi-channel time-series samples by clicking on a plot, then train a CNN to predict the labeled position automatically.

**Component 2 — Tabular Regression Pipeline**  
Query data from multiple sources, merge and engineer features, train PLS regression models per job, and browse results via a Flask dashboard.

---

## Project Structure

```
├── annotation/
│   ├── interactive.py       # interactive matplotlib annotation classes
│   └── annotate.py          # annotation-only script (no training)
├── modeling/
│   ├── pipeline.py          # full tabular regression pipeline
│   └── import_config.py     # bulk-import job definitions from CSV
├── dashboard/
│   ├── app.py               # Flask results dashboard
│   └── templates/
│       ├── entry.html
│       └── table.html
├── utils/
│   ├── preprocess.py        # normalization, feature engineering, plotting
│   ├── io.py                # JSON config helpers, section printer
│   └── spark.py             # PySpark / Hive query utilities
├── notebooks/               # prototype notebooks
├── train.py                 # full annotation + CNN training script
└── config.json              # pipeline job config
```

---

## Component 1: Interactive Annotation & CNN Regressor

### What it does

Given a collection of multi-channel time-series samples (e.g., multiple sensor channels recorded over time for each experiment run), this component lets you:

- Visualize each trace interactively
- Click on the plot to mark an event or endpoint for that trace
- Save all annotations to a CSV
- Train a CNN regressor to predict the labeled position from the raw trace

Typical use cases: detecting process endpoints, transition points, anomaly onset, or any time-indexed event in sensor data.

### Quick start

**Step 1 — Prepare your data**

Your input CSV needs:
- A run/experiment ID column (e.g., `run_id`)
- A time index column (e.g., `elapsed_time`)
- One or more sensor/channel columns

```
run_id, elapsed_time, channel_1, channel_2, ..., channel_N
```

**Step 2 — Annotation only**

Edit the parameters at the top of [annotation/annotate.py](annotation/annotate.py) and run:

```bash
python -m annotation.annotate
```

A single matplotlib window opens and is reused for every sample — the plot updates in-place rather than closing and reopening. Click anywhere on the x-axis to mark the event position and advance to the next sample. If `OUTPUT_PATH` already exists on disk the annotation step is skipped entirely and the existing file is used.

**Step 3 — Annotate + train in one pass**

Edit the parameters at the top of [train.py](train.py) and run:

```bash
python train.py
```

This annotates (or loads existing annotations), trains a CNN, and saves per-sample predictions to `OUTPUT_PATH`. The model is saved under `artifacts/`.

### Parameters (`train.py`)

| Parameter | Default | Description |
|---|---|---|
| `DATA_PATH` | `data/traces.csv` | Input time-series CSV |
| `ANNOTATION_PATH` | `data/annotations.csv` | Annotation file to write or load |
| `OUTPUT_PATH` | `data/predictions.csv` | Per-sample prediction output |
| `GROUP_BY_COL` | `run_id` | Column used to group samples |
| `TIME_INDEX_COL` | `elapsed_time` | Time / x-axis column |
| `CHANNEL_COLS` | `None` | Channel columns; `None` = all non-index columns |
| `ANNOTATE_COUNT` | `100` | Samples to annotate; `None` = all |
| `TRIM_RATIO` | `0.1` | Fraction trimmed from each end before display |
| `TEST_SIZE` | `0.20` | Validation split fraction |
| `BATCH_SIZE` | `200` | Training batch size |
| `NUM_EPOCHS` | `100` | Training epochs |
| `USE_GPU` | `True` | Use GPU if available; falls back to CPU automatically |
| `GPU_MEMORY_FRACTION` | `None` | VRAM fraction to reserve (e.g. `0.8`); `None` = dynamic growth |

### GPU configuration

`configure_gpu()` is called before any Keras operations and handles three cases:

| `USE_GPU` | GPU present | Behaviour |
|---|---|---|
| `False` | — | GPU hidden from TensorFlow; trains on CPU |
| `True` | No | Prints a warning and falls back to CPU |
| `True` | Yes | Enables memory growth (`GPU_MEMORY_FRACTION=None`) or reserves a fixed VRAM slice |

`GPU_MEMORY_FRACTION=None` (the default) lets TensorFlow allocate memory on demand — recommended on shared machines. Set it to a value like `0.8` to pre-reserve 80 % of VRAM for more deterministic performance on a dedicated GPU.

### Annotation classes (`annotation/interactive.py`)

**`InteractiveAnnotation_2dplot(data, plottitle=None)`**  
Multi-line 2D plot. Click → integer x index returned.

**`InteractiveAnnotation_heatmap(data)`**  
Heatmap view. Click → float x value returned.

Both expose `.annotate(fig=None, ax=None)`. When `fig` and `ax` are omitted a standalone window is created and closed after the click. When passed in, the axes are updated in-place and the window stays open — this is how the annotation loop in `annotate.py` and `train.py` keeps a single persistent window.

### CNN architecture

Input shape: `(time_steps, num_channels, 1)`

```
Conv2D(32, kernel=(5,1)) → MaxPool → Conv2D(64, kernel=(5,1)) → MaxPool → Flatten → Dense → Dense(1, linear)
```

Loss: MSE. Output: normalized scalar in `[0, 1]`, scaled back to physical units using the time-index range.

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

Each object in the JSON array defines one job. Edit the example `config.json` to match your data sources.

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
| `modelgroup` | str | Column name to split modeling into subgroups (`""` = one model) |
| `data_conversion` | dict | Regex → conversion type (e.g. `{"^run_param::(.*)Time": "hms_to_seconds"}`) |
| `position_type` | str | How to interpret position data: `"chamber"` or `"slot"` |
| `position_column` | str | Column name that holds the position identifier |

After a successful run, the pipeline writes back:
- `TargetDF`, `FeatureDF`, `RunParamDF`, `SensorDF`, `MergedDF`, `ProcessedDF`, `PivotedDF`, `FilteredDF` — parquet file paths
- `ModelFile` — pickled PLS model path
- `R2`, `RMSE`, `N`, `Y-Range` — model metrics
- `plots` — list of actual-vs-predicted plot paths

### Adding jobs in bulk

Populate a CSV with one row per job (columns matching `modeling/import_config.py`), then:

```bash
python -m modeling.import_config
```

### Flask dashboard

```bash
export FLASK_APP=app/app.py
flask run --host=0.0.0.0 --port=8972
```

- `/jobs` — table of all jobs with status and error fields
- `/job?id=<N>` — full config JSON for job at index N

---

## PySpark / Database Setup (`utils/spark.py`)

The `SparkSession` wrapper assumes your data is in Hive/Hadoop tables. Before using the pipeline:

1. Set environment variables (or edit `modeling/pipeline.py` directly):
   ```bash
   export MEASUREMENT_DB=your_measurement_database
   export SENSOR_DB=your_sensor_database
   ```

2. Create `hive_schema.json` at the project root mapping `{db: {table: [columns]}}` — the pipeline uses this to validate context keys before building queries.

3. Adapt the table and column names in `utils/spark.py` to match your actual schema. The generic names used internally are:
   - `measurement_wafer_summary`, `measurement_summary`, `measurement_run_summary`, `measurement_items`
   - `sensor_summary`, `sensor_point`, `tool_registry`, `run_context`
   - Key columns: `batch_id`, `item_id`, `run_date`, `workflow_step`, `data_item_id`, `value`, `RECIPE_NAME`

---

## Data Flow (Component 2)

```
[Target measurements]     ──┐
[Run-level parameters]    ──┼──► merge ──► feature engineering ──► pivot ──► filter ──► PLS model
[Wafer/item features]     ──┤
[Step-level sensor data]  ──┘
```

Intermediate DataFrames are cached as Parquet. File paths are stored back in `config.json` so subsequent runs skip re-querying unchanged data (controlled by `extract_date` and a 14-day reset window).

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
keras / tensorflow
flask
pyspark
pyarrow
```

Install core dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib keras flask pyarrow
```
