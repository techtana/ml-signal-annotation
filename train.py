"""
Full pipeline: annotate time-series samples, then train a CNN regressor
to predict the labeled event position automatically.

Edit the Parameters section before running.
"""

import os
import pickle
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from annotation.interactive import InteractiveAnnotation_2dplot
from utils.preprocess import normalize, equalize_length, plot_training_history


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DATA_PATH        = "data/traces.csv"          # input CSV (read-only — never modified)
ANNOTATION_PATH  = "data/annotations.csv"     # optional pre-existing annotation file to load
OUTPUT_PATH      = "data/output.csv"          # unified output: traces + annotations + predictions
CACHE_PATH       = "data/.session_cache.pkl"  # local session cache; delete to reset

GROUP_BY_COL     = "run_id"                   # column used to group samples
TIME_INDEX_COL   = "elapsed_time"             # x-axis / time column
CHANNEL_COLS     = None                       # signal channel columns; None = all others

ANNOTATE_COUNT   = 100    # samples to annotate; None = all
TRIM_RATIO       = 0.1    # fraction trimmed from each end before display/training
TEST_SIZE        = 0.20   # fraction held out for validation
RANDOM_STATE     = 42
BATCH_SIZE       = 200
NUM_EPOCHS       = 100
VERBOSE          = True

# Prediction display
# "table" — print a summary table of all predicted values
# "plot"  — show a grid of N traces with predicted position overlaid
PREDICT_DISPLAY   = "table"
PREDICT_DISPLAY_N = 9        # number of traces shown in "plot" mode

# GPU settings
# Set to True to use GPU if available, False to force CPU.
# When True and no GPU is found, falls back to CPU automatically.
USE_GPU          = True
# Fraction of GPU memory to pre-allocate (0.0–1.0).
# None lets TensorFlow grow memory dynamically (recommended for shared machines).
GPU_MEMORY_FRACTION = None
# ---------------------------------------------------------------------------


def _load_cache() -> dict:
    """Load the local session cache (annotations + predictions)."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as f:
            return pickle.load(f)
    return {}


def _save_cache(cache: dict):
    """Persist the session cache to disk."""
    os.makedirs(os.path.dirname(CACHE_PATH) or ".", exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)


def load_and_group(path: str, group_col: str, time_col: str, channel_cols):
    df = pd.read_csv(path)
    if time_col in df.columns:
        df = df.sort_values(time_col)
    if channel_cols is None:
        channel_cols = [c for c in df.columns if c not in (group_col, time_col)]

    max_len = 0
    groups = {}
    for key, grp in df.groupby(group_col):
        trace = grp[channel_cols + [time_col]].set_index(time_col).iloc[1:]
        groups[key] = trace
        max_len = max(max_len, len(trace))

    return groups, channel_cols, max_len


def run_annotation(groups: dict, keys: list, trim_ratio: float, count: int | None) -> pd.DataFrame:
    """Interactively annotate samples in a single persistent window.

    Returns a DataFrame with columns [GROUP_BY_COL, 'label'].
    Does not read or write any file — callers manage persistence via cache.
    """
    total = count if count is not None else len(keys)
    annotations = []

    fig, ax = plt.subplots(figsize=(12, 8))

    for i, key in enumerate(keys):
        if count is not None and i >= count:
            break
        trace = groups[key]
        trim  = int(len(trace) * trim_ratio)
        trimmed = normalize(trace.iloc[trim: len(trace) - trim])

        label = InteractiveAnnotation_2dplot(
            trimmed,
            plottitle=f"Sample {key}  ({i + 1} / {total})"
        ).annotate(fig=fig, ax=ax)

        if label is None:
            print(f"  sample {key} skipped (window closed)")
            continue

        annotations.append([key, label])
        print(f"  label = {label}")

    plt.close(fig)
    return pd.DataFrame(annotations, columns=[GROUP_BY_COL, "label"])


def build_datasets(groups: dict, keys: list, df_annotations: pd.DataFrame,
                   max_len: int, trim_ratio: float, test_size: float, random_state: int):
    processed = {}
    for key in keys:
        trace = groups[key]
        trim = int(len(trace) * trim_ratio)
        processed[key] = equalize_length(normalize(trace.iloc[trim: len(trace) - trim]), max_len)

    labels = {}
    for _, row in df_annotations.iterrows():
        key = row[GROUP_BY_COL]
        raw_label = row["label"]
        idx = int(np.argmin(np.abs(processed[key].index - raw_label)))
        labels[key] = idx / max_len

    annotated_keys = [k for k in keys if k in labels]
    train_keys, test_keys = train_test_split(annotated_keys, test_size=test_size, random_state=random_state)

    def to_array(ks):
        return np.expand_dims(np.array([processed[k].values for k in ks]), axis=3)

    X_train = to_array(train_keys)
    y_train = np.array([labels[k] for k in train_keys])
    X_test  = to_array(test_keys)
    y_test  = np.array([labels[k] for k in test_keys])
    X_all   = to_array(keys)

    return X_train, y_train, X_test, y_test, X_all, processed


def configure_gpu():
    """Configure TensorFlow GPU usage based on USE_GPU / GPU_MEMORY_FRACTION."""
    import sys

    # Must be set before TensorFlow is imported to suppress oneDNN verbose logs.
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if not USE_GPU or not gpus:
        if USE_GPU and not gpus:
            if sys.platform == "win32":
                print(
                    "No GPU detected on native Windows.\n"
                    "TensorFlow >= 2.11 dropped native Windows CUDA support.\n"
                    "Install 'tensorflow-directml-plugin' for GPU acceleration:\n"
                    "  pip install tensorflow-directml-plugin"
                )
            else:
                print("No GPU detected — falling back to CPU.")
        else:
            print("GPU disabled by USE_GPU=False — using CPU.")
        tf.config.set_visible_devices([], "GPU")
        return "/CPU:0"

    try:
        if GPU_MEMORY_FRACTION is None:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except (RuntimeError, ValueError):
                    pass  # DirectML and some backends don't support memory growth
            print(f"GPU enabled: {[g.name for g in gpus]}")
        else:
            limit_mb = int(
                GPU_MEMORY_FRACTION
                * _get_gpu_total_memory_mb(gpus[0])
            )
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)],
                )
                print(f"GPU enabled (memory limit {limit_mb} MB): {gpus[0].name}")
            except (RuntimeError, ValueError):
                print(f"GPU enabled (memory limit unsupported, running without cap): {gpus[0].name}")
    except RuntimeError as e:
        print(f"GPU configuration warning: {e}")

    return "/GPU:0"


def _get_gpu_total_memory_mb(gpu_device) -> int:
    """Return total VRAM in MB for the given PhysicalDevice, or 4096 as a fallback."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        values = [int(v.strip()) for v in result.stdout.strip().splitlines() if v.strip().isdigit()]
        return values[0] if values else 4096
    except Exception:
        return 4096


def _plot_predictions(processed: dict, keys: list,
                      prediction_dict: dict, annotation_dict: dict, n: int):
    """Show a grid of N sample traces with predicted and annotated positions overlaid."""
    sample_keys = random.sample(list(prediction_dict), min(n, len(prediction_dict)))
    ncols = min(3, len(sample_keys))
    nrows = (len(sample_keys) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, key in enumerate(sample_keys):
        ax = axes_flat[i]
        trace = processed[key]
        ax.plot(trace.index, trace.values, linewidth=0.8)
        ax.axvline(x=prediction_dict[key], color='red', linestyle='--',
                   linewidth=1.5, label='predicted')
        if key in annotation_dict:
            ax.axvline(x=annotation_dict[key], color='limegreen', linestyle='-',
                       linewidth=1.5, label='annotated')
        ax.set_title(str(key), fontsize=8)
        ax.set_xlabel(trace.index.name or "Time", fontsize=7)
        ax.tick_params(labelsize=6)
        if i == 0:
            ax.legend(fontsize=6)

    for j in range(len(sample_keys), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Predicted positions  (showing {len(sample_keys)} / {len(prediction_dict)})",
        fontsize=10
    )
    plt.tight_layout()
    plt.show()


def build_model(input_shape: tuple, max_len: int, device: str = "/CPU:0"):
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense

    with tf.device(device):
        model = Sequential([
            Input(shape=input_shape),
            Conv2D(32, kernel_size=(5, 1), strides=(1, 1), activation="relu"),
            MaxPooling2D(pool_size=(2, 1), strides=(2, 1)),
            Conv2D(64, kernel_size=(5, 1), activation="relu"),
            MaxPooling2D(pool_size=(2, 1), strides=(2, 1)),
            Flatten(),
            Dense(max_len, activation="relu"),
            Dense(1, activation="linear"),
        ])
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    return model


def main():
    # Configure GPU (or CPU fallback) before any Keras/TF ops
    device = configure_gpu()

    # Load source data — read-only, never modified
    df_raw = pd.read_csv(DATA_PATH)
    groups, channel_cols, max_len = load_and_group(
        DATA_PATH, GROUP_BY_COL, TIME_INDEX_COL, CHANNEL_COLS
    )
    keys = list(groups.keys())
    random.shuffle(keys)

    if VERBOSE:
        sample = normalize(
            equalize_length(
                groups[keys[0]].iloc[int(len(groups[keys[0]]) * TRIM_RATIO):-int(len(groups[keys[0]]) * TRIM_RATIO)],
                max_len
            )
        )
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(sample)
        ax.set_title("Sample trace preview")
        ax.set_xlabel("Time index")
        ax.set_ylabel("Normalized value")
        plt.show()
        print(f"Samples: {len(keys)} | Channels: {len(channel_cols)} | Max length: {max_len}")

    # Resolve annotations: session cache > ANNOTATION_PATH file > interactive
    cache = _load_cache()
    if 'annotations' in cache:
        df_annotations = pd.DataFrame(
            list(cache['annotations'].items()), columns=[GROUP_BY_COL, 'label']
        )
        print(f"Loaded {len(df_annotations)} annotations from session cache.")
    elif os.path.exists(ANNOTATION_PATH):
        df_annotations = pd.read_csv(ANNOTATION_PATH)
        cache['annotations'] = dict(zip(df_annotations[GROUP_BY_COL], df_annotations['label']))
        _save_cache(cache)
        print(f"Loaded {len(df_annotations)} annotations from '{ANNOTATION_PATH}'.")
    else:
        df_annotations = run_annotation(groups, keys, TRIM_RATIO, ANNOTATE_COUNT)
        cache['annotations'] = dict(zip(df_annotations[GROUP_BY_COL], df_annotations['label']))
        _save_cache(cache)

    # Build datasets
    X_train, y_train, X_test, y_test, X_all, processed = build_datasets(
        groups, keys, df_annotations, max_len, TRIM_RATIO, TEST_SIZE, RANDOM_STATE
    )
    if VERBOSE:
        print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    sample_trace = processed[keys[0]]
    time_scale = (sample_trace.index.max() - sample_trace.index.min()) / max_len
    if VERBOSE:
        print(f"1 time index = {time_scale:.4f} units")

    # Build and train model
    input_shape = (max_len, len(channel_cols), 1)
    model = build_model(input_shape, max_len, device=device)
    if VERBOSE:
        model.summary()

    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        verbose=int(VERBOSE),
        validation_data=(X_test, y_test),
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"artifacts/cnn_regressor_{timestamp}.keras"
    os.makedirs("artifacts", exist_ok=True)
    model.save(model_path)
    if VERBOSE:
        plot_training_history(history)

    # Predict all samples and cache predictions
    raw_preds = model.predict(X_all).flatten() * max_len * time_scale
    prediction_dict = {key: float(pred) for key, pred in zip(keys, raw_preds)}
    cache['predictions'] = prediction_dict
    _save_cache(cache)

    # Build unified output DataFrame: traces + annotations + predictions
    # The source CSV (DATA_PATH) is never touched.
    annotation_dict = cache['annotations']
    df_out = df_raw.copy()
    df_out['annotation'] = df_out[GROUP_BY_COL].map(annotation_dict)
    df_out['predicted_position'] = df_out[GROUP_BY_COL].map(prediction_dict)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Unified output saved to: {OUTPUT_PATH}  ({len(df_out)} rows)")

    if VERBOSE:
        if PREDICT_DISPLAY == "plot":
            _plot_predictions(processed, keys, prediction_dict, annotation_dict, PREDICT_DISPLAY_N)
        else:
            summary = (
                df_out[[GROUP_BY_COL, "annotation", "predicted_position"]]
                .drop_duplicates(subset=[GROUP_BY_COL])
                .reset_index(drop=True)
            )
            print(summary.to_string(index=False))
        print(f"Predicted {len(prediction_dict)} samples.")


if __name__ == "__main__":
    main()
