"""
Full pipeline: annotate time-series samples, then train a CNN regressor
to predict the labeled event position automatically.

Edit the Parameters section before running.
"""

import os
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
DATA_PATH        = "data/traces.csv"        # input CSV
ANNOTATION_PATH  = "data/annotations.csv"   # where to write / read annotations
OUTPUT_PATH      = "data/predictions.csv"   # final per-sample predictions

GROUP_BY_COL     = "run_id"                 # column used to group samples
TIME_INDEX_COL   = "elapsed_time"           # x-axis / time column
CHANNEL_COLS     = None                     # signal channel columns; None = all others

ANNOTATE_COUNT   = 100    # samples to annotate; None = all
TRIM_RATIO       = 0.1    # fraction trimmed from each end before display/training
TEST_SIZE        = 0.20   # fraction held out for validation
RANDOM_STATE     = 42
BATCH_SIZE       = 200
NUM_EPOCHS       = 100
VERBOSE          = True

# GPU settings
# Set to True to use GPU if available, False to force CPU.
# When True and no GPU is found, falls back to CPU automatically.
USE_GPU          = True
# Fraction of GPU memory to pre-allocate (0.0–1.0).
# None lets TensorFlow grow memory dynamically (recommended for shared machines).
GPU_MEMORY_FRACTION = None
# ---------------------------------------------------------------------------


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

    If ANNOTATION_PATH already exists on disk the annotation step is
    skipped and the existing file is loaded instead.
    """
    if os.path.exists(ANNOTATION_PATH):
        print(f"Annotations already exist at '{ANNOTATION_PATH}' — skipping annotation.")
        return pd.read_csv(ANNOTATION_PATH)

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
    # Preprocess all traces
    processed = {}
    for key in keys:
        trace = groups[key]
        trim = int(len(trace) * trim_ratio)
        processed[key] = equalize_length(normalize(trace.iloc[trim: len(trace) - trim]), max_len)

    # Convert annotations to normalized scalar labels
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
    """Configure TensorFlow GPU usage based on USE_GPU / GPU_MEMORY_FRACTION.

    Returns the device string to use for model training.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    if not USE_GPU or not gpus:
        if USE_GPU and not gpus:
            print("No GPU detected — falling back to CPU.")
        else:
            print("GPU disabled by USE_GPU=False — using CPU.")
        tf.config.set_visible_devices([], "GPU")
        return "/CPU:0"

    # GPU available and requested
    try:
        if GPU_MEMORY_FRACTION is None:
            # Grow memory only as needed; avoids allocating the entire VRAM upfront
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU enabled (memory growth): {[g.name for g in gpus]}")
        else:
            # Pre-allocate a fixed fraction of VRAM on the first GPU
            limit_mb = int(
                GPU_MEMORY_FRACTION
                * _get_gpu_total_memory_mb(gpus[0])
            )
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)],
            )
            print(f"GPU enabled (memory limit {limit_mb} MB): {gpus[0].name}")
    except RuntimeError as e:
        # Configuration must happen before any GPU ops; warn but continue
        print(f"GPU configuration warning: {e}")

    return "/GPU:0"


def _get_gpu_total_memory_mb(gpu_device) -> int:
    """Return total VRAM in MB for the given PhysicalDevice, or 4096 as a fallback."""
    try:
        import subprocess, re
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        values = [int(v.strip()) for v in result.stdout.strip().splitlines() if v.strip().isdigit()]
        return values[0] if values else 4096
    except Exception:
        return 4096


def build_model(input_shape: tuple, max_len: int, device: str = "/CPU:0"):
    import tensorflow as tf
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

    with tf.device(device):
        model = Sequential([
            Conv2D(32, kernel_size=(5, 1), strides=(1, 1), activation="relu", input_shape=input_shape),
            MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            Conv2D(64, kernel_size=(5, 1), activation="relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(max_len, activation="relu"),
            Dense(1, activation="linear"),
        ])
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    return model


def main():
    # Configure GPU (or CPU fallback) before any Keras/TF ops
    device = configure_gpu()

    # Load and group data
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

    # Annotate (skipped automatically if ANNOTATION_PATH exists)
    df_annotations = run_annotation(groups, keys, TRIM_RATIO, ANNOTATE_COUNT)
    if not os.path.exists(ANNOTATION_PATH):
        os.makedirs(os.path.dirname(ANNOTATION_PATH) or ".", exist_ok=True)
        df_annotations.to_csv(ANNOTATION_PATH, index=False)

    # Build datasets
    X_train, y_train, X_test, y_test, X_all, processed = build_datasets(
        groups, keys, df_annotations, max_len, TRIM_RATIO, TEST_SIZE, RANDOM_STATE
    )
    if VERBOSE:
        print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")

    # Time scale factor: converts normalized index back to physical time units
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

    # Predict on all samples
    predictions = model.predict(X_all).flatten() * max_len * time_scale
    results = [[key, pred] for key, pred in zip(keys, predictions)]

    if VERBOSE:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hist(predictions)
        ax.set_title("Distribution of predicted event positions")
        ax.set_xlabel("Predicted position")
        plt.show()
        print(f"Predicted {len(results)} samples")

    pd.DataFrame(results, columns=[GROUP_BY_COL, "predicted_position"]).to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
