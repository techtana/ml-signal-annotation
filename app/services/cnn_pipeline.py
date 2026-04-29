from __future__ import annotations

import datetime
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.preprocess import equalize_length, normalize


ANNOTATIONS_DIR = Path("data") / "annotations"


def annotation_path_for(trace_path: str) -> Path:
    """Derive the companion annotation file path from a trace file path.

    data/traces/foo.csv  ->  data/annotations/foo_annotations.csv
    """
    return ANNOTATIONS_DIR / (Path(trace_path).stem + "_annotations.csv")


@dataclass(frozen=True)
class CnnConfig:
    data_path: str = "data/traces/sample_a_traces.csv"
    output_path: str = "data/predictions/predictions.csv"

    group_by_col: str = "run_id"
    time_index_col: str = "elapsed_time"
    channel_cols: list[str] | None = None

    trim_ratio: float = 0.1
    test_size: float = 0.2
    random_state: int = 42
    batch_size: int = 200
    num_epochs: int = 100

    use_gpu: bool = True
    gpu_memory_fraction: float | None = None

    artifacts_dir: str = "artifacts"


def normalize_sample_key(value) -> str:
    """Normalize sample IDs so values like 1, 1.0, and '1' match consistently."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return text
    except ValueError:
        return text


def load_and_group(path: str, group_col: str, time_col: str, channel_cols: list[str] | None):
    df = pd.read_csv(path)
    if time_col in df.columns:
        df = df.sort_values(time_col)
    if channel_cols is None:
        channel_cols = [c for c in df.columns if c not in (group_col, time_col, "label")]

    max_len = 0
    groups: dict[str, pd.DataFrame] = {}
    for key, grp in df.groupby(group_col):
        trace = grp[channel_cols + [time_col]].set_index(time_col).iloc[1:]
        groups[normalize_sample_key(key)] = trace
        max_len = max(max_len, len(trace))

    return groups, channel_cols, max_len


def build_datasets(
    *,
    groups: dict[str, pd.DataFrame],
    keys: list[str],
    df_annotations: pd.DataFrame,
    max_len: int,
    trim_ratio: float,
    test_size: float,
    random_state: int,
    group_by_col: str,
):
    processed: dict[str, pd.DataFrame] = {}
    for key in keys:
        trace = groups[key]
        trim = int(len(trace) * trim_ratio)
        processed[key] = equalize_length(normalize(trace.iloc[trim : len(trace) - trim]), max_len)

    labels: dict[str, float] = {}
    for _, row in df_annotations.iterrows():
        key = normalize_sample_key(row[group_by_col])
        raw_label = float(row["label"])
        if key not in processed:
            continue
        idx = int(np.argmin(np.abs(processed[key].index - raw_label)))
        labels[key] = idx / max_len

    annotated_keys = [k for k in keys if k in labels]
    if len(annotated_keys) < 2:
        raise ValueError("Need at least 2 annotated samples to train (got < 2).")

    train_keys, test_keys = train_test_split(
        annotated_keys, test_size=test_size, random_state=random_state
    )

    def to_array(ks: list[str]):
        return np.expand_dims(np.array([processed[k].values for k in ks]), axis=3)

    X_train = to_array(train_keys)
    y_train = np.array([labels[k] for k in train_keys])
    X_test = to_array(test_keys)
    y_test = np.array([labels[k] for k in test_keys])
    X_all = to_array(keys)

    return X_train, y_train, X_test, y_test, X_all, processed


def configure_gpu(*, use_gpu: bool, gpu_memory_fraction: float | None) -> str:
    """Configure TensorFlow GPU usage. Returns the device string for training."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if (not use_gpu) or (not gpus):
        tf.config.set_visible_devices([], "GPU")
        return "/CPU:0"

    try:
        if gpu_memory_fraction is None:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            limit_mb = int(gpu_memory_fraction * _get_gpu_total_memory_mb())
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=limit_mb)],
            )
    except RuntimeError:
        # If TF initialized already, ignore and proceed.
        pass

    return "/GPU:0"


def _get_gpu_total_memory_mb() -> int:
    """Best-effort total VRAM in MB, falls back to 4096."""
    try:
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        values = [int(v.strip()) for v in result.stdout.strip().splitlines() if v.strip().isdigit()]
        return values[0] if values else 4096
    except Exception:
        return 4096


def build_model(*, input_shape: tuple[int, int, int], max_len: int, device: str):
    import tensorflow as tf
    from keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D
    from keras.models import Sequential

    with tf.device(device):
        model = Sequential(
            [
                Input(shape=input_shape),
                Conv2D(32, kernel_size=(5, 1), strides=(1, 1), activation="relu"),
                MaxPooling2D(pool_size=(2, 1), strides=(2, 1)),
                Conv2D(64, kernel_size=(5, 1), activation="relu"),
                MaxPooling2D(pool_size=(2, 1), strides=(2, 1)),
                Flatten(),
                Dense(max_len, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )
        model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_squared_error"])
    return model


def train_and_predict(cfg: CnnConfig) -> dict:
    """Train CNN regressor and write predictions CSV. Returns a small summary dict."""
    device = configure_gpu(use_gpu=cfg.use_gpu, gpu_memory_fraction=cfg.gpu_memory_fraction)

    groups, channel_cols, max_len = load_and_group(
        cfg.data_path, cfg.group_by_col, cfg.time_index_col, cfg.channel_cols
    )
    keys = list(groups.keys())

    ann_path = annotation_path_for(cfg.data_path)
    if ann_path.exists():
        df_ann = pd.read_csv(ann_path)
        df_ann[cfg.group_by_col] = df_ann[cfg.group_by_col].astype(str).map(normalize_sample_key)
        df_ann["label"] = pd.to_numeric(df_ann["label"], errors="coerce")
        df_ann = df_ann[df_ann["label"].notna()][[cfg.group_by_col, "label"]].drop_duplicates(
            subset=[cfg.group_by_col], keep="last"
        )
    else:
        df_ann = pd.DataFrame(columns=[cfg.group_by_col, "label"])

    X_train, y_train, X_test, y_test, X_all, processed = build_datasets(
        groups=groups,
        keys=keys,
        df_annotations=df_ann,
        max_len=max_len,
        trim_ratio=cfg.trim_ratio,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        group_by_col=cfg.group_by_col,
    )

    sample_trace = processed[keys[0]]
    time_scale = (sample_trace.index.max() - sample_trace.index.min()) / max_len

    input_shape = (max_len, len(channel_cols), 1)
    model = build_model(input_shape=input_shape, max_len=max_len, device=device)

    history = model.fit(
        X_train,
        y_train,
        batch_size=cfg.batch_size,
        epochs=cfg.num_epochs,
        verbose=1,
        validation_data=(X_test, y_test),
    )

    artifacts_dir = Path(cfg.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = artifacts_dir / f"cnn_regressor_{timestamp}.keras"
    model.save(model_path.as_posix())

    # Save sidecar metadata so predict_only can validate input compatibility.
    meta = {
        "data_path": cfg.data_path,
        "max_len": max_len,
        "num_channels": len(channel_cols),
        "channel_cols": channel_cols,
        "group_by_col": cfg.group_by_col,
        "time_index_col": cfg.time_index_col,
        "trim_ratio": cfg.trim_ratio,
        "input_shape": list(input_shape),
    }
    model_path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    predictions = model.predict(X_all, verbose=0).flatten() * max_len * time_scale
    out_df = pd.DataFrame(
        [[key, float(pred)] for key, pred in zip(keys, predictions)],
        columns=[cfg.group_by_col, "predicted_position"],
    )
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(cfg.output_path, index=False)

    return {
        "device": device,
        "model_path": model_path.as_posix(),
        "num_samples": len(keys),
        "num_annotated": int(df_ann[cfg.group_by_col].isin(keys).sum()) if cfg.group_by_col in df_ann.columns else 0,
        "final_loss": float(history.history["loss"][-1]) if history.history.get("loss") else None,
        "final_val_loss": float(history.history["val_loss"][-1]) if history.history.get("val_loss") else None,
        "output_path": cfg.output_path,
    }


def load_model_meta(model_path: str) -> dict:
    """Read the JSON sidecar saved alongside a model file. Returns {} if absent."""
    p = Path(model_path).with_suffix(".json")
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def predict_only(*, cfg: CnnConfig, model_path: str) -> dict:
    """Load an existing model, validate input compatibility, and write predictions CSV."""
    from keras.models import load_model

    # Load model first so we can validate before processing the entire dataset.
    model = load_model(model_path)
    expected_shape = model.input_shape          # (None, time_steps, channels, 1)
    expected_max_len: int = expected_shape[1]
    expected_channels: int = expected_shape[2]

    # Read sidecar for column-name hints (no TF required to produce it).
    meta = load_model_meta(model_path)
    trained_channel_cols: list[str] = meta.get("channel_cols", [])

    groups, channel_cols, _ = load_and_group(
        cfg.data_path, cfg.group_by_col, cfg.time_index_col, cfg.channel_cols
    )
    keys = list(groups.keys())

    if not keys:
        raise ValueError("No samples found in the data file.")

    # --- channel count ---
    if len(channel_cols) != expected_channels:
        hint = (
            f" Model was trained on: {trained_channel_cols}." if trained_channel_cols else ""
        )
        raise ValueError(
            f"Channel mismatch: model expects {expected_channels} channel(s) but "
            f"this data has {len(channel_cols)} ({', '.join(channel_cols)}).{hint}"
        )

    # --- column names (if metadata available) ---
    if trained_channel_cols and sorted(channel_cols) != sorted(trained_channel_cols):
        raise ValueError(
            f"Column name mismatch: model was trained on {trained_channel_cols} "
            f"but data has {channel_cols}. Rename columns or use the original training CSV."
        )

    # Preprocess using the model's expected max_len, not the new data's natural length.
    # This ensures X_all matches the model's input shape exactly.
    processed: dict[str, pd.DataFrame] = {}
    for key in keys:
        trace = groups[key]
        trim = int(len(trace) * cfg.trim_ratio)
        trimmed = trace.iloc[trim: len(trace) - trim]
        if len(trimmed) == 0:
            raise ValueError(
                f"Sample '{key}' has no rows after trimming "
                f"(trim_ratio={cfg.trim_ratio}, trace length={len(trace)})."
            )
        processed[key] = equalize_length(normalize(trimmed), expected_max_len)

    X_all = np.expand_dims(np.array([processed[k].values for k in keys]), axis=3)

    # Final shape guard — catches any remaining mismatch before handing to TF.
    actual = X_all.shape[1:]
    expected = (expected_max_len, expected_channels, 1)
    if actual != expected:
        raise ValueError(
            f"Input shape mismatch after preprocessing: got {actual}, "
            f"model expects {expected}."
        )

    sample_trace = processed[keys[0]]
    time_scale = (sample_trace.index.max() - sample_trace.index.min()) / expected_max_len

    predictions = model.predict(X_all, verbose=0).flatten() * expected_max_len * time_scale
    out_df = pd.DataFrame(
        [[key, float(pred)] for key, pred in zip(keys, predictions)],
        columns=[cfg.group_by_col, "predicted_position"],
    )
    Path(cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(cfg.output_path, index=False)

    return {
        "model_path": model_path,
        "num_samples": len(keys),
        "output_path": cfg.output_path,
        "expected_channels": expected_channels,
        "expected_max_len": expected_max_len,
    }

