"""
Full pipeline: annotate time-series samples, then train a CNN regressor
to predict the labeled event position automatically.

Edit the Parameters section before running.
"""

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


def run_annotation(groups: dict, keys: list, max_len: int, trim_ratio: float, count: int | None):
    annotations = []
    total = count if count is not None else len(keys)

    for i, key in enumerate(keys):
        if count is not None and i >= count:
            break
        trace = groups[key]
        trim = int(len(trace) * trim_ratio)
        trimmed = normalize(trace.iloc[trim: len(trace) - trim])
        label = InteractiveAnnotation_2dplot(
            trimmed,
            plottitle=f"Sample {key}  ({i + 1} / {total})"
        ).annotate()
        annotations.append([key, label])
        print(f"  label = {label}")

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


def build_model(input_shape: tuple, max_len: int):
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

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

    # Annotate
    df_annotations = run_annotation(groups, keys, max_len, TRIM_RATIO, ANNOTATE_COUNT)
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
    model = build_model(input_shape, max_len)
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
    model_path = f"cnn_regressor_{timestamp}"
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
