"""
Basic annotation script.

Loads grouped time-series data from a CSV, displays each group
interactively, and saves the user's click-selected labels to a CSV.

Edit the Parameters section below before running.
"""

import random
import numpy as np
import pandas as pd
from sklearn import preprocessing
from annotation.interactive import InteractiveAnnotation_2dplot


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
DATA_PATH       = "data/traces.csv"       # input CSV with time-series data
OUTPUT_PATH     = "data/annotations.csv"  # where to write the label output
GROUP_BY_COL    = "run_id"                # column used to group samples
TIME_INDEX_COL  = "elapsed_time"          # column used as the x-axis index
# Columns to use as signal channels; set to None to use all remaining cols
CHANNEL_COLS    = None
TRIM_RATIO      = 0.1    # fraction trimmed from each end before display
ANNOTATE_COUNT  = 100    # number of samples to annotate; None = all
RANDOM_SHUFFLE  = True   # shuffle sample order before annotating
# ---------------------------------------------------------------------------


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)


def main():
    df = pd.read_csv(DATA_PATH)
    if TIME_INDEX_COL in df.columns:
        df = df.sort_values(TIME_INDEX_COL)

    channel_cols = CHANNEL_COLS or [c for c in df.columns if c not in (GROUP_BY_COL, TIME_INDEX_COL)]

    collection = {}
    keys = []
    for key, group in df.groupby(GROUP_BY_COL):
        keys.append(key)
        trace = group[channel_cols + [TIME_INDEX_COL]].set_index(TIME_INDEX_COL).iloc[1:]
        collection[key] = trace

    if RANDOM_SHUFFLE:
        random.shuffle(keys)

    total = ANNOTATE_COUNT if ANNOTATE_COUNT is not None else len(keys)
    annotations = []

    for i, key in enumerate(keys):
        if ANNOTATE_COUNT is not None and i >= ANNOTATE_COUNT:
            break
        trace = collection[key]
        trim = int(len(trace) * TRIM_RATIO)
        trace = normalize(trace.iloc[trim: len(trace) - trim])
        label = InteractiveAnnotation_2dplot(
            trace,
            plottitle=f"Annotating sample {key}  ({i + 1} / {total})"
        ).annotate()
        annotations.append([key, label])
        print(f"  label = {label}")

    pd.DataFrame(annotations, columns=[GROUP_BY_COL, "label"]).to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(annotations)} annotations to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
