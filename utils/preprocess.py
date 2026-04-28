"""
Preprocessing and feature-engineering utilities shared across the pipeline
and the annotation/training scripts.
"""

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing


# ---------------------------------------------------------------------------
# General array / dataframe utilities
# ---------------------------------------------------------------------------

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize all columns; preserves index and column names."""
    scaler = preprocessing.MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)


def equalize_length(df: pd.DataFrame, max_length: int) -> pd.DataFrame:
    """Pad a DataFrame to `max_length` rows by repeating the last row."""
    arr = df.values
    if len(arr) < max_length:
        pad = np.tile(arr[-1], (max_length - len(arr), 1))
        arr = np.vstack([arr, pad])
    return pd.DataFrame(arr, columns=df.columns)


def dropna_column_percent(df: pd.DataFrame, threshold: float):
    """Drop columns whose missing-value fraction exceeds `threshold`.

    Returns the filtered DataFrame and the list of dropped column labels.
    """
    missing_frac = df.isna().sum(axis=0) / len(df)
    drop_cols = missing_frac[missing_frac > threshold].index
    return df.drop(columns=drop_cols), drop_cols


# ---------------------------------------------------------------------------
# Training diagnostics
# ---------------------------------------------------------------------------

def plot_training_history(history):
    """Plot loss and MSE curves from a Keras training History object."""
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.title("Cross Entropy Loss")
    plt.plot(history.history["loss"], color="blue", label="train")
    plt.plot(history.history["val_loss"], color="orange", label="validation")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Mean Squared Error")
    plt.plot(history.history["mean_squared_error"], color="blue", label="train")
    plt.plot(history.history["val_mean_squared_error"], color="orange", label="validation")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Run-parameter column helpers
# ---------------------------------------------------------------------------

RUN_PARAM_PATTERN = "^run_param::"


def get_run_param_cols(df: pd.DataFrame, pattern: str = RUN_PARAM_PATTERN) -> list:
    """Return all column names in `df` that match the run-parameter prefix."""
    cols = [c for c in df.columns if re.match(pattern, c)]
    print("Run-param cols:\n\t" + "\n\t".join(cols))
    return cols


def get_run_param_shortnames(origin_cols: list) -> list:
    """Collapse tool-specific prefixes from run-parameter column names.

    e.g. ``run_param::TOOL_A::::2::CycleTime`` → ``run_param::2::CycleTime``
    """
    shortnames = list(set(
        "run_param::" + "::".join(col.split(" - ")[0].split("::")[3:])
        for col in origin_cols
    ))
    print("Run-param shortnames:\n\t" + "\n\t".join(shortnames))
    return shortnames


def get_run_param_by_position(process_position, row, shortnames: list, method: str = "chamber") -> list:
    """Look up the run-parameter value for this row's process position.

    Parameters
    ----------
    process_position : str or int
        The position identifier for the current row (e.g. chamber ID, slot).
    row : pandas.Series
        A single row of the merged DataFrame.
    shortnames : list
        Short run-parameter names from ``get_run_param_shortnames``.
    method : {"chamber", "slot"}
        How to interpret `process_position`.

    Returns
    -------
    list of float
        One value per entry in `shortnames`.
    """

    def resolve_chamber(pos):
        pos = str(pos)
        if len(pos) == 1:
            return pos + "0"
        if len(pos) == 2:
            return pos[0] + "0"
        return str(row.get("tool_id", pos))

    def resolve_slot():
        return str(row.get("tool_id", process_position))

    def regex_lookup(pos, item, dataseries=row):
        pattern = f"{pos}::(.*)" + item[len("run_param"):]
        r = re.compile(pattern)
        cols = dataseries.keys().tolist()
        matched = list(filter(r.search, cols))
        return pattern, cols, matched

    if method == "chamber":
        resolved = resolve_chamber(process_position)
    elif method == "slot":
        resolved = resolve_slot()
    else:
        raise ValueError(f"Unsupported method: {method}")

    results = []
    for item in shortnames:
        attempt = 1
        while attempt:
            _, _, matched = regex_lookup(resolved, item)
            if len(matched) >= 1:
                break
            if attempt == 1:
                resolved = resolve_chamber(process_position)
                attempt += 1
            elif attempt == 2:
                resolved = resolve_slot()
                attempt += 1
            else:
                raise LookupError(f"No column matched for run-param item '{item}' at position '{resolved}'")

        values = list(filter(np.isfinite, row[matched].values))
        results.append(np.mean(values) if values else np.nan)

    return results


# ---------------------------------------------------------------------------
# Data type conversions
# ---------------------------------------------------------------------------

def hms_to_seconds(value) -> float:
    """Convert a hhmmss-encoded float to total seconds."""
    s = str(int(value)).zfill(6)
    return 3600 * int(s[:2]) + 60 * int(s[2:4]) + int(s[4:])


def millis_to_seconds(value) -> float:
    """Convert milliseconds to seconds."""
    return float(value) * 1000


def convert_value(value, conv_type: str):
    """Apply a named type conversion to a single value.

    Supported `conv_type` values: ``"hms_to_seconds"``, ``"hms"``,
    ``"millis_to_seconds"``, ``"millis"``.
    """
    if pd.isna(value):
        return np.nan
    t = conv_type.lower()
    if t in ("hms_to_seconds", "hms"):
        return hms_to_seconds(value)
    if t in ("millis_to_seconds", "millis"):
        return millis_to_seconds(value)
    raise ValueError(f"Unrecognized conv_type: {conv_type!r}")


# ---------------------------------------------------------------------------
# Regression reporting
# ---------------------------------------------------------------------------

def generate_linear_equation(x_params: list, coefficients) -> str:
    """Format a PLS regression equation as a readable string."""
    model = {x_params[i]: coefficients.flatten()[i] for i in range(len(x_params))}
    terms = [f"[{k}] * {v:.10f}" for k, v in model.items()]
    return "(" + ") + (".join(terms) + ")"


def display_weights(predictors: list, targets: list, models: dict):
    """Print a formatted coefficient table for each model group."""
    maxlen = max(len(p) for p in predictors)
    for group, model in models.items():
        print(f"Group: {group}")
        for n, p in enumerate(predictors):
            print(" " * (maxlen - 8), "   ", (" " * 8 + "|") * (n + 1), p, sep="")
        print(" " * maxlen, " ", "+-----", "-" * 8 * len(predictors), "+", sep="")
        for target in targets:
            print(target, end=f"{' ' * (maxlen - len(str(target)))}\t |  ")
            for coef in model.coef_:
                for w in [round(v, 2) for v in coef]:
                    print(w, end=f"{' ' * (8 - len(str(w)))} ")
            print("\b|")
        print(" " * maxlen, " ", "+-----", "-" * 8 * len(predictors), "+", sep="")


# ---------------------------------------------------------------------------
# Slot/position mapping
# ---------------------------------------------------------------------------

def map_slot_to_position(slot, position_definition: dict) -> str:
    """Map a numeric slot index to a named position using a range definition.

    Parameters
    ----------
    slot : int or str
        Slot number.
    position_definition : dict
        ``{position_name: [min_slot, max_slot]}`` ranges.

    Returns
    -------
    str
        Matched position name, or ``"N/A"`` if no range matches.
    """
    try:
        slot = int(slot)
    except (ValueError, TypeError):
        raise ValueError(f"'map_slot_to_position' requires a numeric slot; got {slot!r}")
    for name, (lo, hi) in position_definition.items():
        if lo <= slot <= hi:
            return name
    return "N/A"
