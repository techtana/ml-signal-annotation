from __future__ import annotations

from pathlib import Path

import pandas as pd

from .cnn_pipeline import normalize_sample_key
from .trace_files import read_trace_csv, write_trace_csv


def load_annotations(path: str, group_by_col: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=[group_by_col, "label"])
    df = read_trace_csv(path)
    if group_by_col not in df.columns or "label" not in df.columns:
        return pd.DataFrame(columns=[group_by_col, "label"])
    df[group_by_col] = df[group_by_col].map(normalize_sample_key)
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(float)
    df = df[df["label"].notna()][[group_by_col, "label"]].drop_duplicates(subset=[group_by_col], keep="last")
    return df.copy()


def upsert_annotation(*, path: str, group_by_col: str, key: str, label: float) -> pd.DataFrame:
    df = read_trace_csv(path)
    if group_by_col not in df.columns:
        raise ValueError(f"Missing group-by column: {group_by_col}")

    key = normalize_sample_key(key)
    label = float(label)
    normalized_keys = df[group_by_col].map(normalize_sample_key)
    if "label" not in df.columns:
        df["label"] = pd.Series([pd.NA] * len(df), dtype="Float64")
    else:
        df["label"] = pd.to_numeric(df["label"], errors="coerce").astype("Float64")

    mask = normalized_keys == key
    if not mask.any():
        raise ValueError(f"Sample '{key}' not found in trace file.")
    df.loc[mask, "label"] = label

    write_trace_csv(df, path)
    return load_annotations(path, group_by_col)

