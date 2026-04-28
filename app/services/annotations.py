from __future__ import annotations

import pandas as pd

from .cnn_pipeline import annotation_path_for, normalize_sample_key


def load_annotations(trace_path: str, group_by_col: str) -> pd.DataFrame:
    ann_path = annotation_path_for(trace_path)
    if not ann_path.exists():
        return pd.DataFrame(columns=[group_by_col, "label"])
    df = pd.read_csv(ann_path)
    if group_by_col not in df.columns or "label" not in df.columns:
        return pd.DataFrame(columns=[group_by_col, "label"])
    df[group_by_col] = df[group_by_col].astype(str).map(normalize_sample_key)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df[df["label"].notna()][[group_by_col, "label"]].drop_duplicates(
        subset=[group_by_col], keep="last"
    )
    return df.copy()


def upsert_annotation(*, path: str, group_by_col: str, key: str, label: float) -> pd.DataFrame:
    ann_path = annotation_path_for(path)
    ann_path.parent.mkdir(parents=True, exist_ok=True)

    key_norm = normalize_sample_key(key)
    label = float(label)

    if ann_path.exists():
        df = pd.read_csv(ann_path)
        if group_by_col not in df.columns:
            df = pd.DataFrame(columns=[group_by_col, "label"])
        else:
            df[group_by_col] = df[group_by_col].astype(str).map(normalize_sample_key)
    else:
        df = pd.DataFrame(columns=[group_by_col, "label"])

    mask = df[group_by_col] == key_norm
    if mask.any():
        df.loc[mask, "label"] = label
    else:
        new_row = pd.DataFrame({group_by_col: [key_norm], "label": [label]})
        df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(ann_path, index=False)
    return load_annotations(path, group_by_col)
