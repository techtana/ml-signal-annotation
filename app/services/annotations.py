from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_annotations(path: str, group_by_col: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=[group_by_col, "label"])
    df = pd.read_csv(p)
    if group_by_col not in df.columns or "label" not in df.columns:
        return pd.DataFrame(columns=[group_by_col, "label"])
    df[group_by_col] = df[group_by_col].astype(str)
    # Web annotations can land on non-integer x positions, so keep labels float-safe.
    df["label"] = pd.to_numeric(df["label"], errors="coerce").astype(float)
    return df[[group_by_col, "label"]].copy()


def upsert_annotation(*, path: str, group_by_col: str, key: str, label: float) -> pd.DataFrame:
    df = load_annotations(path, group_by_col)
    key = str(key)
    label = float(label)

    if (df[group_by_col] == key).any():
        df.loc[df[group_by_col] == key, "label"] = label
    else:
        df = pd.concat([df, pd.DataFrame([{group_by_col: key, "label": label}])], ignore_index=True)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df

