from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import session

from ...services.cnn_pipeline import CnnConfig, load_and_group


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for col in result.columns:
        mn, mx = result[col].min(), result[col].max()
        result[col] = 0.0 if mn == mx else (result[col] - mn) / (mx - mn)
    return result


def _sample_sort_key(value: str):
    text = str(value).strip()
    try:
        return (0, int(text))
    except ValueError:
        try:
            return (1, float(text))
        except ValueError:
            return (2, text.lower())


def _get_groups_and_keys(cfg: CnnConfig):
    groups, channel_cols, max_len = load_and_group(
        cfg.data_path, cfg.group_by_col, cfg.time_index_col, cfg.channel_cols
    )
    keys = sorted(groups.keys(), key=_sample_sort_key)
    return groups, keys, channel_cols, max_len


def _trace_file_options() -> list[str]:
    traces_dir = Path("data") / "traces"
    if not traces_dir.exists():
        return []
    return [p.as_posix() for p in sorted(traces_dir.glob("*.csv"))]


def _active_trace_path(cfg: CnnConfig) -> str | None:
    return session.get("active_trace_path") or cfg.data_path
