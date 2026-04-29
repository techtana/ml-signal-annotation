from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ANNOTATIONS_DIR = Path("data") / "annotations"


def annotation_path_for(trace_path: str) -> Path:
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
