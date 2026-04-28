from __future__ import annotations

import shutil
import tempfile
import uuid
from pathlib import Path

import pandas as pd


TEMP_DIR = Path(tempfile.gettempdir()) / "ml_cnn_annotation_tool_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_trace(file_storage) -> str:
    suffix = Path(file_storage.filename or "trace.csv").suffix or ".csv"
    target = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"
    file_storage.save(target)
    return target.as_posix()


def delete_temp_trace(path: str | None) -> None:
    if not path:
        return
    p = Path(path)
    try:
        if p.exists() and TEMP_DIR in p.parents:
            p.unlink()
    except OSError:
        pass


def read_trace_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def write_trace_csv(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def is_temp_upload(path: str | None) -> bool:
    if not path:
        return False
    p = Path(path)
    return p.exists() and TEMP_DIR in p.parents

