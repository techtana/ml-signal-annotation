from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import jsonify, render_template, session

from ...services.cnn_pipeline import annotation_path_for, normalize_sample_key
from ...services.config_store import load_config
from ._helpers import _active_trace_path
from . import bp


@bp.get("/train")
def train():
    cfg = load_config()
    active_trace_path = _active_trace_path(cfg)
    from ...services.annotations import load_annotations
    df_ann = (
        load_annotations(active_trace_path, cfg.group_by_col)
        if active_trace_path and Path(active_trace_path).exists()
        else pd.DataFrame()
    )
    return render_template(
        "cnn/train.html",
        cfg=cfg,
        annotated_count=len(df_ann),
        active_trace_path=active_trace_path,
    )


@bp.get("/training-data")
def training_data():
    """Return all trace samples + annotations as JSON for client-side training."""
    from ...services.cnn_pipeline import load_and_group
    cfg = load_config()
    active_trace_path = session.get("active_trace_path") or cfg.data_path
    if not active_trace_path or not Path(active_trace_path).exists():
        return jsonify({"error": "No active trace file. Select one first."}), 404

    groups, channel_cols, max_len = load_and_group(
        active_trace_path, cfg.group_by_col, cfg.time_index_col, cfg.channel_cols
    )

    ann_path = annotation_path_for(active_trace_path)
    annotations: dict[str, float] = {}
    if ann_path.exists():
        df_ann = pd.read_csv(ann_path)
        for _, row in df_ann.iterrows():
            key = normalize_sample_key(row[cfg.group_by_col])
            if pd.notna(row.get("label")):
                annotations[key] = float(row["label"])

    samples = {
        key: {
            "time": [float(v) for v in df.index.values],
            "channels": [[float(v) for v in row] for row in df.values],
        }
        for key, df in groups.items()
    }

    return jsonify({
        "samples": samples,
        "annotations": annotations,
        "channel_cols": channel_cols,
        "max_len": max_len,
        "cfg": {
            "trim_ratio": cfg.trim_ratio,
            "test_size": cfg.test_size,
            "random_state": cfg.random_state,
            "num_epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "group_by_col": cfg.group_by_col,
        },
    })
