from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import jsonify, render_template, request

from ...services.cnn_pipeline import annotation_path_for, normalize_sample_key, load_and_group
from ...services.config_store import load_config
from ._helpers import _trace_file_options, _active_trace_path
from . import bp


@bp.get("/predict")
def predict():
    cfg = load_config()
    csv_options = _trace_file_options()
    from flask import session
    selected_data_path = session.get("active_trace_path") or cfg.data_path
    return render_template(
        "cnn/predict.html",
        cfg=cfg,
        csv_options=csv_options,
        selected_data_path=selected_data_path,
    )


@bp.get("/trace-data")
def trace_data():
    """Return trace samples (+ annotations if present) as JSON for client-side inference."""
    cfg = load_config()
    path = request.args.get("path", "").strip() or cfg.data_path
    if not Path(path).exists():
        return jsonify({"error": f"File not found: {path}"}), 404

    groups, channel_cols, max_len = load_and_group(
        path, cfg.group_by_col, cfg.time_index_col, cfg.channel_cols
    )

    ann_path = annotation_path_for(path)
    annotations: dict[str, float] = {}
    if ann_path.exists():
        try:
            df_ann = pd.read_csv(ann_path)
            for _, row in df_ann.iterrows():
                key = normalize_sample_key(row[cfg.group_by_col])
                if pd.notna(row.get("label")):
                    annotations[key] = float(row["label"])
        except Exception:
            pass

    samples = {
        key: {
            "time": [float(v) for v in df.index.values],
            "channels": [[float(v) for v in row] for row in df.values],
        }
        for key, df in groups.items()
    }

    return jsonify({
        "samples": samples,
        "channel_cols": channel_cols,
        "max_len": max_len,
        "annotations": annotations,
        "group_by_col": cfg.group_by_col,
    })
