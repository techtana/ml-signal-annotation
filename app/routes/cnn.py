from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
from flask import Blueprint, flash, jsonify, redirect, render_template, request, send_file, session, url_for

from ..services.annotations import load_annotations, upsert_annotation
from ..services.cnn_pipeline import (
    CnnConfig,
    annotation_path_for,
    load_and_group,
    normalize_sample_key,
)
from ..services.config_store import load_config, save_config
from ..services.state import load_state, reset_state, save_state
from ..services.trace_files import delete_temp_trace, is_temp_upload, save_uploaded_trace

bp = Blueprint("cnn", __name__)


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize each column, preserving index and column names."""
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


@bp.get("/")
def home():
    return redirect(url_for("cnn.source"))


@bp.route("/settings", methods=["GET", "POST"])
def settings():
    cfg = load_config()
    if request.method == "POST":
        def _float(name: str, default: float) -> float:
            v = request.form.get(name, "").strip()
            return float(v) if v else default

        def _int(name: str, default: int) -> int:
            v = request.form.get(name, "").strip()
            return int(v) if v else default

        channel_cols_raw = request.form.get("channel_cols", "").strip()
        channel_cols = None
        if channel_cols_raw:
            channel_cols = [c.strip() for c in channel_cols_raw.split(",") if c.strip()]

        cfg = CnnConfig(
            data_path=request.form.get("data_path", cfg.data_path).strip() or cfg.data_path,
            output_path=request.form.get("output_path", cfg.output_path).strip() or cfg.output_path,
            group_by_col=request.form.get("group_by_col", cfg.group_by_col).strip() or cfg.group_by_col,
            time_index_col=request.form.get("time_index_col", cfg.time_index_col).strip() or cfg.time_index_col,
            channel_cols=channel_cols,
            trim_ratio=_float("trim_ratio", cfg.trim_ratio),
            test_size=_float("test_size", cfg.test_size),
            random_state=_int("random_state", cfg.random_state),
            batch_size=_int("batch_size", cfg.batch_size),
            num_epochs=_int("num_epochs", cfg.num_epochs),
            use_gpu=bool(request.form.get("use_gpu")),
            gpu_memory_fraction=(
                _float("gpu_memory_fraction", cfg.gpu_memory_fraction)
                if request.form.get("gpu_memory_fraction", "").strip()
                else None
            ),
            artifacts_dir=request.form.get("artifacts_dir", cfg.artifacts_dir).strip() or cfg.artifacts_dir,
        )
        save_config(cfg)
        flash("Saved CNN settings.", "success")
        return redirect(url_for("cnn.settings"))

    return render_template("cnn/settings.html", cfg=cfg)


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


@bp.route("/source", methods=["GET", "POST"])
def source():
    cfg = load_config()
    sample_files = _trace_file_options()
    active_trace_path = _active_trace_path(cfg)

    if request.method == "POST":
        previous = session.get("active_trace_path")
        selected_sample = request.form.get("sample_path", "").strip()
        uploaded = request.files.get("trace_file")

        new_path = None
        if uploaded and uploaded.filename:
            new_path = save_uploaded_trace(uploaded)
            flash("Uploaded trace file loaded for this session only.", "success")
        elif selected_sample:
            new_path = selected_sample
            flash("Sample trace file selected.", "success")
        else:
            flash("Choose a sample file or upload a trace CSV.", "danger")
            return render_template(
                "cnn/source.html",
                cfg=cfg,
                sample_files=sample_files,
                active_trace_path=active_trace_path,
            )

        if is_temp_upload(previous) and previous != new_path:
            delete_temp_trace(previous)

        session["active_trace_path"] = new_path
        save_state(reset_state(keys=[], trim_ratio=cfg.trim_ratio))
        return redirect(url_for("cnn.annotate"))

    return render_template(
        "cnn/source.html",
        cfg=cfg,
        sample_files=sample_files,
        active_trace_path=active_trace_path,
    )


@bp.get("/download-sample")
def download_sample():
    trace_path = Path("data") / "traces" / "sample_b_traces.csv"
    ann_path = annotation_path_for(trace_path.as_posix())

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if trace_path.exists():
            zf.write(trace_path, "sample_traces.csv")
        if ann_path.exists():
            zf.write(ann_path, "sample_annotations.csv")
    buf.seek(0)

    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="sample_data.zip")


@bp.get("/annotate")
def annotate():
    cfg = load_config()
    if "active_trace_path" not in session:
        flash("Choose a trace CSV before annotating.", "info")
        return redirect(url_for("cnn.source"))
    active_trace_path = _active_trace_path(cfg)
    if not active_trace_path or not Path(active_trace_path).exists():
        flash("Select or upload a trace CSV before annotating.", "warning")
        return redirect(url_for("cnn.source"))

    runtime_cfg = replace(cfg, data_path=active_trace_path)
    groups, keys, channel_cols, _ = _get_groups_and_keys(runtime_cfg)
    state = load_state()
    if state is None or not state.keys:
        state = reset_state(keys=keys, trim_ratio=runtime_cfg.trim_ratio)
    else:
        if state.keys != keys:
            current_key = None
            if 0 <= state.idx < len(state.keys):
                current_key = state.keys[state.idx]
            elif state.keys:
                current_key = state.keys[-1]
            state = reset_state(keys=keys, trim_ratio=runtime_cfg.trim_ratio)
            if current_key in keys:
                state.idx = keys.index(current_key)
                save_state(state)

    df_ann = load_annotations(active_trace_path, cfg.group_by_col)

    review = state.idx >= len(state.keys)
    requested_key = request.args.get("key", "").strip()
    if requested_key and requested_key in groups:
        key = requested_key
        try:
            idx = state.keys.index(requested_key)
        except ValueError:
            idx = 0
    else:
        idx = min(state.idx, max(len(state.keys) - 1, 0))
        key = state.keys[idx] if state.keys else ""

    if not key:
        flash("No samples found in the input CSV.", "danger")
        return render_template("cnn/annotate.html", cfg=cfg, ready=False)

    trace = groups[key]
    trim = int(len(trace) * runtime_cfg.trim_ratio)
    trimmed = _normalize_df(trace.iloc[trim : len(trace) - trim])

    existing = df_ann[df_ann[cfg.group_by_col] == str(key)]
    existing_label = float(existing["label"].iloc[0]) if len(existing) else None

    x = [float(v) for v in trimmed.index.values]
    series = [
        {"name": col, "y": [float(v) for v in trimmed[col].values]}
        for col in trimmed.columns
    ]

    return render_template(
        "cnn/annotate.html",
        cfg=cfg,
        ready=True,
        active_trace_path=active_trace_path,
        key=key,
        idx=idx,
        total=len(state.keys),
        x=x,
        series=series,
        existing_label=existing_label,
        annotated_count=len(df_ann),
        channel_cols=channel_cols,
        review=review,
        has_prev=(idx > 0),
        has_next=(idx + 1 < len(state.keys)),
        prev_key=(state.keys[idx - 1] if idx > 0 else None),
        next_key=(state.keys[idx + 1] if idx + 1 < len(state.keys) else None),
    )


@bp.post("/annotate/reset")
def annotate_reset():
    cfg = load_config()
    active_trace_path = _active_trace_path(cfg)
    if not active_trace_path or not Path(active_trace_path).exists():
        flash("Select or upload a trace CSV before annotating.", "warning")
        return redirect(url_for("cnn.source"))
    runtime_cfg = replace(cfg, data_path=active_trace_path)
    groups, keys, _, _ = _get_groups_and_keys(runtime_cfg)
    reset_state(keys=keys, trim_ratio=cfg.trim_ratio)
    flash("Reset annotation session.", "info")
    return redirect(url_for("cnn.annotate"))


@bp.post("/annotate/skip")
def annotate_skip():
    state = load_state()
    if state is None:
        return redirect(url_for("cnn.annotate"))
    state.idx += 1
    save_state(state)
    return redirect(url_for("cnn.annotate"))


@bp.post("/annotate/label")
def annotate_label():
    cfg = load_config()
    active_trace_path = _active_trace_path(cfg)
    if not active_trace_path or not Path(active_trace_path).exists():
        return jsonify({"ok": False, "error": "Select or upload a trace CSV first."}), 400
    payload = request.get_json(silent=True) or {}
    key = str(payload.get("key", "")).strip()
    label = payload.get("label", None)
    if not key or label is None:
        return jsonify({"ok": False, "error": "key and label are required"}), 400

    upsert_annotation(path=active_trace_path, group_by_col=cfg.group_by_col, key=key, label=float(label))

    state = load_state()
    if state is not None and state.idx < len(state.keys) and state.keys[state.idx] == key:
        state.idx += 1
        save_state(state)

    return jsonify({"ok": True})


@bp.get("/train")
def train():
    cfg = load_config()
    active_trace_path = _active_trace_path(cfg)
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


@bp.get("/predict")
def predict():
    cfg = load_config()
    csv_options = _trace_file_options()
    selected_data_path = session.get("active_trace_path") or cfg.data_path
    return render_template(
        "cnn/predict.html",
        cfg=cfg,
        csv_options=csv_options,
        selected_data_path=selected_data_path,
    )
