from __future__ import annotations

import io
import zipfile
from dataclasses import replace
from pathlib import Path

from flask import flash, jsonify, redirect, render_template, request, send_file, session, url_for

from ...services.annotations import load_annotations, upsert_annotation
from ...services.cnn_pipeline import annotation_path_for
from ...services.config_store import load_config
from ...services.state import load_state, reset_state, save_state
from ...services.trace_files import delete_temp_trace, is_temp_upload, save_uploaded_trace
from ._helpers import _normalize_df, _get_groups_and_keys, _trace_file_options, _active_trace_path
from . import bp


@bp.get("/")
def home():
    return redirect(url_for("cnn.source"))


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


@bp.post("/upload-trace")
def upload_trace():
    cfg = load_config()
    uploaded = request.files.get("trace_file")
    if not uploaded or not uploaded.filename:
        return jsonify({"ok": False, "error": "No file provided"}), 400
    previous = session.get("active_trace_path")
    new_path = save_uploaded_trace(uploaded)
    if is_temp_upload(previous) and previous != new_path:
        delete_temp_trace(previous)
    session["active_trace_path"] = new_path
    save_state(reset_state(keys=[], trim_ratio=cfg.trim_ratio))
    return jsonify({"ok": True, "path": new_path})


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
