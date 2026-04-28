from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for

from ..services.annotations import load_annotations, upsert_annotation
from ..services.cnn_pipeline import CnnConfig, find_latest_model, load_and_group, predict_only, train_and_predict
from ..services.config_store import load_config, save_config
from ..services.state import load_state, reset_state, save_state
from utils.preprocess import normalize

bp = Blueprint("cnn", __name__)


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
    return redirect(url_for("cnn.annotate"))


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
            annotation_path=request.form.get("annotation_path", cfg.annotation_path).strip() or cfg.annotation_path,
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


@bp.get("/annotate")
def annotate():
    cfg = load_config()
    if not Path(cfg.data_path).exists():
        flash(f"Missing input data CSV: {cfg.data_path}", "danger")
        return render_template("cnn/annotate.html", cfg=cfg, ready=False)

    groups, keys, channel_cols, _ = _get_groups_and_keys(cfg)
    state = load_state()
    if state is None or not state.keys:
        state = reset_state(keys=keys, trim_ratio=cfg.trim_ratio)
    else:
        # If dataset contents or ordering changed, rebuild state in sorted order.
        if state.keys != keys:
            current_key = None
            if 0 <= state.idx < len(state.keys):
                current_key = state.keys[state.idx]
            elif state.keys:
                current_key = state.keys[-1]

            state = reset_state(keys=keys, trim_ratio=cfg.trim_ratio)
            if current_key in keys:
                state.idx = keys.index(current_key)
                save_state(state)

    df_ann = load_annotations(cfg.annotation_path, cfg.group_by_col)

    # Review mode: allow browsing/adjusting labels even after reaching the end.
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
    trim = int(len(trace) * cfg.trim_ratio)
    trimmed = normalize(trace.iloc[trim : len(trace) - trim])

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
    groups, keys, _, _ = _get_groups_and_keys(cfg)
    reset_state(keys=keys, trim_ratio=cfg.trim_ratio)
    flash("Reset annotation session.", "info")
    return redirect(url_for("cnn.annotate"))


@bp.post("/annotate/skip")
def annotate_skip():
    cfg = load_config()
    state = load_state()
    if state is None:
        return redirect(url_for("cnn.annotate"))
    state.idx += 1
    save_state(state)
    return redirect(url_for("cnn.annotate"))


@bp.post("/annotate/label")
def annotate_label():
    cfg = load_config()
    payload = request.get_json(silent=True) or {}
    key = str(payload.get("key", "")).strip()
    label = payload.get("label", None)
    if not key or label is None:
        return jsonify({"ok": False, "error": "key and label are required"}), 400

    upsert_annotation(path=cfg.annotation_path, group_by_col=cfg.group_by_col, key=key, label=float(label))

    state = load_state()
    if state is not None and state.idx < len(state.keys) and state.keys[state.idx] == key:
        state.idx += 1
        save_state(state)

    return jsonify({"ok": True})


@bp.route("/train", methods=["GET", "POST"])
def train():
    cfg = load_config()
    summary = None
    if request.method == "POST":
        if not Path(cfg.annotation_path).exists():
            flash(f"Missing annotations CSV: {cfg.annotation_path}", "danger")
            return redirect(url_for("cnn.train"))
        try:
            summary = train_and_predict(cfg)
            flash("Training complete. Predictions written.", "success")
        except Exception as e:
            flash(f"Training failed: {e}", "danger")

    df_ann = load_annotations(cfg.annotation_path, cfg.group_by_col)
    return render_template("cnn/train.html", cfg=cfg, summary=summary, annotated_count=len(df_ann))


@bp.route("/predict", methods=["GET", "POST"])
def predict():
    cfg = load_config()
    latest = find_latest_model(cfg.artifacts_dir)
    summary = None
    if request.method == "POST":
        model_path = request.form.get("model_path", "").strip() or latest
        if not model_path:
            flash("No model found. Train first or provide a model path.", "danger")
            return redirect(url_for("cnn.predict"))
        try:
            summary = predict_only(cfg=cfg, model_path=model_path)
            flash("Prediction complete. Predictions written.", "success")
        except Exception as e:
            flash(f"Prediction failed: {e}", "danger")

    return render_template("cnn/predict.html", cfg=cfg, latest_model=latest, summary=summary)


@bp.get("/results")
def results():
    cfg = load_config()
    ann = load_annotations(cfg.annotation_path, cfg.group_by_col)
    pred_path = Path(cfg.output_path)
    preds = None
    if pred_path.exists():
        preds = pd.read_csv(pred_path)

    return render_template(
        "cnn/results.html",
        cfg=cfg,
        annotations=ann.to_dict(orient="records"),
        predictions=(preds.to_dict(orient="records") if preds is not None else None),
    )

