from __future__ import annotations

from flask import flash, redirect, render_template, request, url_for

from ...services.cnn_pipeline import CnnConfig
from ...services.config_store import load_config, save_config
from . import bp


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
