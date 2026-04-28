from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Blueprint, abort, render_template, request

bp = Blueprint("pipeline", __name__)

CONFIG_PATH = Path("config.json")


@bp.get("/jobs")
def jobs():
    """Pretty-ish summary table of all pipeline jobs (Component 2)."""
    if not CONFIG_PATH.exists():
        abort(404, description=f"Missing {CONFIG_PATH.as_posix()}")

    columns = ["last_report", "report_date", "status", "error"]
    df = pd.read_json(CONFIG_PATH)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        abort(400, description=f"config.json missing columns: {missing}")

    data = df[columns].copy()
    records = data.to_dict(orient="records")
    for i, rec in enumerate(records):
        rec["_idx"] = i
    return render_template("pipeline/jobs.html", records=records, colnames=columns)


@bp.get("/job")
def job():
    """Show the full config entry for a single job by index."""
    if not CONFIG_PATH.exists():
        abort(404, description=f"Missing {CONFIG_PATH.as_posix()}")

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    job_id = request.args.get("id", "")
    if not job_id.isnumeric():
        abort(400, description="id must be a non-negative integer")

    idx = int(job_id)
    if idx >= len(data):
        abort(404, description=f"No job at index {idx}")

    response = json.dumps(data[idx], indent=2, ensure_ascii=False)
    return render_template("pipeline/job.html", response=response, job_id=idx)

