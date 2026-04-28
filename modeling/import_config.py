"""
Bulk-import pipeline job definitions from a CSV into config.json.

CSV columns (required):
    status, source_type, position_type, position_column,
    source_name, source_step, source_tool, start_date,
    sensor_aggregation, modelgroup, target_step, target_items,
    feature_items, data_conversion

Run:
    python -m modeling.import_config
"""

import json
import re
import pandas as pd

CONFIG_PATH = "config.json"
CSV_PATH    = "job_candidates.csv"


def parse_job(row: pd.Series) -> dict:
    """Convert a CSV row into a pipeline job definition dict."""
    job = {}

    # Reporting containers (populated by pipeline.py)
    job["last_report"] = ""
    job["report_date"]  = ""

    # Status and identity
    job["status"]          = row["status"]
    job["source_type"]     = row["source_type"]
    job["position_type"]   = row["position_type"]
    job["position_column"] = row["position_column"]
    job["source_name"]     = row["source_name"]
    job["source_step"]     = row["source_step"]
    job["source_tool"]     = row["source_tool"]

    if not re.match(r"\d{4}-\d{2}-\d{2}", row["start_date"]):
        raise ValueError(f"start_date wrong format: {row['start_date']!r} — expected YYYY-MM-DD")
    job["start_date"] = row["start_date"]

    # Sensor aggregation list
    job["sensor_aggregation"] = [s.strip() for s in row["sensor_aggregation"].split(",")]
    job["modelgroup"]         = row["modelgroup"]

    # Target items (Y variables): CSV column "target_items" is comma-separated names;
    # "target_step" is the single measurement step they all come from.
    job["target_items"] = [
        {"source_step": row["target_step"], "name": name.strip(), "limits": []}
        for name in row["target_items"].split(",")
    ]

    # Feature items (additional X variables): comma-separated column names
    job["feature_items"] = [f.strip() for f in row["feature_items"].split(",")]

    # Run-parameter enable flag (default on)
    job["run_params_enable"] = 1

    # Extract date (will be filled by pipeline.py after first run)
    job["extract_date"] = ""

    # Data conversion: optional regex→conv_type mapping
    dc = str(row.get("data_conversion", "")).strip().lower()
    if dc in ("hms_to_seconds", "hms", "millis_to_seconds", "millis"):
        job["data_conversion"] = {"^run_param::(.*)Time": dc}
    else:
        job["data_conversion"] = {}

    # Table lookup dicts (filled by pipeline.py during validation)
    job["target_tables"]  = {}
    job["feature_tables"] = {}

    return job


def main():
    sheet = pd.read_csv(CSV_PATH)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    added = 0
    for _, row in sheet.iterrows():
        if row.isna().all():
            continue
        row_status = str(row.get("status", "")).strip().lower()
        if row_status in ("", "add", "new", "reset"):
            config.append(parse_job(row))
            added += 1

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=4)

    print(f"config.json updated — {added} job(s) added, {len(config)} total.")


if __name__ == "__main__":
    main()
