"""
Lightweight Flask dashboard for browsing pipeline job status and configs.

Start:
    export FLASK_APP=dashboard/app.py
    export FLASK_ENV=development
    flask run --host=0.0.0.0 --port=8972
"""

from flask import Flask, render_template, request
import json
import pandas as pd

app = Flask(__name__, template_folder="templates")

CONFIG_PATH = "config.json"


@app.route("/")
def home():
    return "ML Regression Pipeline Dashboard"


@app.route("/jobs")
def jobs():
    """Show a summary table of all pipeline jobs."""
    columns = ["last_report", "report_date", "status", "error"]
    data = pd.read_json(CONFIG_PATH)[columns].copy()

    def view_link(row):
        return f'<a href="/job?id={row.name}">View</a>'

    data["last_report"] = data.apply(view_link, axis=1)
    return data.to_html(escape=False)


@app.route("/job", methods=["GET"])
def job():
    """Show the full config entry for a single job by index."""
    with open(CONFIG_PATH, "r") as f:
        data = json.load(f)

    job_id = request.args.get("id", "")
    if not job_id.isnumeric():
        return "id must be a non-negative integer", 400

    idx = int(job_id)
    if idx >= len(data):
        return f"No job at index {idx}", 404

    response = json.dumps(data[idx], indent=4, separators=(",", ": "))
    return render_template("entry.html", response=response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8972)
