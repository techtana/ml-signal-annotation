"""
Flask entrypoint (app-centric).

Run (PowerShell):
  $env:FLASK_APP = "app/app.py"
  $env:FLASK_ENV = "development"
  flask run --host 0.0.0.0 --port 8972
"""

from __future__ import annotations

from . import create_app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8972, debug=True)
