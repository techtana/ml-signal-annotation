from __future__ import annotations

from flask import Blueprint, render_template

bp = Blueprint("core", __name__)


@bp.get("/")
def index():
    return render_template("index.html")

