from __future__ import annotations

from flask import Blueprint

bp = Blueprint("cnn", __name__)

from . import annotate, settings, train, predict  # noqa: F401, E402
