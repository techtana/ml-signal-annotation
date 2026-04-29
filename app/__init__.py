from __future__ import annotations

from flask import Flask


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    app.config.from_mapping(
        SECRET_KEY="dev",  # override in production (env var / config file)
    )

    from .routes.core import bp as core_bp
    from .routes.cnn import bp as cnn_bp  # package: app/routes/cnn/

    app.register_blueprint(core_bp)
    app.register_blueprint(cnn_bp, url_prefix="/cnn")

    return app
