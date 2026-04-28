from __future__ import annotations

import os

from app import create_app


HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8972"))
DEBUG = os.environ.get("FLASK_DEBUG", "1") in {"1", "true", "True"}


app = create_app()


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG)
