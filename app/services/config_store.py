from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .cnn_pipeline import CnnConfig


def _config_path() -> Path:
    p = Path("artifacts") / "web"
    p.mkdir(parents=True, exist_ok=True)
    return p / "cnn_config.json"


def load_config() -> CnnConfig:
    p = _config_path()
    if not p.exists():
        return CnnConfig()
    data = json.loads(p.read_text(encoding="utf-8"))
    # Keep backward/forward compatibility by filtering fields
    allowed = set(CnnConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in allowed}
    return CnnConfig(**filtered)


def save_config(cfg: CnnConfig) -> None:
    p = _config_path()
    p.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

