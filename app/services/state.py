from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class AnnotationState:
    keys: list[str]
    idx: int = 0
    trim_ratio: float = 0.1


def _state_path() -> Path:
    p = Path("artifacts") / "web"
    p.mkdir(parents=True, exist_ok=True)
    return p / "annotation_state.json"


def load_state() -> AnnotationState | None:
    p = _state_path()
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return AnnotationState(**data)


def save_state(state: AnnotationState) -> None:
    p = _state_path()
    p.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")


def reset_state(*, keys: list[str], trim_ratio: float) -> AnnotationState:
    state = AnnotationState(keys=[str(k) for k in keys], idx=0, trim_ratio=float(trim_ratio))
    save_state(state)
    return state

