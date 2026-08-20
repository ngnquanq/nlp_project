from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Configuration must be a mapping: {config_path}")
    data["_config_path"] = str(config_path)
    return data


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def portable_project_path(path: str | Path) -> str:
    """Return a path suitable for saved artifacts without exposing a home path."""
    candidate = repo_path(path).resolve()
    try:
        return candidate.relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return candidate.name

