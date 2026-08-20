from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    """Hash relative names and contents of all files in a directory tree."""
    digest = hashlib.sha256()
    for file_path in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(file_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {path}:{line_number}: {error}") from error
            if not isinstance(value, dict):
                raise ValueError(f"Expected object on {path}:{line_number}")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def package_versions(names: Iterable[str]) -> dict[str, str | None]:
    from importlib.metadata import PackageNotFoundError, version

    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def base_manifest() -> dict[str, Any]:
    conda_packages: list[dict[str, Any]] | None = None
    pip_packages: list[str] | None = None
    if os.environ.get("CONDA_PREFIX"):
        try:
            conda_packages = json.loads(
                subprocess.check_output(["conda", "list", "--json"], text=True)
            )
        except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
            conda_packages = None
    try:
        pip_packages = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        pip_packages = None
    accelerator: dict[str, Any] = {"cuda_available": False, "devices": []}
    try:
        import torch

        accelerator["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            accelerator["cuda_version"] = torch.version.cuda
            accelerator["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_packages": conda_packages,
        "pip_packages": pip_packages,
        "accelerator": accelerator,
    }
