from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence, TextIO


DETERMINISM_ENV = {
    # Required by torch.use_deterministic_algorithms for CUDA >= 10.2 matmuls.
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}


def determinism_env(seed: int) -> dict[str, str]:
    """Environment overrides that make a child training/decoding run repeatable."""
    return {**DETERMINISM_ENV, "PYTHONHASHSEED": str(seed)}


def resolved_env(env: Mapping[str, str] | None) -> dict[str, str] | None:
    return None if env is None else {**os.environ, **env}


class _TeeStream:
    """Mirror text without taking ownership of either underlying stream."""

    def __init__(self, console: TextIO, transcript: TextIO) -> None:
        self.console = console
        self.transcript = transcript

    def write(self, value: str) -> int:
        self.console.write(value)
        self.transcript.write(value)
        return len(value)

    def flush(self) -> None:
        self.console.flush()
        self.transcript.flush()

    def isatty(self) -> bool:
        return False

    def close(self) -> None:
        """Leave stream lifetime to the console and ``tee_output`` owners.

        Some logging libraries retain ``sys.stderr`` while output is redirected
        and close that retained object during interpreter shutdown. Closing a
        tee must not close the real console or an already-managed transcript.
        """

    @property
    def encoding(self) -> str:
        return self.console.encoding or "utf-8"


def _available_log_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for index in range(1000):
        suffix = f".{stamp}" if index == 0 else f".{stamp}.{index}"
        candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a transcript beside {path}")


@contextmanager
def tee_output(path: Path) -> Iterator[Path]:
    """Capture stdout/stderr without ever truncating an existing transcript."""
    path.parent.mkdir(parents=True, exist_ok=True)
    destination = _available_log_path(path)
    print(f"Logging to {destination}", file=sys.stderr)
    with destination.open("x", encoding="utf-8") as transcript:
        stdout = _TeeStream(sys.stdout, transcript)
        stderr = _TeeStream(sys.stderr, transcript)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            yield destination


def run_logged(
    command: Sequence[str],
    log_path: Path,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND\n")
        log.write(json.dumps(list(command), ensure_ascii=False) + "\n\n")
        if env:
            log.write("ENV OVERRIDES\n")
            log.write(json.dumps(dict(env), ensure_ascii=False, sort_keys=True) + "\n\n")
        log.flush()
        process = subprocess.run(
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=resolved_env(env),
        )
    if process.returncode:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}; inspect {log_path}"
        )


def require_keys(config: dict[str, Any], *keys: str) -> None:
    missing = [key for key in keys if key not in config]
    if missing:
        raise ValueError(f"Missing configuration keys: {missing}")

