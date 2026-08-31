from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class UISettings:
    checkpoint_path: Path | None
    data_bin_path: Path | None
    device: str = "cpu"
    port: int = 8000
    serve_frontend: bool = True

    @classmethod
    def from_env(cls) -> "UISettings":
        checkpoint = os.environ.get("MT_CHECKPOINT_PATH")
        data_bin = os.environ.get("MT_DATA_BIN_PATH")
        device = os.environ.get("MT_DEVICE", "cpu").strip().lower()
        port_text = os.environ.get("MT_PORT", "8000")
        try:
            port = int(port_text)
        except ValueError as exc:
            raise ValueError("MT_PORT must be an integer") from exc
        if not 1 <= port <= 65535:
            raise ValueError("MT_PORT must be between 1 and 65535")
        if device not in {"cpu", "cuda"}:
            raise ValueError("MT_DEVICE must be either 'cpu' or 'cuda'")
        return cls(
            checkpoint_path=Path(checkpoint).expanduser() if checkpoint else None,
            data_bin_path=Path(data_bin).expanduser() if data_bin else None,
            device=device,
            port=port,
        )

    def artifact_error(self) -> str | None:
        if self.checkpoint_path is None or self.data_bin_path is None:
            return "Set MT_CHECKPOINT_PATH and MT_DATA_BIN_PATH before translating."
        if not self.checkpoint_path.is_file():
            return "The configured E1 checkpoint does not exist or is not a file."
        if not self.data_bin_path.is_dir():
            return "The configured Fairseq data-bin does not exist or is not a directory."
        required = ("dict.vi.txt", "dict.zh.txt")
        if missing := [name for name in required if not (self.data_bin_path / name).is_file()]:
            return f"The configured data-bin is missing: {', '.join(missing)}."
        return None
