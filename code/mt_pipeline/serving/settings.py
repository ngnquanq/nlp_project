from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path

from mt_pipeline.config import load_yaml, repo_path
from mt_pipeline.serving.registry import E2_DEFAULT_MAX_PROMPT_TOKENS


DEFAULT_E2_CONFIG_PATH = "configs/e2_qwen3_qlora.yaml"


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _path_env(name: str) -> Path | None:
    raw = os.environ.get(name)
    return Path(raw).expanduser() if raw else None


@dataclass(frozen=True)
class UISettings:
    checkpoint_path: Path | None
    data_bin_path: Path | None
    device: str = "cpu"
    port: int = 8000
    serve_frontend: bool = True

    # --- E2, read by the gateway ------------------------------------------
    e2_base_url: str | None = None
    e2_timeout_seconds: float = 900.0
    e2_health_ttl_seconds: float = 5.0

    # --- E2, read by the sidecar ------------------------------------------
    e2_port: int = 8001
    e2_config_path: str = DEFAULT_E2_CONFIG_PATH
    e2_adapter_path: Path | None = None
    e2_manifest_path: Path | None = None
    e2_max_prompt_tokens: int = E2_DEFAULT_MAX_PROMPT_TOKENS
    e2_device: str = "cuda"

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

        e2_device = os.environ.get("MT_E2_DEVICE", "cuda").strip().lower()
        if e2_device not in {"cpu", "cuda"}:
            raise ValueError("MT_E2_DEVICE must be either 'cpu' or 'cuda'")
        e2_port = _int_env("MT_E2_PORT", 8001)
        if not 1 <= e2_port <= 65535:
            raise ValueError("MT_E2_PORT must be between 1 and 65535")
        base_url = os.environ.get("MT_E2_URL")

        return cls(
            checkpoint_path=Path(checkpoint).expanduser() if checkpoint else None,
            data_bin_path=Path(data_bin).expanduser() if data_bin else None,
            device=device,
            port=port,
            e2_base_url=base_url.rstrip("/") if base_url else None,
            e2_timeout_seconds=_float_env("MT_E2_TIMEOUT_SECONDS", 900.0),
            e2_health_ttl_seconds=_float_env("MT_E2_HEALTH_TTL_SECONDS", 5.0),
            e2_port=e2_port,
            e2_config_path=os.environ.get("MT_E2_CONFIG_PATH", DEFAULT_E2_CONFIG_PATH),
            e2_adapter_path=_path_env("MT_E2_ADAPTER_PATH"),
            e2_manifest_path=_path_env("MT_E2_MANIFEST_PATH"),
            e2_max_prompt_tokens=_int_env(
                "MT_E2_MAX_PROMPT_TOKENS", E2_DEFAULT_MAX_PROMPT_TOKENS
            ),
            e2_device=e2_device,
        )

    @classmethod
    def for_e2_sidecar(cls) -> "UISettings":
        """Settings for the E2 process: no frontend, no E1, its own port."""
        settings = cls.from_env()
        return replace(
            settings,
            checkpoint_path=None,
            data_bin_path=None,
            serve_frontend=False,
            port=settings.e2_port,
            e2_base_url=None,
        )

    @property
    def e2_enabled(self) -> bool:
        return self.e2_base_url is not None

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

    # ------------------------------------------------------------------ E2

    def resolved_e2_config_path(self) -> Path:
        return repo_path(self.e2_config_path)

    def resolved_e2_adapter_path(self, config: dict | None = None) -> Path | None:
        if self.e2_adapter_path is not None:
            return self.e2_adapter_path
        if config is None:
            return None
        checkpoint_dir = config.get("checkpoint_dir")
        return repo_path(checkpoint_dir) / "adapter" if checkpoint_dir else None

    def resolved_e2_manifest_path(self, config: dict | None = None) -> Path | None:
        if self.e2_manifest_path is not None:
            return self.e2_manifest_path
        if config is None:
            return None
        work_dir = config.get("work_dir")
        return repo_path(work_dir) / "run_manifest.json" if work_dir else None

    def e2_artifact_error(self) -> str | None:
        """Mirror of artifact_error() for the E2 sidecar's own artifacts."""
        if self.e2_device != "cuda":
            return "E2 yêu cầu GPU CUDA; đặt MT_E2_DEVICE=cuda."
        config_path = self.resolved_e2_config_path()
        if not config_path.is_file():
            return f"Không tìm thấy config E2: {config_path}."
        try:
            config = load_yaml(config_path)
        except Exception:
            return f"Không đọc được config E2: {config_path}."

        adapter = self.resolved_e2_adapter_path(config)
        if adapter is None or not (adapter / "adapter_config.json").is_file():
            return f"Không tìm thấy adapter E2 (thiếu adapter_config.json): {adapter}."

        manifest = self.resolved_e2_manifest_path(config)
        if manifest is None or not manifest.is_file():
            return f"Không tìm thấy run_manifest.json của E2: {manifest}."
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            return f"Không đọc được run_manifest.json: {manifest}."
        if not payload.get("resolved_model_revision"):
            return f"run_manifest.json thiếu khoá resolved_model_revision: {manifest}."
        return None
