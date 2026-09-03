"""Gateway-side proxy for a model that lives in another process.

E1 and E2 cannot share a Python environment (fairseq 0.12.2 needs numpy<2, the
Qwen stack needs numpy 2.x), so the browser talks to one origin and this class
forwards E2 requests to the sidecar over localhost HTTP.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import httpx

from mt_pipeline.serving.errors import ServiceError
from mt_pipeline.serving.registry import ModelSpec
from mt_pipeline.serving.schemas import ModelHealth, ModelLimits
from mt_pipeline.serving.settings import UISettings
from mt_pipeline.serving.translator import TranslationInputError, TranslationResult


LOGGER = logging.getLogger(__name__)

# The sidecar answers health from the event loop even while weights load in its
# threadpool, so this stays short regardless of the translate timeout.
HEALTH_TIMEOUT_SECONDS = 3.0


def _limits_of(spec: ModelSpec) -> ModelLimits:
    return ModelLimits(
        max_characters=spec.limits.max_characters,
        max_units=spec.limits.max_units,
        unit=spec.limits.unit,
        unit_label=spec.limits.unit_label,
        client_estimate=spec.limits.client_estimate,
        chars_per_unit=spec.limits.chars_per_unit,
    )


def _error_body(response: httpx.Response) -> dict:
    try:
        payload = response.json()
    except ValueError:
        return {}
    error = payload.get("error") if isinstance(payload, dict) else None
    return error if isinstance(error, dict) else {}


class RemoteTranslator:
    """Translator protocol over HTTP, plus a TTL-cached health probe."""

    def __init__(
        self,
        base_url: str,
        spec: ModelSpec,
        *,
        timeout_seconds: float = 900.0,
        health_ttl_seconds: float = 5.0,
        client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.spec = spec
        self.model_id = spec.model_id
        self.device = "cuda"
        self.parameter_count: int | None = None
        self.limits = spec.limits
        # The gateway has no Qwen tokenizer; it enforces the character cap and
        # leaves the authoritative token check to the process that owns E2.
        self.count_units = None

        self._health_ttl = health_ttl_seconds
        self._clock = clock
        self._cached: ModelHealth | None = None
        self._cached_at: float | None = None
        self._owns_client = client is None
        self._client = client or httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                connect=5.0, read=timeout_seconds, write=10.0, pool=5.0
            ),
        )

    # ---------------------------------------------------------------- health

    def _unreachable(self, detail: str) -> ModelHealth:
        return ModelHealth(
            key=self.spec.key,
            model_id=self.spec.model_id,
            label=self.spec.label,
            sublabel=self.spec.sublabel,
            status="unavailable",
            device=self.device,
            parameter_count=None,
            message=(
                f"Không kết nối được tiến trình E2 tại {self.base_url} ({detail}). "
                "Chạy `make ui-api-e2`."
            ),
            limits=_limits_of(self.spec),
            reports_unknown_tokens=self.spec.reports_unknown_tokens,
            slow_first_request=self.spec.slow_first_request,
        )

    def _probe(self) -> ModelHealth:
        response = self._client.get("/api/health", timeout=HEALTH_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        entries = payload.get("models") or []
        remote = next(
            (entry for entry in entries if entry.get("key") == self.spec.key),
            entries[0] if entries else None,
        )
        if remote is None:
            return self._unreachable("sidecar reported no models")
        # Identity and copy stay the gateway's; status and limits are the
        # sidecar's, since it is the process that actually owns the model.
        health = ModelHealth(**remote)
        health.key = self.spec.key
        health.label = self.spec.label
        health.sublabel = self.spec.sublabel
        return health

    def describe(self) -> ModelHealth:
        now = self._clock()
        if (
            self._cached is not None
            and self._cached_at is not None
            and now - self._cached_at < self._health_ttl
        ):
            return self._cached
        try:
            health = self._probe()
        except Exception as exc:
            LOGGER.warning("E2 health probe failed: %s", type(exc).__name__)
            health = self._unreachable(type(exc).__name__)
        self._cached = health
        self._cached_at = now
        return health

    # ------------------------------------------------------------- translate

    def translate(self, text: str) -> TranslationResult:
        try:
            response = self._client.post(
                "/api/translate", json={"text": text, "model": self.spec.key}
            )
        except httpx.ConnectError as exc:
            raise ServiceError(
                503,
                "E2_SIDECAR_UNREACHABLE",
                f"Không kết nối được tiến trình E2 tại {self.base_url}. "
                "Chạy `make ui-api-e2`.",
            ) from exc
        except httpx.ReadTimeout as exc:
            raise ServiceError(
                504,
                "E2_TIMEOUT",
                "E2 chưa trả lời kịp. Lần dịch đầu tiên phải nạp mô hình 8B "
                "nên có thể mất vài phút; hãy thử lại.",
            ) from exc
        except httpx.HTTPError as exc:
            raise ServiceError(
                502,
                "E2_UPSTREAM_ERROR",
                "Tiến trình E2 gặp lỗi. Xem log của tiến trình :8001.",
            ) from exc

        if response.status_code == 200:
            # A successful translate proves the sidecar's state changed
            # (not_loaded -> ready); drop the cached health so the next
            # /api/health reflects it.
            self._cached = None
            self._cached_at = None
            body = response.json()
            return TranslationResult(
                translation=body["translation"],
                normalized_input=body["normalized_input"],
                source_token_count=body["source_token_count"],
                target_token_count=body["target_token_count"],
                unknown_tokens=list(body.get("unknown_tokens") or []),
                latency_ms=body["latency_ms"],
            )

        error = _error_body(response)
        code = error.get("code")
        message = error.get("message")
        upstream_id = error.get("request_id")
        if upstream_id:
            LOGGER.info(
                "E2 upstream error status=%s code=%s upstream_request_id=%s",
                response.status_code,
                code,
                upstream_id,
            )

        if response.status_code == 422:
            raise TranslationInputError(
                code or "INPUT_TOO_LONG",
                message or "Đầu vào không hợp lệ với E2.",
            )
        if response.status_code == 503:
            self._cached = None
            self._cached_at = None
            raise ServiceError(
                503, code or "MODEL_UNAVAILABLE", message or "E2 chưa sẵn sàng."
            )
        raise ServiceError(
            502,
            "E2_UPSTREAM_ERROR",
            "Tiến trình E2 gặp lỗi. Xem log của tiến trình :8001.",
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def build_remote_e2_translator(settings: UISettings, spec: ModelSpec) -> RemoteTranslator:
    assert settings.e2_base_url is not None
    return RemoteTranslator(
        settings.e2_base_url,
        spec,
        timeout_seconds=settings.e2_timeout_seconds,
        health_ttl_seconds=settings.e2_health_ttl_seconds,
    )
