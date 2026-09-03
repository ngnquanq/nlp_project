"""Deferred model loading behind the Translator protocol."""

from __future__ import annotations

import logging
import threading
from time import perf_counter
from typing import Callable

from mt_pipeline.serving.translator import (
    InputLimits,
    ModelArtifactError,
    TranslationResult,
    Translator,
)


LOGGER = logging.getLogger(__name__)


class LazyTranslator:
    """Satisfies the Translator protocol immediately, loads on first use.

    The app's lifespan builds translators eagerly. Wrapping the expensive model
    in this keeps startup instant while the real weights are constructed inside
    the first ``translate`` call, which already runs in a worker thread.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        build: Callable[[], Translator],
        count_units: Callable[[str], int] | None = None,
        limits: InputLimits | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.limits = limits
        self.count_units = count_units
        self.status = "not_loaded"
        self.message: str | None = None
        self.parameter_count: int | None = None
        self._build = build
        self._inner: Translator | None = None
        self._lock = threading.Lock()

    def _ensure_loaded(self) -> Translator:
        inner = self._inner
        if inner is not None:
            return inner
        with self._lock:
            # Re-check: a concurrent first request may have loaded it already.
            if self._inner is not None:
                return self._inner
            if self.status == "unavailable":
                # Sticky failure. Retrying a multi-minute load on every request
                # would wedge the machine; the operator restarts the process.
                raise ModelArtifactError(self.message or "Model failed to load.")
            started = perf_counter()
            try:
                inner = self._build()
            except Exception as exc:
                self.status = "unavailable"
                self.message = str(exc) or type(exc).__name__
                LOGGER.error("%s failed to load: %s", self.model_id, type(exc).__name__)
                raise
            LOGGER.info(
                "%s loaded in %.1fs", self.model_id, perf_counter() - started
            )
            self._inner = inner
            self.status = "ready"
            self.message = None
            self.parameter_count = getattr(inner, "parameter_count", None)
            self.device = getattr(inner, "device", self.device)
            if getattr(inner, "limits", None) is not None:
                self.limits = inner.limits
            return inner

    def translate(self, text: str) -> TranslationResult:
        return self._ensure_loaded().translate(text)

    def reset(self) -> None:
        """Drop a failed load so a future retry hook can try again."""
        with self._lock:
            self._inner = None
            self.status = "not_loaded"
            self.message = None
