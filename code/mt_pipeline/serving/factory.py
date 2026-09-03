"""Application factory for the local translation UI.

Kept separate from :mod:`mt_pipeline.serving.app` because that module builds an
E1 app at import time; the E2 sidecar needs the factory without that side effect.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from mt_pipeline.config import REPO_ROOT
from mt_pipeline.serving.errors import ServiceError
from mt_pipeline.serving.registry import (
    DEFAULT_MODEL_KEY,
    E1_KEY,
    E1_SPEC,
    E2_KEY,
    ModelSpec,
    e2_spec,
    resolve,
)
from mt_pipeline.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelHealth,
    ModelLimits,
    TranslationRequest,
    TranslationResponse,
)
from mt_pipeline.serving.settings import UISettings
from mt_pipeline.serving.translator import (
    E1Translator,
    InputLimits,
    ModelArtifactError,
    TranslationFailure,
    TranslationInputError,
    Translator,
    prepare_for,
)


LOGGER = logging.getLogger(__name__)
TranslatorFactory = Callable[[UISettings], Translator]


def _default_translator_factory(settings: UISettings) -> Translator:
    artifact_error = settings.artifact_error()
    if artifact_error:
        raise RuntimeError(artifact_error)
    assert settings.checkpoint_path is not None
    assert settings.data_bin_path is not None
    return E1Translator(settings.checkpoint_path, settings.data_bin_path, settings.device)


def _load_error_message(key: str, exc: Exception, settings: UISettings) -> str:
    """Turn a factory failure into the corrective message the UI shows."""
    if key == E1_KEY:
        artifact_error = settings.artifact_error()
        if artifact_error:
            return artifact_error
        if isinstance(exc, ModelArtifactError):
            return str(exc)
        return "Không thể nạp E1. Kiểm tra checkpoint, data-bin và môi trường Fairseq."

    # Only consult the local E2 artifacts when this process actually owns them;
    # on the gateway the model lives behind an HTTP hop.
    if not settings.e2_enabled:
        artifact_error = settings.e2_artifact_error()
        if artifact_error:
            return artifact_error
    if isinstance(exc, ModelArtifactError):
        return str(exc)
    return "Không thể nạp E2. Kiểm tra adapter, run_manifest.json và GPU."


def _model_limits(limits: InputLimits) -> ModelLimits:
    return ModelLimits(
        max_characters=limits.max_characters,
        max_units=limits.max_units,
        unit=limits.unit,
        unit_label=limits.unit_label,
        client_estimate=limits.client_estimate,
        chars_per_unit=limits.chars_per_unit,
    )


def _fallback_device(key: str, settings: UISettings) -> str:
    return settings.e2_device if key == E2_KEY else settings.device


def _describe(
    spec: ModelSpec,
    translator: Translator | None,
    message: str | None,
    settings: UISettings,
) -> ModelHealth:
    """Build one model's health entry.

    A translator may expose ``describe()`` to answer for itself — the remote
    E2 proxy does, so the gateway reports the sidecar's own status and limits
    rather than a duplicated guess.
    """
    if translator is not None:
        describe = getattr(translator, "describe", None)
        if callable(describe):
            return describe()

    if translator is None:
        return ModelHealth(
            key=spec.key,
            model_id=spec.model_id,
            label=spec.label,
            sublabel=spec.sublabel,
            status="unavailable",
            device=_fallback_device(spec.key, settings),
            parameter_count=None,
            message=message,
            limits=_model_limits(spec.limits),
            reports_unknown_tokens=spec.reports_unknown_tokens,
            slow_first_request=spec.slow_first_request,
        )

    return ModelHealth(
        key=spec.key,
        model_id=getattr(translator, "model_id", spec.model_id),
        label=spec.label,
        sublabel=spec.sublabel,
        status=getattr(translator, "status", "ready"),
        device=getattr(translator, "device", _fallback_device(spec.key, settings)),
        parameter_count=getattr(translator, "parameter_count", None),
        message=getattr(translator, "message", None) or message,
        limits=_model_limits(getattr(translator, "limits", None) or spec.limits),
        reports_unknown_tokens=spec.reports_unknown_tokens,
        slow_first_request=spec.slow_first_request,
    )


def _aggregate_status(models: Sequence[ModelHealth]) -> str:
    usable = [model for model in models if model.status != "unavailable"]
    if not usable:
        return "model_unavailable"
    if len(usable) == len(models):
        return "ready"
    return "degraded"


def create_app(
    settings: UISettings | None = None,
    translator_factory: TranslatorFactory | None = None,
    *,
    e2_translator_factory: TranslatorFactory | None = None,
    models: Sequence[str] | None = None,
) -> FastAPI:
    resolved_settings = settings or UISettings.from_env()

    specs: dict[str, ModelSpec] = {
        E1_KEY: E1_SPEC,
        E2_KEY: e2_spec(resolved_settings.e2_max_prompt_tokens),
    }

    factories: dict[str, TranslatorFactory] = {
        E1_KEY: translator_factory or _default_translator_factory
    }
    if e2_translator_factory is not None:
        factories[E2_KEY] = e2_translator_factory
    elif resolved_settings.e2_enabled:
        # MT_E2_URL is set, so E2 lives in the sidecar process.
        from mt_pipeline.serving.remote import build_remote_e2_translator

        factories[E2_KEY] = lambda s: build_remote_e2_translator(s, specs[E2_KEY])

    if models is None:
        model_keys = [E1_KEY] + ([E2_KEY] if E2_KEY in factories else [])
    else:
        model_keys = list(models)
    missing = [key for key in model_keys if key not in factories]
    if missing:
        raise ValueError(f"No translator factory supplied for: {', '.join(missing)}")
    default_key = DEFAULT_MODEL_KEY if DEFAULT_MODEL_KEY in model_keys else model_keys[0]

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.translators = {}
        app.state.model_messages = {}
        app.state.inference_locks = {key: asyncio.Lock() for key in model_keys}
        for key in model_keys:
            try:
                app.state.translators[key] = await run_in_threadpool(
                    factories[key], resolved_settings
                )
                app.state.model_messages[key] = None
            except Exception as exc:  # Keep the UI available for artifact setup.
                app.state.translators[key] = None
                app.state.model_messages[key] = _load_error_message(
                    key, exc, resolved_settings
                )
                LOGGER.warning("%s model unavailable: %s", key, type(exc).__name__)
        yield
        for translator in app.state.translators.values():
            close = getattr(translator, "close", None)
            if callable(close):
                close()
        app.state.translators = {}

    app = FastAPI(
        title="Local Vietnamese–Classical Chinese MT",
        version="2.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(ServiceError)
    async def handle_service_error(_request: Request, exc: ServiceError) -> JSONResponse:
        request_id = uuid4().hex
        payload = ErrorResponse(
            error={"code": exc.code, "message": exc.message, "request_id": request_id}
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

    @app.get("/api/health", response_model=HealthResponse)
    async def health(request: Request) -> HealthResponse:
        entries = []
        for key in model_keys:
            entry = await run_in_threadpool(
                _describe,
                specs[key],
                request.app.state.translators.get(key),
                request.app.state.model_messages.get(key),
                resolved_settings,
            )
            entries.append(entry)
        return HealthResponse(
            status=_aggregate_status(entries),
            default_model=default_key,
            models=entries,
        )

    @app.post(
        "/api/translate",
        response_model=TranslationResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def translate(payload: TranslationRequest, request: Request) -> TranslationResponse:
        spec = resolve(payload.model)
        if spec is None or spec.key not in model_keys:
            raise ServiceError(
                422,
                "UNKNOWN_MODEL",
                f"Không có mô hình '{payload.model}' trong dịch vụ này.",
            )

        translator: Translator | None = request.app.state.translators.get(spec.key)
        limits = spec.limits
        counter = spec.count_units
        if translator is not None:
            limits = getattr(translator, "limits", None) or limits
            counter = getattr(translator, "count_units", None) or counter

        try:
            prepared = prepare_for(payload.text, limits, counter)
        except TranslationInputError as exc:
            raise ServiceError(422, exc.code, exc.message) from exc

        if translator is None:
            raise ServiceError(
                503,
                "MODEL_UNAVAILABLE",
                request.app.state.model_messages.get(spec.key)
                or f"{spec.label} chưa sẵn sàng.",
            )

        try:
            async with request.app.state.inference_locks[spec.key]:
                result = await run_in_threadpool(translator.translate, prepared.text)
        except TranslationInputError as exc:
            raise ServiceError(422, exc.code, exc.message) from exc
        except ServiceError:
            raise
        except TranslationFailure as exc:
            raise ServiceError(500, exc.code, exc.message) from exc
        except Exception as exc:
            request_id = uuid4().hex
            LOGGER.error(
                "Translation failed request_id=%s model=%s failure_type=%s",
                request_id,
                spec.key,
                type(exc).__name__,
            )
            error = ErrorResponse(
                error={
                    "code": "TRANSLATION_FAILED",
                    "message": "Không thể tạo bản dịch. Hãy kiểm tra model rồi thử lại.",
                    "request_id": request_id,
                }
            )
            return JSONResponse(status_code=500, content=error.model_dump())

        return TranslationResponse(
            translation=result.translation,
            normalized_input=result.normalized_input,
            source_token_count=result.source_token_count,
            target_token_count=result.target_token_count,
            unknown_tokens=result.unknown_tokens,
            latency_ms=result.latency_ms,
            model_id=getattr(translator, "model_id", spec.model_id),
        )

    frontend_dist = Path(REPO_ROOT) / "ui" / "dist"
    if resolved_settings.serve_frontend and frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app
