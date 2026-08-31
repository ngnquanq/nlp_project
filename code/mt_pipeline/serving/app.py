from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
import logging
from pathlib import Path
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from mt_pipeline.config import REPO_ROOT
from mt_pipeline.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    TranslationRequest,
    TranslationResponse,
)
from mt_pipeline.serving.settings import UISettings
from mt_pipeline.serving.translator import (
    E1Translator,
    MODEL_ID,
    ModelArtifactError,
    TranslationInputError,
    Translator,
    prepare_input,
)


LOGGER = logging.getLogger(__name__)
TranslatorFactory = Callable[[UISettings], Translator]


class ServiceError(RuntimeError):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


def _default_translator_factory(settings: UISettings) -> Translator:
    artifact_error = settings.artifact_error()
    if artifact_error:
        raise RuntimeError(artifact_error)
    assert settings.checkpoint_path is not None
    assert settings.data_bin_path is not None
    return E1Translator(settings.checkpoint_path, settings.data_bin_path, settings.device)


def create_app(
    settings: UISettings | None = None,
    translator_factory: TranslatorFactory | None = None,
) -> FastAPI:
    resolved_settings = settings or UISettings.from_env()
    factory = translator_factory or _default_translator_factory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.translator = None
        app.state.model_message = None
        app.state.inference_lock = asyncio.Lock()
        try:
            app.state.translator = await run_in_threadpool(factory, resolved_settings)
        except Exception as exc:  # Keep the UI available for artifact setup.
            artifact_error = resolved_settings.artifact_error()
            if artifact_error:
                app.state.model_message = artifact_error
            elif isinstance(exc, ModelArtifactError):
                app.state.model_message = str(exc)
            else:
                app.state.model_message = (
                    "Không thể nạp E1. Kiểm tra checkpoint, data-bin và môi trường Fairseq."
                )
            LOGGER.warning("E1 model unavailable: %s", type(exc).__name__)
        yield
        app.state.translator = None

    app = FastAPI(
        title="E1 Vietnamese–Classical Chinese MT",
        version="1.0.0",
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
        translator: Translator | None = request.app.state.translator
        if translator is None:
            return HealthResponse(
                status="model_unavailable",
                model_id=MODEL_ID,
                device=resolved_settings.device,
                message=request.app.state.model_message,
            )
        return HealthResponse(
            status="ready",
            model_id=translator.model_id,
            device=translator.device,
            parameter_count=translator.parameter_count,
        )

    @app.post(
        "/api/translate",
        response_model=TranslationResponse,
        responses={422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def translate(payload: TranslationRequest, request: Request) -> TranslationResponse:
        try:
            prepared = prepare_input(payload.text)
        except TranslationInputError as exc:
            raise ServiceError(422, exc.code, exc.message) from exc

        translator: Translator | None = request.app.state.translator
        if translator is None:
            raise ServiceError(
                503,
                "MODEL_UNAVAILABLE",
                request.app.state.model_message
                or "E1 chưa sẵn sàng. Kiểm tra đường dẫn checkpoint và data-bin.",
            )

        try:
            async with request.app.state.inference_lock:
                result = await run_in_threadpool(translator.translate, prepared.text)
        except TranslationInputError as exc:
            raise ServiceError(422, exc.code, exc.message) from exc
        except Exception as exc:
            request_id = uuid4().hex
            LOGGER.error(
                "Translation failed request_id=%s failure_type=%s",
                request_id,
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
            model_id=translator.model_id,
        )

    frontend_dist = Path(REPO_ROOT) / "ui" / "dist"
    if resolved_settings.serve_frontend and frontend_dist.is_dir():
        app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")

    return app


app = create_app()
