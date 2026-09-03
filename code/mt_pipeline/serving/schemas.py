from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from mt_pipeline.serving.registry import DEFAULT_MODEL_KEY


ModelStatus = Literal["ready", "not_loaded", "unavailable"]


class ModelLimits(BaseModel):
    max_characters: int
    # Always populated: the browser renders this number directly.
    max_units: int
    unit: Literal["position", "token"]
    unit_label: str
    client_estimate: bool = False
    chars_per_unit: float | None = None


class ModelHealth(BaseModel):
    key: str
    model_id: str
    label: str
    sublabel: str
    status: ModelStatus
    device: str
    parameter_count: int | None = None
    message: str | None = None
    limits: ModelLimits
    reports_unknown_tokens: bool
    slow_first_request: bool


class HealthResponse(BaseModel):
    status: Literal["ready", "degraded", "model_unavailable"]
    default_model: str
    models: list[ModelHealth]


class TranslationRequest(BaseModel):
    text: str = Field(description="Vietnamese source passage")
    model: str = Field(
        default=DEFAULT_MODEL_KEY,
        description='Model key ("e1"/"e2") or full model id',
    )


class TranslationResponse(BaseModel):
    translation: str
    normalized_input: str
    source_token_count: int
    target_token_count: int
    unknown_tokens: list[str]
    latency_ms: int
    model_id: str


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail
