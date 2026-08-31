from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ready", "model_unavailable"]
    model_id: str
    device: str
    parameter_count: int | None = None
    message: str | None = None


class TranslationRequest(BaseModel):
    text: str = Field(description="Vietnamese source passage")


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
