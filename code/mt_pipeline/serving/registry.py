"""Per-model serving metadata shared by the gateway and the E2 sidecar.

This module stays free of FastAPI, torch and transformers so both processes
can import it regardless of which model they own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from mt_pipeline.serving.translator import (
    E1_LIMITS,
    MAX_INPUT_CHARACTERS,
    MODEL_ID,
    InputLimits,
    _count_e1_positions,
)


E1_KEY = "e1"
E2_KEY = "e2"
DEFAULT_MODEL_KEY = E1_KEY

E2_MODEL_ID = "e2_qwen3_8b_qlora_vi_zh_v1"

# configs/e2_qwen3_qlora.yaml sets max_sequence_length: 768, but that is the
# training window covering prompt *and* target, and predict_qlora never
# truncates at inference. Capping the chat-templated prompt at 512 leaves at
# least 256 positions for generation inside the window the adapter was fit on.
E2_DEFAULT_MAX_PROMPT_TOKENS = 512

# Measured over the 510 validation sources (work/e1_fairseq_vi_zh_v1/text/valid.vi)
# with the E2 adapter's tokenizer: 45,346 characters / 13,380 tokens = 3.39.
# The browser cannot run Qwen BPE, so it uses this divisor to estimate a token
# count; the server holds the authoritative check.
E2_CHARS_PER_TOKEN = 3.39


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    label: str
    sublabel: str
    limits: InputLimits
    count_units: Callable[[str], int] | None
    reports_unknown_tokens: bool
    slow_first_request: bool


def e2_limits(
    max_prompt_tokens: int = E2_DEFAULT_MAX_PROMPT_TOKENS,
    chars_per_unit: float = E2_CHARS_PER_TOKEN,
) -> InputLimits:
    return InputLimits(
        max_characters=MAX_INPUT_CHARACTERS,
        max_units=max_prompt_tokens,
        unit="token",
        unit_label="token",
        over_limit_message=(
            f"E2 hỗ trợ tối đa {max_prompt_tokens} token trong prompt. "
            "Hãy rút gọn câu nguồn."
        ),
        client_estimate=True,
        chars_per_unit=chars_per_unit,
    )


E1_SPEC = ModelSpec(
    key=E1_KEY,
    model_id=MODEL_ID,
    # Kept as exactly "E1" so UI copy built as f"{label} sẵn sàng" is unchanged.
    label="E1",
    sublabel="Fairseq Transformer",
    limits=E1_LIMITS,
    count_units=_count_e1_positions,
    reports_unknown_tokens=True,
    slow_first_request=False,
)


def e2_spec(
    max_prompt_tokens: int = E2_DEFAULT_MAX_PROMPT_TOKENS,
    count_units: Callable[[str], int] | None = None,
) -> ModelSpec:
    """Build the E2 spec.

    The gateway leaves ``count_units`` as None because it has no Qwen
    tokenizer; the sidecar passes its own prompt token counter.
    """
    return ModelSpec(
        key=E2_KEY,
        model_id=E2_MODEL_ID,
        label="E2",
        sublabel="Qwen3-8B QLoRA 4-bit",
        limits=e2_limits(max_prompt_tokens),
        count_units=count_units,
        reports_unknown_tokens=False,
        slow_first_request=True,
    )


E2_SPEC = e2_spec()

_BY_KEY = {spec.key: spec for spec in (E1_SPEC, E2_SPEC)}
_BY_MODEL_ID = {spec.model_id: spec for spec in (E1_SPEC, E2_SPEC)}


def resolve(name: str) -> ModelSpec | None:
    """Look a model up by short key ("e1") or by full model id."""
    return _BY_KEY.get(name) or _BY_MODEL_ID.get(name)
