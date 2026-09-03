from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Literal, Protocol

from mt_pipeline.normalize import detokenize_chinese, normalize_whitespace


MODEL_ID = "e1_fairseq_vi_zh_v1"
EXPECTED_PARAMETER_COUNT = 67_652_608
MAX_SOURCE_POSITIONS = 256
MAX_INPUT_CHARACTERS = 4_000


@dataclass(frozen=True)
class InputLimits:
    """Per-model input ceilings.

    ``max_units`` is always a real number because the browser renders it
    directly; whether a given process can actually count those units is
    signalled separately, by passing ``count_units=None`` to :func:`prepare_for`.
    """

    max_characters: int
    max_units: int
    unit: Literal["position", "token"]
    unit_label: str
    over_limit_message: str
    client_estimate: bool = False
    chars_per_unit: float | None = None


class TranslationInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ModelArtifactError(RuntimeError):
    pass


class TranslationFailure(RuntimeError):
    """The model ran but produced nothing usable."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class PreparedInput:
    text: str
    source_token_count: int
    encoded_position_count: int


@dataclass(frozen=True)
class TranslationResult:
    translation: str
    normalized_input: str
    source_token_count: int
    target_token_count: int
    unknown_tokens: list[str]
    latency_ms: int


class Translator(Protocol):
    model_id: str
    device: str
    parameter_count: int | None

    def translate(self, text: str) -> TranslationResult: ...


def _count_e1_positions(text: str) -> int:
    return len(text.split()) + 1  # Fairseq appends EOS.


E1_LIMITS = InputLimits(
    max_characters=MAX_INPUT_CHARACTERS,
    max_units=MAX_SOURCE_POSITIONS,
    unit="position",
    unit_label="vị trí",
    over_limit_message=(
        f"E1 hỗ trợ tối đa {MAX_SOURCE_POSITIONS} vị trí mã hoá, "
        "bao gồm token kết thúc câu."
    ),
)


def prepare_for(
    text: str,
    limits: InputLimits,
    count_units: Callable[[str], int] | None,
) -> PreparedInput:
    """Normalize and validate one request against a single model's limits.

    ``count_units`` is None when this process cannot count the model's own
    units — the gateway has no Qwen tokenizer, so it enforces the character
    cap and leaves the authoritative token check to the process that owns
    the model.
    """
    normalized = normalize_whitespace(text)
    if not normalized:
        raise TranslationInputError("EMPTY_INPUT", "Hãy nhập câu tiếng Việt cần dịch.")
    if len(normalized) > limits.max_characters:
        raise TranslationInputError(
            "INPUT_TOO_LONG",
            f"Văn bản không được vượt quá {limits.max_characters:,} ký tự.",
        )
    source_token_count = len(normalized.split())
    if count_units is None:
        return PreparedInput(normalized, source_token_count, 0)
    unit_count = count_units(normalized)
    if unit_count > limits.max_units:
        raise TranslationInputError("INPUT_TOO_LONG", limits.over_limit_message)
    return PreparedInput(normalized, source_token_count, unit_count)


def prepare_input(text: str) -> PreparedInput:
    return prepare_for(text, E1_LIMITS, _count_e1_positions)


def _disable_fairseq_encoder_fastpath() -> None:
    """Apply the same PyTorch compatibility guard used by CLI generation."""
    from fairseq.modules.transformer_layer import TransformerEncoderLayerBase

    if getattr(TransformerEncoderLayerBase, "_group10_fastpath_patch", False):
        return
    original_init = TransformerEncoderLayerBase.__init__

    def compatible_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.can_use_fastpath = False

    TransformerEncoderLayerBase.__init__ = compatible_init
    TransformerEncoderLayerBase._group10_fastpath_patch = True


class E1Translator:
    model_id = MODEL_ID

    def __init__(self, checkpoint_path: Path, data_bin_path: Path, device: str) -> None:
        _disable_fairseq_encoder_fastpath()
        import torch
        from fairseq.models.transformer import TransformerModel

        if device == "cuda" and not torch.cuda.is_available():
            raise ModelArtifactError("CUDA was requested but is not available.")

        self.device = device
        self._torch = torch
        self._hub = TransformerModel.from_pretrained(
            str(checkpoint_path.parent),
            checkpoint_file=checkpoint_path.name,
            data_name_or_path=str(data_bin_path),
            beam=7,
            lenpen=1.0,
            max_len_a=1.2,
            max_len_b=10,
        )
        self._hub.eval()
        if device == "cuda":
            self._hub.cuda()

        self.parameter_count = sum(
            parameter.numel()
            for model in self._hub.models
            for parameter in model.parameters()
        )
        if self.parameter_count != EXPECTED_PARAMETER_COUNT:
            raise ModelArtifactError(
                "The supplied checkpoint/data-bin do not match the expected E1 model "
                f"({self.parameter_count:,} parameters loaded; "
                f"expected {EXPECTED_PARAMETER_COUNT:,})."
            )

    def translate(self, text: str) -> TranslationResult:
        prepared = prepare_input(text)
        encoded = self._hub.encode(prepared.text)
        if int(encoded.numel()) > MAX_SOURCE_POSITIONS:
            raise TranslationInputError(
                "INPUT_TOO_LONG",
                "E1 hỗ trợ tối đa 256 vị trí mã hoá, bao gồm token kết thúc câu.",
            )

        source_dictionary = self._hub.task.source_dictionary
        unknown_tokens: list[str] = []
        seen: set[str] = set()
        for token in prepared.text.split():
            if source_dictionary.index(token) == source_dictionary.unk() and token not in seen:
                seen.add(token)
                unknown_tokens.append(token)

        started = perf_counter()
        with self._torch.inference_mode():
            generated = self._hub.translate(
                prepared.text,
                beam=7,
                lenpen=1.0,
                max_len_a=1.2,
                max_len_b=10,
            )
        latency_ms = round((perf_counter() - started) * 1000)
        translation = detokenize_chinese(str(generated))
        return TranslationResult(
            translation=translation,
            normalized_input=prepared.text,
            source_token_count=prepared.source_token_count,
            target_token_count=len(translation),
            unknown_tokens=unknown_tokens,
            latency_ms=latency_ms,
        )
