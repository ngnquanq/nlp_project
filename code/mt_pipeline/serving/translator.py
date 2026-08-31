from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

from mt_pipeline.normalize import detokenize_chinese, normalize_whitespace


MODEL_ID = "e1_fairseq_vi_zh_v1"
EXPECTED_PARAMETER_COUNT = 67_652_608
MAX_SOURCE_POSITIONS = 256
MAX_INPUT_CHARACTERS = 4_000


class TranslationInputError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ModelArtifactError(RuntimeError):
    pass


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
    parameter_count: int

    def translate(self, text: str) -> TranslationResult: ...


def prepare_input(text: str) -> PreparedInput:
    normalized = normalize_whitespace(text)
    if not normalized:
        raise TranslationInputError("EMPTY_INPUT", "Hãy nhập câu tiếng Việt cần dịch.")
    if len(normalized) > MAX_INPUT_CHARACTERS:
        raise TranslationInputError(
            "INPUT_TOO_LONG",
            f"Văn bản không được vượt quá {MAX_INPUT_CHARACTERS:,} ký tự.",
        )
    source_token_count = len(normalized.split())
    encoded_position_count = source_token_count + 1  # Fairseq appends EOS.
    if encoded_position_count > MAX_SOURCE_POSITIONS:
        raise TranslationInputError(
            "INPUT_TOO_LONG",
            "E1 hỗ trợ tối đa 256 vị trí mã hoá, bao gồm token kết thúc câu.",
        )
    return PreparedInput(normalized, source_token_count, encoded_position_count)


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
