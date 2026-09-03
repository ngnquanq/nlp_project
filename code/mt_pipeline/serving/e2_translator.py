"""Single-sentence E2 (Qwen3-8B QLoRA) inference for the local UI.

`llm_runner.predict_qlora` is split-oriented: it reads a dataset, loops rows and
writes JSONL. This module replays the exact same per-sentence recipe for one
string, reusing its private prompt/quantization helpers so the UI cannot drift
from the offline pipeline.

Only stdlib and torch-free modules are imported at module scope, so the file
also imports under the Fairseq gateway environment and its pure logic stays
unit-testable there.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from time import perf_counter
from typing import Any

from mt_pipeline.config import load_yaml
from mt_pipeline.llm_runner import _prompt_messages, _quantization_config
from mt_pipeline.normalize import clean_llm_generation, contains_thinking
from mt_pipeline.serving.lazy import LazyTranslator
from mt_pipeline.serving.registry import E2_MODEL_ID, e2_limits
from mt_pipeline.serving.settings import UISettings
from mt_pipeline.serving.translator import (
    InputLimits,
    ModelArtifactError,
    TranslationFailure,
    TranslationResult,
    Translator,
)


LOGGER = logging.getLogger(__name__)

QLORA_BACKEND = "qlora"


def read_resolved_revision(manifest_path: Path) -> str:
    """Read the base-model revision the adapter was actually fit against.

    Deliberately never falls back to config["model"]["revision"]: predict_qlora
    reads the manifest, and silently diverging would load a different base model
    than the adapter expects.
    """
    if not manifest_path.is_file():
        raise ModelArtifactError(f"Không tìm thấy run_manifest.json của E2: {manifest_path}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(f"Không đọc được run_manifest.json: {manifest_path}.") from exc
    revision = manifest.get("resolved_model_revision")
    if not revision:
        raise ModelArtifactError(
            f"run_manifest.json thiếu khoá resolved_model_revision: {manifest_path}."
        )
    return str(revision)


def compute_max_new_tokens(decoding: dict[str, Any], source_token_count: int) -> int:
    """The identical budget predict_qlora derives per row."""
    return min(
        decoding["max_new_tokens_cap"],
        math.ceil(decoding["output_length_multiplier"] * source_token_count)
        + decoding["output_length_offset"],
    )


def finalize_generation(raw: str) -> str:
    """predict_qlora's two guards, as typed errors an HTTP layer can map."""
    if contains_thinking(raw):
        raise TranslationFailure(
            "THINKING_EMITTED",
            "E2 trả về nội dung suy luận thay vì bản dịch. Hãy thử lại.",
        )
    prediction = clean_llm_generation(raw)
    if not prediction:
        raise TranslationFailure(
            "EMPTY_GENERATION",
            "E2 không sinh ra bản dịch nào. Hãy rút gọn hoặc diễn đạt lại câu nguồn.",
        )
    return prediction


class E2Preload:
    """The cheap half of E2: config, artifact checks and the tokenizer.

    Built eagerly at sidecar startup so a missing GPU or adapter is reported in
    seconds, and so prompt token counts are exact from the very first request —
    before any weights exist.
    """

    def __init__(self, settings: UISettings) -> None:
        config_path = settings.resolved_e2_config_path()
        if not config_path.is_file():
            raise ModelArtifactError(f"Không tìm thấy config E2: {config_path}.")
        self.config = load_yaml(config_path)
        backend = self.config.get("backend")
        if backend != QLORA_BACKEND:
            raise ModelArtifactError(
                f"Config E2 phải có backend '{QLORA_BACKEND}', đang là '{backend}'."
            )

        self.device = settings.e2_device
        if self.device != "cuda":
            raise ModelArtifactError("E2 yêu cầu GPU CUDA; đặt MT_E2_DEVICE=cuda.")

        import torch

        if not torch.cuda.is_available():
            raise ModelArtifactError(
                "E2 yêu cầu GPU CUDA nhưng torch không thấy thiết bị nào."
            )

        adapter_dir = settings.resolved_e2_adapter_path(self.config)
        if adapter_dir is None or not (adapter_dir / "adapter_config.json").is_file():
            raise ModelArtifactError(
                f"Không tìm thấy adapter E2 (thiếu adapter_config.json): {adapter_dir}."
            )
        self.adapter_dir = adapter_dir

        manifest_path = settings.resolved_e2_manifest_path(self.config)
        if manifest_path is None:
            raise ModelArtifactError("Không xác định được đường dẫn run_manifest.json của E2.")
        self.revision = read_resolved_revision(manifest_path)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.adapter_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        self.tokenizer = tokenizer

        self.limits: InputLimits = e2_limits(settings.e2_max_prompt_tokens)

    def build_prompt(self, source: str) -> str:
        # enable_thinking=False is load-bearing: the adapter's chat template
        # injects an empty <think></think> block when it is false, and training
        # used the same flag.
        return self.tokenizer.apply_chat_template(
            _prompt_messages(self.config, source),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def count_prompt_tokens(self, source: str) -> int:
        prompt = self.build_prompt(source)
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])


class E2Translator:
    """The expensive half: quantized base weights plus the LoRA adapter."""

    model_id = E2_MODEL_ID

    def __init__(self, preload: E2Preload) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, set_seed

        self._torch = torch
        self._preload = preload
        self.device = preload.device
        self.limits = preload.limits
        self.count_units = preload.count_prompt_tokens

        config = preload.config
        model_config = config["model"]
        set_seed(config["seed"])
        started = perf_counter()
        base = AutoModelForCausalLM.from_pretrained(
            model_config["name"],
            revision=preload.revision,
            trust_remote_code=model_config["trust_remote_code"],
            quantization_config=_quantization_config(config),
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, preload.adapter_dir)
        model.eval()
        self._model = model
        self.parameter_count = None  # 4-bit packing makes numel() meaningless.
        LOGGER.info("E2 weights ready in %.1fs", perf_counter() - started)

    def translate(self, text: str) -> TranslationResult:
        preload = self._preload
        tokenizer = preload.tokenizer
        decoding = preload.config["decoding"]

        prompt = preload.build_prompt(text)
        inputs = tokenizer(prompt, return_tensors="pt").to(self._model.device)
        source_token_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        max_new_tokens = compute_max_new_tokens(decoding, source_token_count)

        started = perf_counter()
        try:
            with self._torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    num_beams=decoding["num_beams"],
                    do_sample=decoding["do_sample"],
                    max_new_tokens=max_new_tokens,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
        except self._torch.cuda.OutOfMemoryError as exc:
            self._torch.cuda.empty_cache()
            raise TranslationFailure(
                "GPU_OUT_OF_MEMORY",
                "GPU hết bộ nhớ khi dịch. Hãy thử câu ngắn hơn.",
            ) from exc
        latency_ms = round((perf_counter() - started) * 1000)

        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        translation = finalize_generation(raw)
        return TranslationResult(
            translation=translation,
            normalized_input=text,
            source_token_count=source_token_count,
            target_token_count=len(translation),
            unknown_tokens=[],  # Qwen BPE has no out-of-vocabulary concept.
            latency_ms=latency_ms,
        )


def build_e2_translator(settings: UISettings) -> Translator:
    """Sidecar factory: cheap checks now, weights on the first request."""
    preload = E2Preload(settings)
    return LazyTranslator(
        model_id=E2_MODEL_ID,
        device=preload.device,
        build=lambda: E2Translator(preload),
        count_units=preload.count_prompt_tokens,
        limits=preload.limits,
    )
