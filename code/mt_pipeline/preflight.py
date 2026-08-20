"""Measure the cost of a knowledge-augmented QLoRA run before committing GPU hours.

Knowledge hints are cheap in Fairseq positions but expensive in Qwen tokens: 64
candidates add ~163 tokens to a source averaging 25. Since every added token is paid
for over three epochs, the candidate budget has to be chosen from measured totals
rather than intuition, and a run that cannot finish in one session should be caught
here rather than at hour nine.

Counts are exact — this loads the real tokenizer and encodes the real split.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .config import load_yaml, repo_path
from .io_utils import write_json


# E2's measured baseline: 2,061,379 encoded train tokens trained in 20,328 s.
E2_REFERENCE_TOKENS = 2_061_379
E2_REFERENCE_SECONDS = 20_328.3189


def _budget_report(
    config: dict[str, Any], tokenizer, candidates: int, split: str
) -> dict[str, Any]:
    from .llm_runner import _encoded_examples, knowledge_augmenter

    scoped = copy.deepcopy(config)
    scoped["knowledge"]["max_candidates"] = candidates
    augmenter = knowledge_augmenter(scoped)
    examples = _encoded_examples(scoped, tokenizer, split, augmenter)

    lengths = [len(example["input_ids"]) for example in examples]
    total = sum(lengths)
    max_length = scoped["model"]["max_sequence_length"]
    training = scoped["training"]
    effective_batch = (
        training["per_device_batch_size"] * training["gradient_accumulation_steps"]
    )
    # No packing for knowledge runs: one example per sequence.
    updates = -(-len(examples) // effective_batch) * training["epochs"]
    return {
        "max_candidates": candidates,
        "sequences": len(examples),
        "total_tokens": total,
        "mean_tokens": round(total / len(examples), 1),
        "max_tokens": max(lengths),
        "over_window": sum(1 for length in lengths if length >= max_length),
        "tokens_vs_e2": round(total / E2_REFERENCE_TOKENS, 3),
        "projected_updates": updates,
        "projected_hours": round(
            total / E2_REFERENCE_TOKENS * E2_REFERENCE_SECONDS / 3600, 2
        ),
    }


def prompt_smoke_test(config: dict[str, Any], adapter_dir: Path, rows: int) -> dict[str, Any]:
    """Decode a handful of val rows with the hint format, using an existing adapter.

    Checks the failure modes that would abort a full decode hours in: `predict_qlora`
    raises on any `<think>` output and on an empty prediction, and
    `clean_llm_generation` only strips a fixed prefix list. Translation quality here is
    meaningless — the adapter was fit without hints — but the guards are real.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .data import load_dataset_config, read_split
    from .llm_runner import _hint_for, _prompt_messages, knowledge_augmenter
    from .normalize import clean_llm_generation, contains_thinking

    augmenter = knowledge_augmenter(config)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    from .llm_runner import _quantization_config

    base = AutoModelForCausalLM.from_pretrained(
        config["model"]["name"],
        revision=config["model"]["revision"],
        trust_remote_code=config["model"]["trust_remote_code"],
        quantization_config=_quantization_config(config),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()

    dataset = load_dataset_config(config["dataset_config"])
    findings = []
    for row in read_split(dataset, "val")[:rows]:
        hint = _hint_for(augmenter, row.source)
        prompt = tokenizer.apply_chat_template(
            _prompt_messages(config, row.source, hint),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                num_beams=1,
                do_sample=False,
                max_new_tokens=64,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        raw = tokenizer.decode(
            output[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()
        findings.append({
            "sample_id": row.sample_id,
            "hint_present": bool(hint),
            "emitted_thinking": contains_thinking(raw),
            "empty_after_cleaning": not clean_llm_generation(raw),
            "raw_preview": raw[:60],
        })
    return {
        "rows": len(findings),
        "thinking_failures": sum(f["emitted_thinking"] for f in findings),
        "empty_failures": sum(f["empty_after_cleaning"] for f in findings),
        "samples": findings,
    }


def knowledge_preflight(
    config_path: str | Path,
    candidates: list[int],
    split: str = "train",
    output_path: str | Path | None = None,
    smoke_rows: int = 0,
    smoke_adapter: str | Path | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    config = load_yaml(config_path)
    if config.get("backend") != "qlora_knowledge":
        raise ValueError("Preflight applies to backend: qlora_knowledge")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        revision=config["model"]["revision"],
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    budgets = [_budget_report(config, tokenizer, value, split) for value in candidates]
    report = {
        "experiment_id": config["experiment_id"],
        "split": split,
        "reference": {
            "experiment_id": "e2_qwen3_8b_qlora_vi_zh_v1",
            "train_tokens": E2_REFERENCE_TOKENS,
            "train_seconds": E2_REFERENCE_SECONDS,
        },
        "budgets": budgets,
    }
    if smoke_rows:
        if smoke_adapter is None:
            raise ValueError("--smoke-rows requires --smoke-adapter")
        report["prompt_smoke_test"] = prompt_smoke_test(
            config, repo_path(smoke_adapter), smoke_rows
        )
    if output_path is not None:
        write_json(repo_path(output_path), report)
    return report
