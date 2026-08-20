from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

from .config import load_yaml, portable_project_path, repo_path
from .data import dataset_fingerprint, load_dataset_config, read_split
from .io_utils import base_manifest, package_versions, sha256_file, write_json, write_jsonl
from .normalize import clean_llm_generation, contains_thinking, score_form_chinese
from .schema import PredictionRecord


def _torch_dtype(name: str):
    import torch

    values = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    try:
        return values[name]
    except KeyError as error:
        raise ValueError(f"Unsupported compute dtype: {name}") from error


def _quantization_config(config: dict[str, Any]):
    from transformers import BitsAndBytesConfig

    quant = config["quantization"]
    return BitsAndBytesConfig(
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["quant_type"],
        bnb_4bit_use_double_quant=quant["double_quant"],
        bnb_4bit_compute_dtype=_torch_dtype(quant["compute_dtype"]),
    )


QLORA_BACKENDS = {"qlora", "qlora_knowledge"}


def knowledge_augmenter(config: dict[str, Any]):
    """The retriever for knowledge-augmented backends; None for plain QLoRA."""
    if config.get("backend") != "qlora_knowledge":
        return None
    from .knowledge import KnowledgeAugmenter

    return KnowledgeAugmenter(config)


def _hint_for(augmenter, source: str) -> str:
    return "" if augmenter is None else augmenter.render_hint(source).source


def _prompt_messages(
    config: dict[str, Any], source: str, hint: str = ""
) -> list[dict[str, str]]:
    user = config["prompt"]["user_template"].format(source=source)
    if hint:
        user = f"{user}\n{hint}"
    return [
        {"role": "system", "content": config["prompt"]["system"]},
        {"role": "user", "content": user},
    ]


def _encoded_examples(
    config: dict[str, Any], tokenizer, split: str, augmenter=None
) -> list[dict[str, list[int]]]:
    dataset = load_dataset_config(config["dataset_config"])
    max_length = config["model"]["max_sequence_length"]
    examples: list[dict[str, list[int]]] = []
    for row in read_split(dataset, split):
        messages = _prompt_messages(config, row.source, _hint_for(augmenter, row.source))
        prefix_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_text = tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": row.reference}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
        prefix_ids = tokenizer(prefix_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length,
        )["input_ids"]
        if full_ids[: len(prefix_ids)] != prefix_ids:
            raise ValueError(
                f"Chat-template prefix mismatch for {row.sample_id}; refusing unsafe response masking"
            )
        if len(prefix_ids) >= len(full_ids):
            raise ValueError(f"Target was fully truncated for {row.sample_id}")
        labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids) :]
        examples.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})
    return examples


def _pack_examples(
    examples: list[dict[str, list[int]]], max_length: int, seed: int
) -> list[dict[str, list[int]]]:
    """Greedily pack complete chat examples without masking response tokens."""
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    packed: list[dict[str, list[int]]] = []
    current = {"input_ids": [], "attention_mask": [], "labels": []}
    for example in shuffled:
        length = len(example["input_ids"])
        if length > max_length:
            raise ValueError("Example exceeds max sequence length after tokenization")
        if current["input_ids"] and len(current["input_ids"]) + length > max_length:
            packed.append(current)
            current = {"input_ids": [], "attention_mask": [], "labels": []}
        for key in current:
            current[key].extend(example[key])
    if current["input_ids"]:
        packed.append(current)
    return packed


def _trainer_checkpoints(work_dir: Path) -> list[Path]:
    checkpoints: list[tuple[int, Path]] = []
    for path in (work_dir / "trainer").glob("checkpoint-*"):
        try:
            step = int(path.name.rsplit("-", 1)[1])
        except ValueError:
            continue
        if path.is_dir():
            checkpoints.append((step, path))
    return [path for _, path in sorted(checkpoints)]


def ensure_clean_adapter_dir(
    checkpoint_dir: Path,
    config: dict[str, Any],
    work_dir: Path | None = None,
) -> Path | None:
    """Refuse overwrite, or resolve the exact Trainer checkpoint for a resume."""
    adapter_config = checkpoint_dir / "adapter" / "adapter_config.json"
    scoped_work = work_dir if work_dir is not None else checkpoint_dir
    trainer_checkpoints = _trainer_checkpoints(scoped_work)
    if config.get("resume"):
        if not trainer_checkpoints:
            raise RuntimeError(
                "resume: true requires an existing work/trainer/checkpoint-* directory"
            )
        return trainer_checkpoints[-1]
    if adapter_config.exists():
        raise RuntimeError(
            f"Refusing to train: {adapter_config.parent} already holds a trained adapter "
            "that this run would overwrite. Remove it for a clean run, or set "
            "resume: true in the config."
        )
    if trainer_checkpoints:
        raise RuntimeError(
            f"Refusing to train: {trainer_checkpoints[-1]} contains unfinished Trainer "
            "state. Set resume: true to continue it or use a clean run directory."
        )
    return None


def train_qlora(config_path: str | Path) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("E2 QLoRA training requires a CUDA GPU")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    config = load_yaml(config_path)
    if config.get("backend") not in QLORA_BACKENDS:
        raise ValueError(f"QLoRA training requires one of {sorted(QLORA_BACKENDS)}")
    set_seed(config["seed"])
    augmenter = knowledge_augmenter(config)
    model_config = config["model"]
    checkpoint_dir = repo_path(config["checkpoint_dir"])
    work_dir = repo_path(config["work_dir"])
    resume_checkpoint = ensure_clean_adapter_dir(checkpoint_dir, config, work_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        trust_remote_code=model_config["trust_remote_code"],
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        revision=model_config["revision"],
        trust_remote_code=model_config["trust_remote_code"],
        quantization_config=_quantization_config(config),
        device_map="auto",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=config["training"]["gradient_checkpointing"]
    )
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            bias=lora["bias"],
            target_modules=lora["target_modules"],
        ),
    )
    encoded_train = _encoded_examples(config, tokenizer, "train", augmenter)
    training = config["training"]
    # Packing concatenates examples into one window with no attention reset, so a
    # sequence can see the previous example's knowledge hints while generating its own
    # translation. Harmless at E2's ~107-token mean; corrupting once a hint block is
    # present. Knowledge runs therefore train one example per sequence, padded.
    pack = training.get("pack_examples", augmenter is None)
    train_examples = (
        _pack_examples(encoded_train, model_config["max_sequence_length"], config["seed"])
        if pack
        else encoded_train
    )
    validation_examples = _encoded_examples(config, tokenizer, "val", augmenter)
    arguments = TrainingArguments(
        output_dir=str(work_dir / "trainer"),
        num_train_epochs=training["epochs"],
        per_device_train_batch_size=training["per_device_batch_size"],
        per_device_eval_batch_size=training["per_device_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        lr_scheduler_type=training["scheduler"],
        warmup_ratio=training["warmup_ratio"],
        weight_decay=training["weight_decay"],
        max_grad_norm=training["max_grad_norm"],
        logging_steps=training["logging_steps"],
        eval_strategy="steps",
        eval_steps=training["eval_steps"],
        save_strategy="steps",
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        optim="paged_adamw_8bit",
        fp16=training["fp16"],
        bf16=training["bf16"],
        gradient_checkpointing=training["gradient_checkpointing"],
        report_to=[],
        seed=config["seed"],
        data_seed=config["seed"],
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_examples,
        eval_dataset=validation_examples,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            label_pad_token_id=-100,
            return_tensors="pt",
        ),
    )
    result = trainer.train(
        resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None
    )
    adapter_dir = checkpoint_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.save_state()
    write_json(work_dir / "training_history.json", trainer.state.log_history)
    resolved_revision = getattr(model.config, "_commit_hash", None) or model_config["revision"]
    manifest = base_manifest()
    manifest.update({
        "experiment_id": config["experiment_id"],
        "backend": config["backend"],
        "config": config,
        "config_sha256": sha256_file(repo_path(config_path)),
        "resolved_model_revision": resolved_revision,
        "adapter": str(adapter_dir),
        "adapter_config_sha256": sha256_file(adapter_dir / "adapter_config.json"),
        "train_metrics": result.metrics,
        "resumed_from": str(resume_checkpoint) if resume_checkpoint else None,
        "packing_enabled": pack,
        "train_sequences": len(train_examples),
        "train_tokens": sum(len(example["input_ids"]) for example in train_examples),
        "train_examples_after_packing": len(train_examples),
        "dataset": dataset_fingerprint(config["dataset_config"]),
        "packages": package_versions(
            ["torch", "transformers", "peft", "bitsandbytes", "accelerate", "datasets"]
        ),
    })
    if augmenter is not None:
        manifest["knowledge"] = augmenter.prompt_metadata()
    write_json(work_dir / "run_manifest.json", manifest)


def predict_qlora(config_path: str | Path, split: str) -> Path:
    import torch

    if split not in {"val", "test"}:
        raise ValueError("Predictions are allowed only for val or test")
    if not torch.cuda.is_available():
        raise RuntimeError("E2 prediction requires a CUDA GPU")
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    config = load_yaml(config_path)
    if split == "test":
        from .freeze import ensure_selection_frozen

        ensure_selection_frozen(config_path)
    set_seed(config["seed"])
    # Must match training exactly: a prompt shape that differs between fit and decode
    # is the defect that made E2's reported score unreproducible.
    augmenter = knowledge_augmenter(config)
    adapter_dir = repo_path(config["checkpoint_dir"]) / "adapter"
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Missing trained adapter: {adapter_dir}")
    model_config = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    manifest_path = repo_path(config["work_dir"]) / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing training manifest: {manifest_path}")
    import json

    with manifest_path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    resolved_revision = manifest["resolved_model_revision"]
    base = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        revision=resolved_revision,
        trust_remote_code=model_config["trust_remote_code"],
        quantization_config=_quantization_config(config),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    dataset = load_dataset_config(config["dataset_config"])
    rows = read_split(dataset, split)
    decoding = config["decoding"]
    records = []
    for row in rows:
        hint = _hint_for(augmenter, row.source)
        prompt = tokenizer.apply_chat_template(
            _prompt_messages(config, row.source, hint),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        source_token_count = len(tokenizer(row.source, add_special_tokens=False)["input_ids"])
        max_new_tokens = min(
            decoding["max_new_tokens_cap"],
            math.ceil(decoding["output_length_multiplier"] * source_token_count)
            + decoding["output_length_offset"],
        )
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                num_beams=decoding["num_beams"],
                do_sample=decoding["do_sample"],
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output_ids[0, inputs["input_ids"].shape[-1] :]
        raw = tokenizer.decode(generated, skip_special_tokens=True).strip()
        if contains_thinking(raw):
            raise ValueError(f"Thinking content emitted for {row.sample_id}")
        prediction = clean_llm_generation(raw)
        if not prediction:
            raise ValueError(f"Empty prediction for {row.sample_id}")
        records.append(
            PredictionRecord(
                sample_id=row.sample_id,
                split=split,
                source=row.source,
                reference=row.reference,
                prediction=prediction,
                prediction_raw=raw,
                prediction_scored=score_form_chinese(prediction),
                experiment_id=config["experiment_id"],
                model_name=model_config["name"],
                checkpoint=portable_project_path(adapter_dir),
                decoding_config=decoding,
                knowledge_hint=hint or None,
            ).to_dict()
        )
    if augmenter is not None:
        # A config typo (backend: qlora in an E4 file) would silently disable retrieval
        # and yield 510 hint-free predictions that pass every downstream gate while
        # looking like a legitimate knowledge run. Preflight measures ~100% hit rate.
        hinted = sum(1 for record in records if record.get("knowledge_hint"))
        if hinted < len(records) // 2:
            raise RuntimeError(
                f"Knowledge backend produced hints for only {hinted}/{len(records)} rows; "
                "retrieval is not working and these predictions are not a knowledge run."
            )
    output = repo_path(config["prediction_dir"]) / f"{config['experiment_id']}.{split}.jsonl"
    write_jsonl(output, records)
    return output
