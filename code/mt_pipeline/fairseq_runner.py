from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .config import load_yaml, portable_project_path, repo_path
from .data import dataset_fingerprint, load_dataset_config, read_split
from .io_utils import base_manifest, package_versions, sha256_file, write_json, write_jsonl
from .normalize import detokenize_chinese, score_form_chinese
from .runtime import determinism_env, resolved_env, run_logged
from .schema import PredictionRecord


def _paths(config: dict[str, Any]) -> dict[str, Path]:
    work = repo_path(config["work_dir"])
    return {
        "work": work,
        "text": work / "text",
        "binary": work / "data-bin",
        "checkpoint": repo_path(config["checkpoint_dir"]),
    }


def prepare_fairseq(config_path: str | Path, force: bool = False) -> dict[str, Path]:
    config = load_yaml(config_path)
    backend = config.get("backend")
    if backend not in {"fairseq", "fairseq_knowledge"}:
        raise ValueError("Fairseq preparation requires a Fairseq backend")
    dataset = load_dataset_config(config["dataset_config"])
    paths = _paths(config)
    paths["text"].mkdir(parents=True, exist_ok=True)
    augmenter = None
    augmentation_report: dict[str, Any] | None = None
    if backend == "fairseq_knowledge":
        from .knowledge import KnowledgeAugmenter

        augmenter = KnowledgeAugmenter(config)
        augmentation_report = {"knowledge": augmenter.metadata(), "splits": {}}
    for split, fairseq_split in (("train", "train"), ("val", "valid"), ("test", "test")):
        rows = read_split(dataset, split)
        sources = []
        candidate_counts = []
        matched_spans = []
        for row in rows:
            if augmenter is None:
                sources.append(row.source)
                continue
            result = augmenter.augment(row.source)
            sources.append(result.source)
            candidate_counts.append(result.candidate_count)
            matched_spans.append(result.matched_spans)
        source_path = paths["text"] / f"{fairseq_split}.vi"
        source_path.write_text("\n".join(sources) + "\n", encoding="utf-8")
        (paths["text"] / f"{fairseq_split}.zh").write_text(
            "\n".join(row.reference_tokenized for row in rows) + "\n", encoding="utf-8"
        )
        if augmentation_report is not None:
            augmentation_report["splits"][split] = {
                "rows": len(rows),
                "augmented_rows": sum(count > 0 for count in candidate_counts),
                "candidate_count_mean": round(sum(candidate_counts) / len(rows), 6),
                "candidate_count_max": max(candidate_counts, default=0),
                "matched_spans_mean": round(sum(matched_spans) / len(rows), 6),
                "source_sha256": sha256_file(source_path),
            }
    if augmentation_report is not None:
        write_json(paths["work"] / "knowledge_augmentation.json", augmentation_report)
    if paths["binary"].exists() and not force:
        return paths
    if paths["binary"].exists():
        shutil.rmtree(paths["binary"])
    preprocess = config["preprocess"]
    command = [
        "fairseq-preprocess",
        "--source-lang",
        preprocess["source_lang"],
        "--target-lang",
        preprocess["target_lang"],
        "--trainpref",
        str(paths["text"] / "train"),
        "--validpref",
        str(paths["text"] / "valid"),
        "--testpref",
        str(paths["text"] / "test"),
        "--destdir",
        str(paths["binary"]),
        "--thresholdsrc",
        str(preprocess["threshold_source"]),
        "--thresholdtgt",
        str(preprocess["threshold_target"]),
        "--workers",
        str(preprocess["workers"]),
    ]
    run_logged(command, paths["work"] / "preprocess.log", cwd=repo_path("."))
    return paths


def ensure_clean_checkpoint_dir(checkpoint_dir: Path, config: dict[str, Any]) -> None:
    """Refuse to silently warm-start from a previous run's checkpoint.

    Fairseq defaults ``--restore-file`` to ``checkpoint_last.pt``, so re-running
    training resumes instead of starting over and yields a different model from the
    same command. Set ``resume: true`` in the config to opt in deliberately.
    """
    last = checkpoint_dir / "checkpoint_last.pt"
    if last.exists() and not config.get("resume"):
        raise RuntimeError(
            f"Refusing to train: {last} already exists and would be resumed from, "
            "producing a model that the same command cannot reproduce. Remove the "
            "checkpoint directory for a clean run, or set 'resume: true' in the config."
        )


def train_fairseq(config_path: str | Path) -> None:
    config = load_yaml(config_path)
    paths = prepare_fairseq(config_path)
    paths["checkpoint"].mkdir(parents=True, exist_ok=True)
    ensure_clean_checkpoint_dir(paths["checkpoint"], config)
    model, training, decoding = config["model"], config["training"], config["decoding"]
    command = [
        sys.executable, "-m", "mt_pipeline.fairseq_train",
        str(paths["binary"]),
        "--source-lang", "vi",
        "--target-lang", "zh",
        "--save-dir", str(paths["checkpoint"]),
        "--arch", model["architecture"],
        "--encoder-layers", str(model["encoder_layers"]),
        "--decoder-layers", str(model["decoder_layers"]),
        "--encoder-embed-dim", str(model["encoder_embed_dim"]),
        "--decoder-embed-dim", str(model["decoder_embed_dim"]),
        "--encoder-ffn-embed-dim", str(model["encoder_ffn_embed_dim"]),
        "--decoder-ffn-embed-dim", str(model["decoder_ffn_embed_dim"]),
        "--encoder-attention-heads", str(model["encoder_attention_heads"]),
        "--decoder-attention-heads", str(model["decoder_attention_heads"]),
        "--dropout", str(model["dropout"]),
        "--optimizer", training["optimizer"],
        "--adam-betas", training["adam_betas"],
        "--lr", str(training["learning_rate"]),
        "--lr-scheduler", training["scheduler"],
        "--warmup-updates", str(training["warmup_updates"]),
        "--criterion", "label_smoothed_cross_entropy",
        "--label-smoothing", str(training["label_smoothing"]),
        "--weight-decay", str(training["weight_decay"]),
        "--clip-norm", str(training["clip_norm"]),
        "--max-tokens", str(training["max_tokens_per_gpu"]),
        "--num-workers", str(training.get("num_workers", 1)),
        "--update-freq", str(training["update_freq"]),
        "--distributed-world-size", str(training["distributed_world_size"]),
        "--max-update", str(training["max_update"]),
        "--max-epoch", str(training["max_epoch"]),
        "--patience", str(training["patience"]),
        "--max-source-positions", str(training["max_source_positions"]),
        "--max-target-positions", str(training["max_target_positions"]),
        "--save-interval", str(training["save_interval"]),
        "--keep-best-checkpoints", str(training["keep_best_checkpoints"]),
        "--seed", str(config["seed"]),
        "--eval-bleu",
        "--eval-bleu-detok", "space",
        "--eval-bleu-args", json.dumps({
            "beam": decoding["beam"],
            "lenpen": decoding["lenpen"],
            "max_len_a": decoding["max_len_a"],
            "max_len_b": decoding["max_len_b"],
        }),
        "--best-checkpoint-metric", "bleu",
        "--maximize-best-checkpoint-metric",
        "--log-format", "json",
        "--log-interval", "50",
    ]
    if model.get("share_decoder_input_output_embed"):
        command.append("--share-decoder-input-output-embed")
    if training.get("no_epoch_checkpoints"):
        command.append("--no-epoch-checkpoints")
    if training.get("fp16"):
        command.append("--fp16")
    if training.get("cpu"):
        command.append("--cpu")
    run_logged(
        command,
        paths["work"] / "train.log",
        cwd=repo_path("."),
        env=determinism_env(config["seed"]),
    )
    checkpoint = paths["checkpoint"] / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise RuntimeError(f"Training finished without {checkpoint}")
    manifest = base_manifest()
    manifest.update({
        "experiment_id": config["experiment_id"],
        "backend": config["backend"],
        "config": config,
        "config_sha256": sha256_file(repo_path(config_path)),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "dataset": dataset_fingerprint(config["dataset_config"]),
        "packages": package_versions(["fairseq", "torch", "sacrebleu", "PyYAML"]),
    })
    if config["backend"] == "fairseq_knowledge":
        with (paths["work"] / "knowledge_augmentation.json").open(encoding="utf-8") as handle:
            manifest["knowledge_augmentation"] = json.load(handle)
    write_json(paths["work"] / "run_manifest.json", manifest)


def _parse_hypotheses(output: str) -> dict[int, str]:
    hypotheses: dict[int, str] = {}
    for line in output.splitlines():
        if not line.startswith("H-"):
            continue
        fields = line.split("\t", 2)
        if len(fields) != 3:
            raise ValueError(f"Malformed fairseq hypothesis: {line}")
        index = int(fields[0][2:])
        hypotheses[index] = fields[2].strip()
    return hypotheses


def predict_fairseq(config_path: str | Path, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("Predictions are allowed only for val or test")
    config = load_yaml(config_path)
    if split == "test":
        from .freeze import ensure_selection_frozen

        ensure_selection_frozen(config_path)
    paths = prepare_fairseq(config_path)
    checkpoint = paths["checkpoint"] / "checkpoint_best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint}")
    decoding = config["decoding"]
    command = [
        sys.executable, "-m", "mt_pipeline.fairseq_generate", str(paths["binary"]),
        "--path", str(checkpoint),
        "--source-lang", "vi",
        "--target-lang", "zh",
        "--gen-subset", "valid" if split == "val" else "test",
        "--beam", str(decoding["beam"]),
        "--lenpen", str(decoding["lenpen"]),
        "--max-len-a", str(decoding["max_len_a"]),
        "--max-len-b", str(decoding["max_len_b"]),
        "--batch-size", str(decoding["batch_size"]),
        "--num-workers", str(config["training"].get("num_workers", 1)),
    ]
    if config["training"].get("cpu"):
        command.append("--cpu")
    # Not run_logged: stdout carries the H- hypothesis lines and must be captured.
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
        env=resolved_env(determinism_env(config["seed"])),
    )
    log_path = paths["work"] / f"generate.{split}.log"
    log_path.write_text(result.stdout + "\nSTDERR\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"Fairseq generation failed; inspect {log_path}")
    hypotheses = _parse_hypotheses(result.stdout)
    dataset = load_dataset_config(config["dataset_config"])
    rows = read_split(dataset, split)
    if set(hypotheses) != set(range(len(rows))):
        raise ValueError(
            f"Expected hypothesis IDs 0..{len(rows)-1}; found {len(hypotheses)} IDs"
        )
    records = []
    for index, row in enumerate(rows):
        raw = hypotheses[index]
        prediction = detokenize_chinese(raw)
        record = PredictionRecord(
            sample_id=row.sample_id,
            split=split,
            source=row.source,
            reference=row.reference,
            prediction=prediction,
            prediction_raw=raw,
            prediction_scored=score_form_chinese(prediction),
            experiment_id=config["experiment_id"],
            model_name=(
                "fairseq-transformer+knowledge-retrieval"
                if config["backend"] == "fairseq_knowledge"
                else "fairseq-transformer"
            ),
            checkpoint=portable_project_path(checkpoint),
            decoding_config=decoding,
        )
        records.append(record.to_dict())
    output = repo_path(config["prediction_dir"]) / f"{config['experiment_id']}.{split}.jsonl"
    write_jsonl(output, records)
    return output
