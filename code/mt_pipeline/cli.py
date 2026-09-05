from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_yaml, repo_path
from .data import audit_dataset
from .error_analysis import prepare_annotation_template, summarize_annotations
from .evaluation import DEFAULT_PROTOCOL, PROTOCOLS, compare_predictions, evaluate_predictions


def _backend(config_path: str) -> str:
    return str(load_yaml(config_path).get("backend"))


def _train(config_path: str) -> None:
    backend = _backend(config_path)
    if backend in {"fairseq", "fairseq_knowledge"}:
        from .fairseq_runner import train_fairseq

        train_fairseq(config_path)
    elif backend in {"qlora", "qlora_knowledge"}:
        from .llm_runner import ensure_clean_adapter_dir, train_qlora
        from .runtime import tee_output

        config = load_yaml(config_path)
        checkpoint_dir = repo_path(config["checkpoint_dir"])
        work_dir = repo_path(config["work_dir"])
        ensure_clean_adapter_dir(checkpoint_dir, config, work_dir)
        with tee_output(work_dir / "train.log"):
            train_qlora(config_path)
    elif backend == "knowledge_v2":
        config = load_yaml(config_path)
        raise RuntimeError(config.get("blocked_reason", "knowledge_v2 is blocked"))
    else:
        raise ValueError(f"Unsupported backend: {backend}")


def _predict(config_path: str, split: str) -> Path:
    backend = _backend(config_path)
    if backend in {"fairseq", "fairseq_knowledge"}:
        from .fairseq_runner import predict_fairseq

        return predict_fairseq(config_path, split)
    if backend in {"qlora", "qlora_knowledge"}:
        from .llm_runner import predict_qlora
        from .runtime import tee_output

        config = load_yaml(config_path)
        if split == "test":
            from .freeze import ensure_selection_frozen

            ensure_selection_frozen(config_path)
        work_dir = repo_path(config["work_dir"])
        with tee_output(work_dir / f"generate.{split}.log"):
            return predict_qlora(config_path, split)
    raise ValueError(f"Backend does not support prediction: {backend}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Group 10 Vietnamese-to-Chinese MT pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("data-audit", help="Validate, hash, and profile the corpus")
    audit.add_argument("--config", required=True)
    audit.add_argument("--output", required=True)

    prepare = subparsers.add_parser("prepare-fairseq", help="Create Fairseq text and binary data")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--force", action="store_true")

    train = subparsers.add_parser("train", help="Train the configured experiment")
    train.add_argument("--config", required=True)

    predict = subparsers.add_parser("predict", help="Generate schema-validated predictions")
    predict.add_argument("--config", required=True)
    predict.add_argument("--split", choices=("val", "test"), required=True)

    freeze = subparsers.add_parser("freeze-selection", help="Lock checkpoint/config after validation")
    freeze.add_argument("--config", required=True)
    freeze.add_argument("--validation-predictions", required=True)
    freeze.add_argument("--validation-metrics", required=True)

    evaluate = subparsers.add_parser("evaluate", help="Compute SacreBLEU and chrF++")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)

    compare = subparsers.add_parser("compare", help="Paired bootstrap system comparison")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--candidate", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--samples", type=int, default=1000)
    compare.add_argument("--seed", type=int, default=12345)
    compare.add_argument("--protocol", choices=PROTOCOLS, default=DEFAULT_PROTOCOL)

    rescore = subparsers.add_parser("rescore-moses", help="Rescore saved predictions without training")
    rescore.add_argument("--prediction-dir", default="predictions")
    rescore.add_argument("--metrics-dir", default="metrics")
    rescore.add_argument("--output-dir", default="metrics/moses")

    errors = subparsers.add_parser("prepare-error-analysis", help="Sample shared test IDs")
    errors.add_argument("--predictions", nargs="+", required=True)
    errors.add_argument("--output", required=True)
    errors.add_argument("--sample-size", type=int, default=100)
    errors.add_argument("--seed", type=int, default=42)

    summary = subparsers.add_parser("summarize-error-analysis", help="Validate and aggregate annotations")
    summary.add_argument("--annotations", required=True)
    summary.add_argument("--output", required=True)

    derive = subparsers.add_parser(
        "derive-run-config", help="Relocate a config's output dirs into a rerun tree"
    )
    derive.add_argument("--config", required=True)
    derive.add_argument("--run-root", required=True)
    derive.add_argument("--output")
    derive.add_argument(
        "--gpu",
        action="store_true",
        help="Force training.cpu=false and training.fp16=true (match E1/E3 runtime path)",
    )

    runs = subparsers.add_parser(
        "compare-runs", help="Check two runs produced the same weights and predictions"
    )
    runs.add_argument("--baseline-checkpoint", required=True)
    runs.add_argument("--candidate-checkpoint", required=True)
    runs.add_argument("--baseline-predictions", required=True)
    runs.add_argument("--candidate-predictions", required=True)
    runs.add_argument("--output")

    repro = subparsers.add_parser(
        "repro-check", help="Verify frozen selections, stored metrics, and dataset hashes"
    )
    repro.add_argument("--scratch-dir", required=True)
    repro.add_argument("--output")

    preflight = subparsers.add_parser(
        "knowledge-preflight", help="Measure token/runtime cost per candidate budget"
    )
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--candidates", default="8,16,24,32")
    preflight.add_argument("--split", default="train")
    preflight.add_argument("--output")
    preflight.add_argument("--smoke-rows", type=int, default=0)
    preflight.add_argument("--smoke-adapter")

    status = subparsers.add_parser("project-status", help="Report completed and missing artifacts")
    status.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/e1_fairseq.yaml",
            "configs/e2_qwen3_qlora.yaml",
            "configs/e3_fairseq_knowledge.yaml",
            "configs/e3_custom_fairseq_knowledge.yaml",
            "configs/e4_qwen3_knowledge.yaml",
        ],
    )
    status.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "data-audit":
        value = audit_dataset(args.config, args.output)
        print(json.dumps({"output": args.output, "dataset_id": value["dataset_id"]}, ensure_ascii=False))
    elif args.command == "prepare-fairseq":
        from .fairseq_runner import prepare_fairseq

        paths = prepare_fairseq(args.config, args.force)
        print(json.dumps({name: str(path) for name, path in paths.items()}))
    elif args.command == "train":
        _train(args.config)
    elif args.command == "predict":
        print(_predict(args.config, args.split))
    elif args.command == "freeze-selection":
        from .freeze import freeze_selection

        print(freeze_selection(
            args.config, args.validation_predictions, args.validation_metrics
        ))
    elif args.command == "evaluate":
        value = evaluate_predictions(args.predictions, args.output, args.protocol)
        print(json.dumps(value["metrics"], ensure_ascii=False))
    elif args.command == "compare":
        value = compare_predictions(
            args.baseline, args.candidate, args.output, args.samples, args.seed, args.protocol
        )
        print(json.dumps(value["metrics"], ensure_ascii=False))
    elif args.command == "rescore-moses":
        from .rescore import rescore_moses

        print(json.dumps(rescore_moses(args.prediction_dir, args.metrics_dir, args.output_dir),
                         ensure_ascii=False, indent=2))
    elif args.command == "prepare-error-analysis":
        print(prepare_annotation_template(
            args.predictions, args.output, args.sample_size, args.seed
        ))
    elif args.command == "summarize-error-analysis":
        value = summarize_annotations(args.annotations, args.output)
        print(json.dumps(value, ensure_ascii=False))
    elif args.command == "derive-run-config":
        from .reruns import derive_run_config

        print(derive_run_config(args.config, args.run_root, args.output, args.gpu))
    elif args.command == "compare-runs":
        from .reruns import compare_runs

        value = compare_runs(
            args.baseline_checkpoint,
            args.candidate_checkpoint,
            args.baseline_predictions,
            args.candidate_predictions,
            args.output,
        )
        print(json.dumps(value, ensure_ascii=False, indent=2))
        if not value["reproducible"]:
            raise SystemExit("Runs diverged; see the report above")
    elif args.command == "repro-check":
        from .repro_check import run_checks

        value = run_checks(args.scratch_dir, args.output)
        print(json.dumps(value, ensure_ascii=False, indent=2))
        if not value["ok"]:
            raise SystemExit("Reproducibility check failed; see the report above")
    elif args.command == "knowledge-preflight":
        from .preflight import knowledge_preflight

        value = knowledge_preflight(
            args.config,
            [int(v) for v in args.candidates.split(",")],
            args.split,
            args.output,
            args.smoke_rows,
            args.smoke_adapter,
        )
        print(json.dumps(value, ensure_ascii=False, indent=2))
    elif args.command == "project-status":
        from .status import project_status

        print(json.dumps(project_status(args.configs, args.output), ensure_ascii=False, indent=2))
    else:
        raise AssertionError(args.command)
