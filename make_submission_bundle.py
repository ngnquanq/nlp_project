"""Assemble the private Group 10 artifact bundle for the course submission.

Specification section 13 requires code, configs, dataset description, predictions,
metrics, and the checkpoints (or a private link to them). The source tree itself goes
through the private GitHub repository per docs/GITHUB_PUBLISHING.md; this script builds
the companion archive of generated artifacts that .gitignore deliberately keeps out of
version control.

The file selection mirrors stage_archive() in e2qwen3-qlora.ipynb so a locally built
bundle and a Kaggle-built one carry the same contents.

    python3 make_submission_bundle.py            # build group10_submission_bundle.zip
    python3 make_submission_bundle.py --dry-run  # list the manifest, write nothing

Restricted material is denied by an explicit blocklist, and a missing required artifact
aborts the build rather than producing a quietly incomplete archive.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
OUTPUT = REPO / "group10_submission_bundle.zip"

E1_ID = "e1_fairseq_vi_zh_v1"
E2_ID = "e2_qwen3_8b_qlora_vi_zh_v1"
E3_ID = "e3_custom_fairseq_knowledge_vi_zh_v1"

# Anything matching these must never reach the archive: the private course
# specification, the bundled paper, the raw corpora, and the Kaggle input zips.
DENY_SUBSTRINGS = (
    "prism-uploads",
    "PROJECT_SPECIFICATION",
    "2021.icon-main",
    "kaggle_input_bundle",
    "prediction_checkpoint.zip",
)
DENY_PREFIXES = ("zh-vi/", "en-vi/", "artifacts/")

# Stage artifacts that are expected to be absent are reported, not fatal.
OPTIONAL_WORK_FILES = ("train.log", "training_history.json", "knowledge_augmentation.json")


class BundleError(RuntimeError):
    pass


def fairseq_sources(experiment_id: str, config: str) -> list[Path]:
    work = REPO / "work" / experiment_id
    sources = [
        REPO / "configs" / config,
        work / "run_manifest.json",
        work / "selection_frozen.json",
        REPO / "predictions" / f"{experiment_id}.val.jsonl",
        REPO / "predictions" / f"{experiment_id}.test.jsonl",
        REPO / "metrics" / f"{experiment_id}.val.json",
        REPO / "metrics" / f"{experiment_id}.test.json",
        REPO / "checkpoint" / experiment_id / "checkpoint_best.pt",
        work / "data-bin",
    ]
    sources.extend(work / name for name in OPTIONAL_WORK_FILES if (work / name).exists())
    sources.extend(sorted(work.glob("train.*.log")))
    return sources


def qlora_sources(experiment_id: str, config: str) -> list[Path]:
    work = REPO / "work" / experiment_id
    trainer = work / "trainer"
    state_path = trainer / "trainer_state.json"
    if not state_path.is_file():
        raise BundleError(f"missing {state_path.relative_to(REPO)}; cannot resolve the E2 checkpoints")
    state = json.loads(state_path.read_text(encoding="utf-8"))

    numbered = [(int(p.name.rsplit("-", 1)[1]), p) for p in trainer.glob("checkpoint-*") if p.is_dir()]
    if not numbered:
        raise BundleError(f"no trainer/checkpoint-* directories under {trainer.relative_to(REPO)}")
    latest = max(numbered)[1]

    sources = [
        REPO / "configs" / config,
        work / "run_manifest.json",
        work / "selection_frozen.json",
        REPO / "predictions" / f"{experiment_id}.val.jsonl",
        REPO / "predictions" / f"{experiment_id}.test.jsonl",
        REPO / "metrics" / f"{experiment_id}.val.json",
        REPO / "metrics" / f"{experiment_id}.test.json",
        REPO / "checkpoint" / experiment_id / "adapter",
        state_path,
        latest,
    ]
    # The selected checkpoint is what the adapter was taken from; ship it when it is
    # not simply the last one written.
    best_name = state.get("best_model_checkpoint")
    if best_name:
        best = trainer / Path(best_name).name
        if best.exists() and best != latest:
            sources.append(best)
    sources.extend(work / name for name in OPTIONAL_WORK_FILES if (work / name).exists())
    sources.extend(sorted(work.glob("train.*.log")))
    curve = work / "training_curves.svg"
    if curve.exists():
        sources.append(curve)
    return sources


def shared_sources() -> list[Path]:
    sources = [
        REPO / "metrics" / "data_audit.json",
        REPO / "metrics" / "e1_vs_e2.test.json",
        REPO / "metrics" / "e1_vs_e3_custom.test.json",
        REPO / "metrics" / "e3_custom_vs_e2.test.json",
        REPO / "docs" / "DATASET.md",          # specification section 13: dataset description
        REPO / "error_analysis" / "manual_sample.jsonl",
        REPO / "error_analysis" / "GUIDELINES.md",
        REPO / "Error_Analysis.py",
        REPO / "Error_Analysis_PreLabeled.xlsx",
        REPO / "Error_Analysis_Labeled.xlsx",
    ]
    for optional in (
        REPO / "metrics" / "moses",
        REPO / "error_analysis" / "summary.json",          # only once annotation is adjudicated
        REPO / "error_analysis" / "automatic_summary.json",
        REPO / "error_analysis" / "automatic_labels.csv",
        REPO / "analysis" / "figures",                     # produced by a Kaggle notebook run
        REPO / "analysis" / "tables",
    ):
        if optional.exists():
            sources.append(optional)
    return sources


def expand(sources: list[Path]) -> list[Path]:
    """Flatten directories to files, keeping the listing order stable."""
    files: list[Path] = []
    missing: list[Path] = []
    for source in sources:
        if not source.exists():
            missing.append(source)
        elif source.is_dir():
            files.extend(sorted(p for p in source.rglob("*") if p.is_file()))
        else:
            files.append(source)
    if missing:
        raise BundleError(
            "required artifacts are missing:\n"
            + "\n".join(f"  {p.relative_to(REPO)}" for p in missing)
        )
    return files


def guard(files: list[Path]) -> None:
    blocked = []
    for path in files:
        rel = path.relative_to(REPO).as_posix()
        if any(token in rel for token in DENY_SUBSTRINGS) or rel.startswith(DENY_PREFIXES):
            blocked.append(rel)
    if blocked:
        raise BundleError(
            "restricted material reached the manifest; refusing to build:\n"
            + "\n".join(f"  {rel}" for rel in blocked)
        )


PRIVATE_README = """Group 10 - Vietnamese to Classical Chinese MT
Private artifact bundle for course submission (specification section 13).

Contents: configs, run manifests, frozen selections, validation/test predictions and
metrics, paired bootstrap comparisons, model checkpoints and adapters, the shared
seeded 100-sentence error-analysis sheet, and the heuristic error-analysis workbooks.

The source tree is delivered separately through the private repository.

DO NOT PUBLISH. This archive contains model outputs, predictions, and data derived
from the restricted course corpus. The raw corpus itself is not included; see
docs/DATASET.md and metrics/data_audit.json for its description and SHA-256 hashes.

Automatic labels in Error_Analysis_Labeled.xlsx are rule-based heuristic proxies
derived from a single reference. They are not human judgments.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="print the manifest without writing the archive")
    parser.add_argument("--output", default=str(OUTPUT), help="archive path (default: %(default)s)")
    args = parser.parse_args()

    groups = {
        "E1 Fairseq": fairseq_sources(E1_ID, "e1_fairseq.yaml"),
        "E2 Qwen3-8B QLoRA": qlora_sources(E2_ID, "e2_qwen3_qlora.yaml"),
        "E3-custom Fairseq knowledge": fairseq_sources(E3_ID, "e3_custom_fairseq_knowledge.yaml"),
        "Shared": shared_sources(),
    }

    total = 0
    manifest: list[Path] = []
    for label, sources in groups.items():
        files = expand(sources)
        size = sum(p.stat().st_size for p in files)
        total += size
        manifest.extend(files)
        print(f"  {label:<30} {len(files):>5} files  {size / 1024**3:>6.2f} GiB")

    guard(manifest)
    seen, unique = set(), []
    for path in manifest:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    print(f"  {'TOTAL':<30} {len(unique):>5} files  {total / 1024**3:>6.2f} GiB")
    print("  restricted-material guard: passed")

    if args.dry_run:
        print("\ndry run; no archive written")
        return 0

    output = Path(args.output)
    print(f"\nwriting {output.name} ...")
    with zipfile.ZipFile(output, "w", allowZip64=True) as archive:
        for path in unique:
            arcname = path.relative_to(REPO).as_posix()
            # Model weights are already dense; deflating them costs minutes and saves little.
            large = path.stat().st_size > 50 * 1024**2
            archive.write(
                path,
                arcname,
                compress_type=zipfile.ZIP_STORED if large else zipfile.ZIP_DEFLATED,
            )
        archive.writestr("PRIVATE_README.txt", PRIVATE_README)

    print(f"done: {output}  ({output.stat().st_size / 1024**3:.2f} GiB, {len(unique) + 1} entries)")
    print("Upload to a restricted-access Drive folder. Do not use anyone-with-link sharing.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BundleError as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
