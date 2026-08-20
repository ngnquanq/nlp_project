from __future__ import annotations

import json
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from .config import load_yaml, repo_path
from .io_utils import sha256_file, write_json
from .normalize import detokenize_chinese, normalize_whitespace, score_form_chinese


@dataclass(frozen=True)
class ParallelRow:
    sample_id: str
    split: str
    source: str
    reference_tokenized: str

    @property
    def reference(self) -> str:
        return detokenize_chinese(self.reference_tokenized)

    @property
    def reference_scored(self) -> str:
        return score_form_chinese(self.reference_tokenized)


def load_dataset_config(path: str | Path) -> dict[str, Any]:
    return load_yaml(path)


def split_paths(dataset: dict[str, Any], split: str) -> tuple[Path, Path]:
    try:
        config = dataset["splits"][split]
    except KeyError as error:
        raise ValueError(f"Unknown split: {split}") from error
    return repo_path(config["source"]), repo_path(config["target"])


def read_split(dataset: dict[str, Any], split: str) -> list[ParallelRow]:
    source_path, target_path = split_paths(dataset, split)
    source_lines = source_path.read_text(encoding="utf-8").splitlines()
    target_lines = target_path.read_text(encoding="utf-8").splitlines()
    if len(source_lines) != len(target_lines):
        raise ValueError(
            f"Parallel length mismatch for {split}: {len(source_lines)} != {len(target_lines)}"
        )
    expected = dataset["splits"][split].get("expected_lines")
    if expected is not None and len(source_lines) != expected:
        raise ValueError(f"Unexpected {split} size: {len(source_lines)} != {expected}")
    return [
        ParallelRow(
            sample_id=f"{split}-{index:06d}",
            split=split,
            source=normalize_whitespace(source),
            reference_tokenized=normalize_whitespace(target),
        )
        for index, (source, target) in enumerate(zip(source_lines, target_lines), 1)
    ]


def _quantile(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def _side_stats(lines: list[str]) -> dict[str, Any]:
    token_counts = [len(line.split()) for line in lines]
    tokens = [token for line in lines for token in line.split()]
    punctuation = Counter(
        character
        for line in lines
        for character in line
        if unicodedata.category(character).startswith("P")
    )
    return {
        "lines": len(lines),
        "blank_lines": sum(not line.strip() for line in lines),
        "edge_whitespace_lines": sum(line != line.strip() for line in lines),
        "tokens": len(tokens),
        "types": len(set(tokens)),
        "hapax_types": sum(count == 1 for count in Counter(tokens).values()),
        "tokens_per_line": {
            "mean": round(statistics.mean(token_counts), 4),
            "p95": _quantile(token_counts, 0.95),
            "max": max(token_counts),
        },
        "punctuation": dict(punctuation.most_common()),
    }


def audit_dataset(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    dataset = load_dataset_config(config_path)
    report: dict[str, Any] = {
        "dataset_id": dataset["dataset_id"],
        "description": dataset["description"],
        "splits": {},
        "overlap": {},
        "issues": [],
    }
    loaded: dict[str, list[ParallelRow]] = {}
    raw_target_characters: set[str] = set()
    for split in ("train", "val", "test"):
        source_path, target_path = split_paths(dataset, split)
        rows = read_split(dataset, split)
        loaded[split] = rows
        sources = source_path.read_text(encoding="utf-8").splitlines()
        targets = target_path.read_text(encoding="utf-8").splitlines()
        raw_target_characters.update(
            character for line in targets for character in line if not character.isspace()
        )
        pairs = list(zip(sources, targets))
        source_targets: dict[str, set[str]] = defaultdict(set)
        for source, target in pairs:
            source_targets[source].add(target)
        replacements = [
            {"sample_id": f"{split}-{index:06d}", "side": side, "text": text}
            for index, (source, target) in enumerate(zip(sources, targets), 1)
            for side, text in (("source", source), ("target", target))
            if "�" in text
        ]
        report["splits"][split] = {
            "quality": dataset["splits"][split]["quality"],
            "source_file": str(source_path.relative_to(repo_path("."))),
            "target_file": str(target_path.relative_to(repo_path("."))),
            "source_sha256": sha256_file(source_path),
            "target_sha256": sha256_file(target_path),
            "parallel_pairs": len(rows),
            "duplicate_pairs": len(pairs) - len(set(pairs)),
            "duplicate_sources": len(sources) - len(set(sources)),
            "duplicate_targets": len(targets) - len(set(targets)),
            "sources_with_multiple_targets": sum(
                len(values) > 1 for values in source_targets.values()
            ),
            "source": _side_stats(sources),
            "target": _side_stats(targets),
            "replacement_characters": replacements,
        }
        report["issues"].extend(replacements)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_rows, right_rows = loaded[left], loaded[right]
        report["overlap"][f"{left}_{right}"] = {
            "pairs": len(
                {(row.source, row.reference_tokenized) for row in left_rows}
                & {(row.source, row.reference_tokenized) for row in right_rows}
            ),
            "sources": len({row.source for row in left_rows} & {row.source for row in right_rows}),
            "targets": len(
                {row.reference_tokenized for row in left_rows}
                & {row.reference_tokenized for row in right_rows}
            ),
        }

    knowledge_path = repo_path(dataset["knowledge"])
    with knowledge_path.open(encoding="utf-8") as handle:
        knowledge = json.load(handle)
    if not isinstance(knowledge, dict):
        raise ValueError("knowledge.json must contain an object keyed by character")
    corpus_chars = raw_target_characters
    keys = set(knowledge)
    report["knowledge"] = {
        "file": dataset["knowledge"],
        "sha256": sha256_file(knowledge_path),
        "entries": len(knowledge),
        "top_level_fields": dict(
            Counter(field for value in knowledge.values() for field in value)
        ),
        "corpus_target_characters": len(corpus_chars),
        "covered_target_characters": len(corpus_chars & keys),
        "coverage": round(len(corpus_chars & keys) / len(corpus_chars), 6),
        "uncovered_characters": sorted(corpus_chars - keys),
    }
    write_json(repo_path(output_path), report)
    return report


def iter_split(dataset: dict[str, Any], split: str) -> Iterator[ParallelRow]:
    yield from read_split(dataset, split)


def dataset_fingerprint(config_path: str | Path) -> dict[str, Any]:
    dataset = load_dataset_config(config_path)
    result: dict[str, Any] = {"dataset_id": dataset["dataset_id"], "splits": {}}
    for split in ("train", "val", "test"):
        source, target = split_paths(dataset, split)
        result["splits"][split] = {
            "source_sha256": sha256_file(source),
            "target_sha256": sha256_file(target),
            "rows": len(read_split(dataset, split)),
        }
    knowledge = repo_path(dataset["knowledge"])
    result["knowledge_sha256"] = sha256_file(knowledge)
    return result
