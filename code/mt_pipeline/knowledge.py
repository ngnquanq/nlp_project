from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .config import repo_path
from .data import load_dataset_config, read_split
from .io_utils import sha256_file
from .normalize import normalize_whitespace


@dataclass(frozen=True)
class AugmentationResult:
    source: str
    candidate_count: int
    matched_spans: int


class KnowledgeAugmenter:
    """Retrieve target-character hints from Vietnamese knowledge lookup forms.

    Retrieval and ranking are backend-agnostic (`rank_candidates`); rendering is not.
    `augment` produces the Fairseq token-append form and `render_hint` produces the
    natural-language form used in the Qwen prompt.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        settings = config["knowledge"]
        self.path = repo_path(settings["file"])
        self.max_candidates = int(settings["max_candidates"])
        # Fairseq-only: a sentinel token appended into the tokenized source. Qwen has
        # no embedding for it, so the Qwen renderer uses `hint_prefix` instead.
        separator = settings.get("separator_token")
        self.separator = None if separator is None else str(separator)
        self.hint_prefix = str(settings.get("hint_prefix", "Gợi ý chữ Hán:"))
        # Fairseq budgets candidates against the encoder position limit; the Qwen
        # renderer budgets against max_candidates alone.
        positions = config.get("training", {}).get("max_source_positions")
        self.max_source_positions = None if positions is None else int(positions)
        if self.max_candidates < 1:
            raise ValueError("knowledge.max_candidates must be positive")
        if self.separator is not None and (
            not self.separator or any(character.isspace() for character in self.separator)
        ):
            raise ValueError("knowledge.separator_token must be one non-whitespace token")

        with self.path.open(encoding="utf-8") as handle:
            knowledge = json.load(handle)
        if not isinstance(knowledge, dict):
            raise ValueError("Knowledge file must be an object keyed by character")

        forms: dict[tuple[str, ...], set[str]] = defaultdict(set)
        fields = ("pronunciation", "nom_query", "lookup_query")
        for top_level_character, value in knowledge.items():
            entries = value.get("character_coverage", {}).get("canonical_entries", [])
            for entry in entries:
                character = str(entry.get("char") or top_level_character)
                raw_forms = [entry.get(field) for field in fields]
                raw_forms.extend(entry.get("query_variants") or [])
                for raw_form in raw_forms:
                    if not isinstance(raw_form, str):
                        continue
                    normalized = normalize_whitespace(raw_form).casefold()
                    if normalized:
                        forms[tuple(normalized.split())].add(character)
        if not forms:
            raise ValueError(f"No Vietnamese lookup forms found in {self.path}")
        self.forms = dict(forms)
        self.form_lengths = tuple(sorted({len(form) for form in forms}))

        dataset = load_dataset_config(config["dataset_config"])
        self.target_frequency = Counter(
            character
            for row in read_split(dataset, "train")
            for character in row.reference_tokenized.split()
        )

    def rank_candidates(self, source: str) -> tuple[str, list[str], int]:
        """Backend-agnostic retrieval: normalized source, full ranked hits, span count.

        Ranking is match count, then training-target frequency, then Unicode order.
        Applies no budget — each renderer imposes its own.
        """
        normalized_source = normalize_whitespace(source)
        tokens = normalized_source.casefold().split()

        scores: Counter[str] = Counter()
        matched_spans = 0
        for start in range(len(tokens)):
            for length in self.form_lengths:
                if start + length > len(tokens):
                    break
                characters = self.forms.get(tuple(tokens[start : start + length]))
                if not characters:
                    continue
                matched_spans += 1
                scores.update(characters)

        ranked = sorted(
            scores,
            key=lambda character: (
                -scores[character],
                -self.target_frequency[character],
                character,
            ),
        )
        return normalized_source, ranked, matched_spans

    def augment(self, source: str) -> AugmentationResult:
        """Fairseq rendering: append a sentinel plus hints to the tokenized source."""
        if self.max_source_positions is None:
            raise ValueError("Fairseq augmentation requires training.max_source_positions")
        if self.separator is None:
            raise ValueError("Fairseq augmentation requires knowledge.separator_token")
        normalized_source = normalize_whitespace(source)
        tokens = normalized_source.casefold().split()
        if len(tokens) >= self.max_source_positions:
            raise ValueError(
                f"Source has {len(tokens)} tokens but max_source_positions is "
                f"{self.max_source_positions}"
            )

        normalized_source, ranked, matched_spans = self.rank_candidates(source)
        if not ranked:
            return AugmentationResult(normalized_source, 0, 0)

        available = self.max_source_positions - len(tokens) - 1
        candidates = ranked[: min(self.max_candidates, available)]
        if not candidates:
            return AugmentationResult(normalized_source, 0, matched_spans)
        augmented = " ".join([normalized_source, self.separator, *candidates])
        return AugmentationResult(augmented, len(candidates), matched_spans)

    def render_hint(self, source: str) -> AugmentationResult:
        """Qwen rendering: a natural-language hint line, budgeted by max_candidates.

        Returns the hint line alone (empty when nothing matched) rather than a rewritten
        source, so the caller decides where it goes in the chat prompt.
        """
        _, ranked, matched_spans = self.rank_candidates(source)
        candidates = ranked[: self.max_candidates]
        if not candidates:
            return AugmentationResult("", 0, matched_spans)
        return AugmentationResult(
            f"{self.hint_prefix} {' '.join(candidates)}", len(candidates), matched_spans
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "file": str(self.path),
            "sha256": sha256_file(self.path),
            "lookup_forms": len(self.forms),
            "max_candidates": self.max_candidates,
            "separator_token": self.separator,
            "ranking": "match_count,target_training_frequency,unicode",
        }

    def prompt_metadata(self) -> dict[str, Any]:
        """Hint configuration recorded with every E4 run and prediction row."""
        return {
            "file": str(self.path),
            "sha256": sha256_file(self.path),
            "lookup_forms": len(self.forms),
            "max_candidates": self.max_candidates,
            "hint_prefix": self.hint_prefix,
            "ranking": "match_count,target_training_frequency,unicode",
        }
