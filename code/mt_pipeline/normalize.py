from __future__ import annotations

import re
import unicodedata


_SPACE_RE = re.compile(r"\s+")
_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)


def normalize_whitespace(text: str) -> str:
    return _SPACE_RE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def detokenize_chinese(text: str) -> str:
    """Join the supplied character-tokenized Chinese into readable text."""
    return "".join(normalize_whitespace(text).split())


def score_form_chinese(text: str) -> str:
    """Return one non-whitespace code point per token for shared scoring."""
    compact = detokenize_chinese(text)
    return " ".join(compact)


def clean_llm_generation(text: str) -> str:
    """Remove only known chat/reasoning wrappers, never rewrite content."""
    value = unicodedata.normalize("NFC", text).strip()
    value = _THINK_RE.sub("", value).strip()
    for prefix in ("assistant:", "Assistant:", "答案：", "译文：", "翻译："):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("“") and value.endswith("”")
    ):
        value = value[1:-1].strip()
    return detokenize_chinese(value)


def contains_thinking(text: str) -> bool:
    lowered = text.lower()
    return "<think>" in lowered or "</think>" in lowered

