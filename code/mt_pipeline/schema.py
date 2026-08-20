from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


REQUIRED_PREDICTION_FIELDS = {
    "sample_id",
    "split",
    "source",
    "reference",
    "prediction",
    "prediction_raw",
    "prediction_scored",
    "experiment_id",
    "model_name",
    "checkpoint",
    "decoding_config",
}


@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    split: str
    source: str
    reference: str
    prediction: str
    prediction_raw: str
    prediction_scored: str
    experiment_id: str
    model_name: str
    checkpoint: str
    decoding_config: dict[str, Any]
    # Set only by knowledge-augmented backends. Dropped from the serialised row when
    # unset so predictions from the other experiments stay byte-identical.
    knowledge_hint: str | None = None

    def validate(self) -> None:
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"Invalid split for {self.sample_id}: {self.split}")
        if not self.sample_id.startswith(f"{self.split}-"):
            raise ValueError(f"sample_id/split mismatch: {self.sample_id}")
        for name in (
            "sample_id",
            "source",
            "reference",
            "prediction",
            "prediction_scored",
            "experiment_id",
            "model_name",
            "checkpoint",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} is empty for {self.sample_id}")
        if not isinstance(self.decoding_config, dict):
            raise ValueError("decoding_config must be an object")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        value = asdict(self)
        if value["knowledge_hint"] is None:
            del value["knowledge_hint"]
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PredictionRecord":
        missing = REQUIRED_PREDICTION_FIELDS - value.keys()
        if missing:
            raise ValueError(f"Missing prediction fields: {sorted(missing)}")
        record = cls(**{name: value[name] for name in REQUIRED_PREDICTION_FIELDS})
        record.validate()
        return record


def validate_prediction_rows(rows: list[dict[str, Any]], expected_count: int | None = None) -> None:
    records = [PredictionRecord.from_dict(row) for row in rows]
    if expected_count is not None and len(records) != expected_count:
        raise ValueError(f"Expected {expected_count} predictions, found {len(records)}")
    ids = [record.sample_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction file contains duplicate sample_id values")
    expected_order = sorted(ids, key=lambda value: int(value.rsplit("-", 1)[1]))
    if ids != expected_order:
        raise ValueError("Predictions are not in source row order")

