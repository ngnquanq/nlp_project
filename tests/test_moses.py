import ast
import json
from pathlib import Path

import pytest

from mt_pipeline.cli import build_parser
from mt_pipeline.evaluation import (
    DEFAULT_PROTOCOL, LEGACY_PROTOCOL, _bleu_inputs, compare_predictions,
    evaluate_predictions, metrics_equivalent, protocol_from_metrics,
)
from mt_pipeline.io_utils import sha256_file, write_jsonl
from mt_pipeline.normalize import score_form_chinese
from mt_pipeline.rescore import rescore_moses


@pytest.fixture
def predictions(tmp_path):
    folder = tmp_path / "predictions"
    paths = []
    for name, text in (("baseline", "漢\x01文天地玄黃"), ("candidate", "漢文天地玄黃")):
        path = folder / f"{name}.test.jsonl"
        write_jsonl(path, [{
            "sample_id": f"test-{index:06d}", "split": "test", "source": "nguồn",
            "reference": "漢文天地玄黃", "prediction": text,
            "prediction_scored": score_form_chinese(text), "prediction_raw": text,
            "experiment_id": name, "model_name": "fixture", "checkpoint": "fixture",
            "decoding_config": {},
        } for index in range(1, 511)])
        paths.append(path)
    return paths


def test_real_moses_rules_and_rare_characters():
    # Fullwidth punctuation distinguishes Moses from 13a on unsegmented input.
    assert _bleu_inputs(["漢文，測試"], DEFAULT_PROTOCOL) == ["漢文 ， 測試"]
    assert _bleu_inputs(["漢文，測試"], LEGACY_PROTOCOL) == ["漢文，測試"]
    text = score_form_chinese("𠀀爲漢文&<>[]")
    assert _bleu_inputs([text], DEFAULT_PROTOCOL) == [text]


def test_moses_changes_bleu_but_preserves_chrf(predictions, tmp_path):
    legacy = evaluate_predictions(predictions[0], tmp_path / "old.json", LEGACY_PROTOCOL)
    moses = evaluate_predictions(predictions[0], tmp_path / "new.json")
    assert legacy["metrics"]["sacrebleu"]["score"] < 100
    assert moses["metrics"]["sacrebleu"]["score"] == pytest.approx(100)
    assert legacy["metrics"]["chrf_pp"] == moses["metrics"]["chrf_pp"]
    assert protocol_from_metrics(legacy["metrics"]) == LEGACY_PROTOCOL
    assert protocol_from_metrics(moses["metrics"]) == DEFAULT_PROTOCOL
    assert not metrics_equivalent(legacy["metrics"], moses["metrics"])
    moses["metrics"]["sacrebleu"]["preprocessing"]["escape"] = True
    fresh = evaluate_predictions(predictions[0], tmp_path / "fresh.json")
    assert not metrics_equivalent(moses["metrics"], fresh["metrics"])


def test_comparison_uses_same_protocol_as_evaluation(predictions, tmp_path):
    for protocol in (DEFAULT_PROTOCOL, LEGACY_PROTOCOL):
        compared = compare_predictions(*predictions, tmp_path / "compare.json",
                                       samples=5, seed=42, protocol=protocol)
        repeated = compare_predictions(*predictions, tmp_path / "repeat.json",
                                       samples=5, seed=42, protocol=protocol)
        assert compared == repeated
        for path, field in zip(predictions, ("baseline_score", "candidate_score")):
            evaluated = evaluate_predictions(path, tmp_path / "metric.json", protocol)
            for name in ("sacrebleu", "chrf_pp"):
                assert compared["metrics"][name][field] == evaluated["metrics"][name]["score"]
        assert protocol_from_metrics(compared["metrics"]) == protocol


def test_migration_preserves_originals_and_verifies_provenance(predictions, tmp_path):
    original = tmp_path / "metrics"
    output = original / "moses"
    for path in predictions:
        evaluate_predictions(path, original / path.with_suffix(".json").name, LEGACY_PROTOCOL)
    compare_predictions(*predictions, original / "pair.test.json", samples=5,
                        seed=42, protocol=LEGACY_PROTOCOL)
    originals = predictions + list(original.glob("*.json"))
    hashes = {path: sha256_file(path) for path in originals}
    report = rescore_moses(predictions[0].parent, original, output)
    assert hashes == {path: sha256_file(path) for path in originals}
    assert len(report["scores"]) == 2
    assert len(report["comparisons"]) == 1
    assert all(row["chrf_pp_delta"] == 0 for row in report["scores"])
    assert report["scores"][0]["bleu_delta"] > 0
    bad_path = original / "baseline.test.json"
    bad = json.loads(bad_path.read_text())
    bad["prediction_sha256"] = "tampered"
    bad_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="do not reproduce"):
        rescore_moses(predictions[0].parent, original, output)


def test_invalid_protocol_and_sample_count(predictions, tmp_path):
    with pytest.raises(ValueError, match="Unknown evaluation"):
        evaluate_predictions(predictions[0], tmp_path / "bad.json", "moses")
    with pytest.raises(ValueError, match="positive"):
        compare_predictions(*predictions, tmp_path / "bad.json", samples=0)
    with pytest.raises(ValueError, match="separate"):
        rescore_moses(predictions[0].parent, tmp_path, tmp_path)
    with pytest.raises(ValueError, match="Unrecognized"):
        protocol_from_metrics({"sacrebleu": {"signature": "tok:none"}})
    args = build_parser().parse_args(["evaluate", "--predictions", "p", "--output", "o"])
    assert args.protocol == DEFAULT_PROTOCOL


def test_notebook_restores_legacy_metrics(predictions, tmp_path, monkeypatch):
    import mt_pipeline.freeze

    monkeypatch.setattr(mt_pipeline.freeze, "ensure_selection_frozen", lambda _: None)
    notebook = json.loads(Path("e2qwen3-qlora.ipynb").read_text())
    definitions = []
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        tree = ast.parse(source, filename=f"notebook-cell-{index}")
        definitions.extend(node for node in tree.body if isinstance(node, ast.FunctionDef)
                           and node.name == "validate_complete_stage")
    assert len(definitions) == 1
    stored_path = tmp_path / "legacy.json"
    evaluate_predictions(predictions[0], stored_path, LEGACY_PROTOCOL)
    protocols = []

    def run_command(*args):
        protocol = args[args.index("--protocol") + 1]
        protocols.append(protocol)
        evaluate_predictions(args[args.index("--predictions") + 1],
                             args[args.index("--output") + 1], protocol)

    namespace = {
        "prediction_path": lambda *_: predictions[0],
        "metric_path": lambda *_: stored_path,
        "read_json": lambda path: json.loads(path.read_text()),
        "run_command": run_command, "KAGGLE_TEMP": tmp_path,
        "sys": __import__("sys"),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), "notebook", "exec"), namespace)
    namespace["validate_complete_stage"]("fixture", "baseline")
    assert protocols == [LEGACY_PROTOCOL, LEGACY_PROTOCOL]
