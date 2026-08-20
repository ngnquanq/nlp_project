import json

import pytest
import yaml

from mt_pipeline.fairseq_runner import prepare_fairseq
from mt_pipeline.knowledge import KnowledgeAugmenter


def _write_fixture(tmp_path):
    knowledge = {
        "甲": {
            "character_coverage": {
                "canonical_entries": [
                    {
                        "char": "甲",
                        "pronunciation": "nam",
                        "nom_query": "nam",
                        "lookup_query": "nam",
                        "query_variants": ["nam"],
                    }
                ]
            }
        },
        "乙": {
            "character_coverage": {
                "canonical_entries": [
                    {
                        "char": "乙",
                        "pronunciation": "nam",
                        "nom_query": "nam",
                        "lookup_query": "nam",
                        "query_variants": [],
                    }
                ]
            }
        },
        "丙": {
            "character_coverage": {
                "canonical_entries": [
                    {
                        "char": "丙",
                        "pronunciation": "nam",
                        "nom_query": "nam thứ",
                        "lookup_query": "nam thứ",
                        "query_variants": ["nam", "nam thứ"],
                    }
                ]
            }
        },
    }
    knowledge_path = tmp_path / "knowledge.json"
    knowledge_path.write_text(json.dumps(knowledge, ensure_ascii=False), encoding="utf-8")

    splits = {}
    values = {
        "train": (["nam thứ", "không khớp"], ["甲 甲 乙", "丙"]),
        "val": (["nam thứ"], ["乙"]),
        "test": (["nam"], ["甲"]),
    }
    for split, (sources, targets) in values.items():
        source_path = tmp_path / f"{split}.vi"
        target_path = tmp_path / f"{split}.zh"
        source_path.write_text("\n".join(sources) + "\n", encoding="utf-8")
        target_path.write_text("\n".join(targets) + "\n", encoding="utf-8")
        splits[split] = {
            "source": str(source_path),
            "target": str(target_path),
            "quality": "test",
            "expected_lines": len(sources),
        }
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(
        yaml.safe_dump(
            {
                "dataset_id": "knowledge-test",
                "description": "fixture",
                "knowledge": str(knowledge_path),
                "splits": splits,
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = {
        "experiment_id": "knowledge-test",
        "backend": "fairseq_knowledge",
        "dataset_config": str(dataset_path),
        "work_dir": str(tmp_path / "work"),
        "checkpoint_dir": str(tmp_path / "checkpoint"),
        "knowledge": {
            "file": str(knowledge_path),
            "max_candidates": 2,
            "separator_token": "__knowledge__",
        },
        "training": {"max_source_positions": 6},
        "preprocess": {
            "source_lang": "vi",
            "target_lang": "zh",
            "threshold_source": 1,
            "threshold_target": 1,
            "workers": 1,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    return config, config_path


def test_knowledge_ranking_is_deterministic_and_capped(tmp_path):
    config, _ = _write_fixture(tmp_path)
    augmenter = KnowledgeAugmenter(config)

    result = augmenter.augment("nam thứ")

    assert result.source == "nam thứ __knowledge__ 丙 甲"
    assert result.candidate_count == 2
    assert result.matched_spans == 2
    assert augmenter.augment("không khớp").source == "không khớp"


def test_knowledge_augmentation_enforces_source_limit(tmp_path):
    config, _ = _write_fixture(tmp_path)
    config["training"]["max_source_positions"] = 2
    augmenter = KnowledgeAugmenter(config)

    with pytest.raises(ValueError, match="max_source_positions"):
        augmenter.augment("nam thứ")


def test_prepare_fairseq_augments_only_source_text(tmp_path):
    _, config_path = _write_fixture(tmp_path)
    binary = tmp_path / "work" / "data-bin"
    binary.mkdir(parents=True)

    paths = prepare_fairseq(config_path)

    train_source = (paths["text"] / "train.vi").read_text(encoding="utf-8").splitlines()
    train_target = (paths["text"] / "train.zh").read_text(encoding="utf-8").splitlines()
    report = json.loads(
        (paths["work"] / "knowledge_augmentation.json").read_text(encoding="utf-8")
    )
    assert train_source[0] == "nam thứ __knowledge__ 丙 甲"
    assert train_target == ["甲 甲 乙", "丙"]
    assert report["splits"]["train"]["augmented_rows"] == 1


# --- Regression pins for the E3 refactor -----------------------------------------
# knowledge.py was split into a retrieval core plus per-backend renderers. The Fairseq
# rendering must stay byte-identical or E3's frozen selection and `make repro-check`
# break. These hashes come from the recorded run manifest at
# work/e3_custom_fairseq_knowledge_vi_zh_v1/run_manifest.json.

E3_AUGMENTED_SOURCE_SHA256 = {
    "train": "e5b3923b4f7443f88206fc2f329135e84f6633c57ee4f6c1de8dbac6052b99b3",  # pragma: allowlist secret
    "val": "a49b498dab782f7528e7b636423ff1851803fef2a8738b51b09a43fd3d4b8200",  # pragma: allowlist secret
    "test": "a41535d4ecd00ea6acdbab5f28cdf77719808b00dfae87879d2bfbacb8fe3011",  # pragma: allowlist secret
}


@pytest.mark.private_data
@pytest.mark.parametrize("split", sorted(E3_AUGMENTED_SOURCE_SHA256))
def test_fairseq_augmentation_matches_recorded_e3_hashes(split, tmp_path):
    import hashlib
    from pathlib import Path

    from mt_pipeline.config import load_yaml, repo_path
    from mt_pipeline.data import load_dataset_config, read_split

    if not repo_path("zh-vi/knowledge.json").exists():
        pytest.skip("restricted corpus not present")

    config = load_yaml("configs/e3_custom_fairseq_knowledge.yaml")
    augmenter = KnowledgeAugmenter(config)
    rows = read_split(load_dataset_config(config["dataset_config"]), split)
    text = "\n".join(augmenter.augment(row.source).source for row in rows) + "\n"

    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert digest == E3_AUGMENTED_SOURCE_SHA256[split]


def test_render_hint_shares_ranking_with_fairseq_augmentation(tmp_path):
    config, _ = _write_fixture(tmp_path)
    augmenter = KnowledgeAugmenter(config)

    fairseq = augmenter.augment("nam thứ")
    prompt = augmenter.render_hint("nam thứ")

    assert fairseq.source == "nam thứ __knowledge__ 丙 甲"
    assert prompt.source == "Gợi ý chữ Hán: 丙 甲"
    assert prompt.candidate_count == fairseq.candidate_count
    assert prompt.matched_spans == fairseq.matched_spans


def test_render_hint_is_empty_when_nothing_matches(tmp_path):
    config, _ = _write_fixture(tmp_path)
    augmenter = KnowledgeAugmenter(config)

    result = augmenter.render_hint("không khớp")

    assert result.source == ""
    assert result.candidate_count == 0


def test_render_hint_needs_no_fairseq_only_settings(tmp_path):
    # An E4 config carries neither separator_token nor training.max_source_positions.
    config, _ = _write_fixture(tmp_path)
    del config["knowledge"]["separator_token"]
    del config["training"]["max_source_positions"]

    augmenter = KnowledgeAugmenter(config)

    assert augmenter.render_hint("nam thứ").source == "Gợi ý chữ Hán: 丙 甲"
    with pytest.raises(ValueError, match="max_source_positions"):
        augmenter.augment("nam thứ")
