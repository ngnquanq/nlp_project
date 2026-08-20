import pytest

from mt_pipeline.data import load_dataset_config, read_split


pytestmark = pytest.mark.private_data


def test_frozen_split_sizes_and_alignment():
    dataset = load_dataset_config("configs/dataset.yaml")
    assert len(read_split(dataset, "train")) == 19218
    assert len(read_split(dataset, "val")) == 510
    assert len(read_split(dataset, "test")) == 510


def test_stable_sample_ids():
    dataset = load_dataset_config("configs/dataset.yaml")
    rows = read_split(dataset, "test")
    assert rows[0].sample_id == "test-000001"
    assert rows[-1].sample_id == "test-000510"

