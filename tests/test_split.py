"""La partición real (data/splits/split.csv) no tiene fuga entre conjuntos."""
import pandas as pd
import pytest

from src.split import check_split, make_split
from src.utils import CLASSES, SPLIT_CSV


@pytest.fixture(scope="module")
def split_df():
    if not SPLIT_CSV.exists():
        pytest.skip("Falta data/splits/split.csv")
    return pd.read_csv(SPLIT_CSV)


def test_no_plant_in_two_sets(split_df):
    assert (split_df.groupby("plant").split.nunique() == 1).all()
    sets = {s: set(split_df[split_df.split == s].plant) for s in ["train", "val", "test"]}
    assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])


def test_sizes_and_classes(split_df):
    assert len(split_df) == 1560 and split_df.plant.nunique() == 390
    assert split_df.groupby("split").plant.nunique().to_dict() == {"train": 273, "val": 58, "test": 59}
    for s in ["train", "val", "test"]:
        assert set(split_df[split_df.split == s].label) == set(CLASSES)


def test_subsets_nested_and_inside_train(split_df):
    check_split(split_df)


def test_make_split_is_deterministic(split_df):
    meta = split_df[["image", "plant", "lot", "view", "label", "width", "height"]]
    first = make_split(meta, n_trials=300)
    second = make_split(meta, n_trials=300)
    assert first.split.tolist() == second.split.tolist()
    check_split(first)
