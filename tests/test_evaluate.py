"""Las métricas rápidas del bootstrap coinciden con sklearn."""
import numpy as np
from sklearn.metrics import f1_score

from src.evaluate import _f1_class, fast_macro_f1, plant_bootstrap_indices, to_binary


def test_fast_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        t, p = rng.integers(0, 6, 100), rng.integers(0, 6, 100)
        ref = f1_score(t, p, labels=range(6), average="macro", zero_division=0)
        assert np.isclose(fast_macro_f1(t, p), ref)


def test_binary_collapse():
    assert to_binary([0, 1, 2, 5]).tolist() == [0, 1, 1, 1]
    t, p = np.array([0, 1, 1, 0]), np.array([0, 1, 0, 1])
    assert np.isclose(_f1_class(t, p, 1), f1_score(t, p))


def test_plant_bootstrap_keeps_plants_together():
    plants = np.repeat(["a", "b", "c"], 4)
    for idx in plant_bootstrap_indices(plants, 20):
        assert len(idx) == 12
        for pl in set(plants[idx]):
            assert (plants[idx] == pl).sum() % 4 == 0
