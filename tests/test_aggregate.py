"""aggregate.py produce los CSV esperados a partir de JSON de corridas sintéticas."""
import numpy as np
import pandas as pd

from src.aggregate import aggregate
from src.evaluate import classification_metrics
from src.utils import CLASSES, write_json_atomic


def fake_run(results_root, exp, config, seed, acc_bias, rng):
    images = [f"C1P{p}{v}.jpg" for p in range(1, 11) for v in ["H1", "H2", "E1", "E2"]]
    y_true = np.arange(len(images)) % len(CLASSES)
    y_pred = np.where(rng.random(len(images)) < acc_bias, y_true, rng.integers(0, 6, len(images)))
    run = f"{config}_f100_s{seed}"
    data = {
        "run": run, "experiment": exp, "config_name": config, "seed": seed, "fraction": 1.0,
        "arch": "cnn", "pretrained": False, "debug_limit": None,
        "params": {"total": 1000, "trainable": 1000, "cbam": 10},
        "best": {"epoch": 3, "val_macro_f1": 0.5}, "epochs_run": 5,
        "timing": {"train_seconds": 60.0, "inference": {"median_ms": 1.0, "device": "cpu"}},
        "test": classification_metrics(y_true, y_pred),
        "predictions": [{"image": im, "plant": im[:-6], "y_true": int(t), "y_pred": int(p)}
                        for im, t, p in zip(images, y_true, y_pred)],
        "interpretation": {"methods": ["gradcam"],
                           "summary": {"leaf_area_mean": 0.3, "in_leaf_gradcam_mean": 0.5},
                           "per_image": [{"image": im, "in_leaf_gradcam": 0.5, "leaf_area": 0.3} for im in images]},
        "code_version": {"commit": "abc"}, "env": {"device": "cpu"},
    }
    write_json_atomic(results_root / "runs" / exp / f"{run}.json", data)
    return data


def test_aggregate_outputs(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "res"
    truth = {}
    for config, bias in [("cnn_scratch", 0.5), ("cnn_cbam_scratch", 0.9)]:
        truth[config] = [fake_run(root, "E1", config, s, bias, rng)["test"]["macro_f1"] for s in range(3)]
    tables = aggregate(root, tmp_path / "out", reps=200)
    for name in ["runs", "summary", "bootstrap", "comparisons"]:
        assert (tmp_path / "out" / f"{name}.csv").exists()
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert set(summary.config) == {"cnn_scratch", "cnn_cbam_scratch"}
    row = summary.set_index("config").loc["cnn_cbam_scratch"]
    assert row.n_seeds == 3
    assert np.isclose(row.macro_f1_mean, np.mean(truth["cnn_cbam_scratch"]))
    assert np.isclose(row.macro_f1_std, np.std(truth["cnn_cbam_scratch"], ddof=1))
    comp = tables["comparisons"]
    h1 = comp[(comp.comparison == "cnn_cbam - cnn (desde cero)") & (comp.metric == "macro_f1")].iloc[0]
    assert h1.ci_low <= h1.point <= h1.ci_high and h1.point > 0
    assert {"macro_f1", "accuracy", "binary_f1", "in_leaf_gradcam"} <= set(tables["bootstrap"].metric)


def test_debug_runs_are_skipped(tmp_path):
    rng = np.random.default_rng(1)
    root = tmp_path / "res"
    fake_run(root, "E1", "cnn_scratch", 0, 0.5, rng)
    fake_run(root, "debug", "cnn_scratch", 1, 0.5, rng)
    tables = aggregate(root, tmp_path / "out", reps=50)
    assert len(tables["runs"]) == 1
