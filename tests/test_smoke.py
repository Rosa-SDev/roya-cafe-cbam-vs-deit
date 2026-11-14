"""Smoke test: las 6 configuraciones entrenan 1 época con datos sintéticos en CPU
y escriben el JSON de resultados; una segunda llamada omite la corrida."""
import json

import pytest

import src.train as train_mod
from src.utils import CONFIG_DIR, load_config

CONFIGS = ["cnn_scratch", "cnn_cbam_scratch", "deit_scratch", "mbv2_pre", "mbv2_cbam_pre", "deit_pre"]


@pytest.mark.parametrize("config", CONFIGS)
def test_train_one_epoch(config, synthetic_data, monkeypatch):
    monkeypatch.setattr(train_mod, "SPLIT_CSV", synthetic_data["split_csv"])
    cfg = load_config(CONFIG_DIR / f"{config}.yaml",
                      ["train.epochs=1", "train.warmup_epochs=0", "model.pretrained=false",
                       "train.batch_size=8", "eval.inference_timing_iters=2", "interpret.examples_per_class=1"])
    out = train_mod.train_run(cfg, seed=0, fraction=1.0, experiment="smoke", data_root=synthetic_data["data_root"],
                              results_root=synthetic_data["results_root"], save_checkpoint=False, device_name="cpu")
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["n_images"] == {"train": 32, "val": 8, "test": 8}
    assert len(data["predictions"]) == 8
    assert 0 <= data["test"]["macro_f1"] <= 1
    methods = data["interpretation"]["methods"]
    assert ("rollout" in methods) if cfg["model"]["arch"] == "deit" else ("gradcam" in methods)
    if "cbam" in config:
        assert "cbam_last" in methods
    for row in data["interpretation"]["per_image"]:
        for m in methods:
            assert 0 <= row[f"in_leaf_{m}"] <= 1
    mtime = out.stat().st_mtime
    train_mod.train_run(cfg, 0, 1.0, "smoke", synthetic_data["data_root"], synthetic_data["results_root"],
                        save_checkpoint=False, device_name="cpu")
    assert out.stat().st_mtime == mtime  # reanudable: no se repite


def test_fraction_scales_schedule():
    cfg = load_config(CONFIG_DIR / "cnn_scratch.yaml")
    s = train_mod.scaled_schedule(cfg, 0.25, steps_per_epoch=9)
    assert (s["epochs"], s["warmup_epochs"], s["patience"], s["eval_every"]) == (240, 20, 60, 4)
    full = train_mod.scaled_schedule(cfg, 1.0, steps_per_epoch=35)
    assert abs(s["total_steps"] - full["total_steps"]) / full["total_steps"] < 0.1


def test_assignment_covers_every_run_once():
    from src.runs import all_runs, check_assignment, runs_for
    check_assignment()
    total = sum(len(runs_for(m)) for m in ["rosa", "yaiza", "daniel"])
    assert total == len(all_runs()) == 30 + 15 + 24
