"""Benchmark de inferencia (en CPU, con un checkpoint sin entrenar) y utilidades de tiempos."""
import pandas as pd
import pytest
import torch

import src.bench_inference as bench
from src.aggregate import stable_epoch_seconds
from src.models.factory import build_model
from src.utils import CONFIG_DIR, cuda_version_of, load_config


def _checkpoint(root, config="cnn_scratch"):
    model = build_model(load_config(CONFIG_DIR / f"{config}.yaml", ["model.pretrained=false"]))
    path = root / "checkpoints" / "E1" / f"{config}_f100_s0.pt"
    path.parent.mkdir(parents=True)
    torch.save(model.state_dict(), path)
    return path


def test_bench_config_cpu(tmp_path):
    row = bench.bench_config("cnn_scratch", _checkpoint(tmp_path), torch.device("cpu"),
                             iters=3, warmup=1, batch_size=2, batches=2)
    assert row["latency_fp32_median_ms"] > 0 and row["throughput_fp32_img_s"] > 0
    assert "latency_fp16_median_ms" not in row  # FP16 solo en GPU


def test_main_writes_csv(tmp_path):
    _checkpoint(tmp_path)
    out = tmp_path / "tablas"
    bench.main(["--results-root", str(tmp_path), "--out", str(out), "--device", "cpu", "--configs", "cnn_scratch",
                "--iters", "3", "--warmup", "1", "--batches", "2", "--batch-size", "2", "--force"])
    df = pd.read_csv(out / "inferencia_rtx3050.csv")
    assert list(df.config) == ["cnn_scratch"]
    assert {"gpu", "torch", "ac_power", "power_plan", "latency_iters"} <= set(df.columns)


def test_guard_refuses_when_runs_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "queue_alive", lambda pid_file: False)
    monkeypatch.setattr(bench, "power_status", lambda: {"ac_power": True, "power_plan": "x"})
    with pytest.raises(SystemExit, match="faltan 69 de las 69"):
        bench.check_ready(tmp_path, force=False)


def test_guard_passes_with_all_69_runs(tmp_path, monkeypatch):
    from src.runs import all_runs, result_path
    for r in all_runs():
        path = result_path(tmp_path, r)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(bench, "queue_alive", lambda pid_file: False)
    monkeypatch.setattr(bench, "power_status", lambda: {"ac_power": True, "power_plan": "x"})
    bench.check_ready(tmp_path, force=False)  # no lanza


def test_guard_refuses_on_battery_and_live_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "queue_alive", lambda pid_file: True)
    monkeypatch.setattr(bench, "power_status", lambda: {"ac_power": False, "power_plan": "x"})
    with pytest.raises(SystemExit, match="alimentación de red") as err:
        bench.check_ready(tmp_path, force=False)
    assert "sigue corriendo" in str(err.value)


def test_cuda_version_and_stable_epoch():
    assert cuda_version_of({"torch": "2.15.0+cu132"}) == "13.2"
    assert cuda_version_of({"torch": "2.5.1+cu121"}) == "12.1"
    assert cuda_version_of({"torch": "2.15.0+cpu"}) is None
    assert cuda_version_of({"torch": "2.15.0+cu132", "cuda": "13.2"}) == "13.2"
    hist = [{"epoch": 1, "epoch_seconds": 22.0}] + [{"epoch": e, "epoch_seconds": 9.5} for e in range(2, 8)]
    assert stable_epoch_seconds(hist) == 9.5
