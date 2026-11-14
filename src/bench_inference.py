"""Benchmark de inferencia de las 6 configuraciones en la GPU local.

Uso (cuando hayan terminado las colas de entrenamiento):
    python -m src.bench_inference [--results-root results] [--out results/tablas]

Mide, con los checkpoints de E1 semilla 0 (results/checkpoints/E1/<config>_f100_s0.pt):
- Latencia por imagen: lote 1, 50 pasadas de calentamiento y 200 cronometradas
  (mediana y percentil 90), con torch.cuda.synchronize() alrededor de cada una.
- Throughput: lote 32, 10 lotes de calentamiento y 50 cronometrados (img/s).
- En FP32 (columna principal) y en FP16 con autocast (columna extra; el
  entrenamiento usó precisión mixta).

La entrada son tensores aleatorios fijos de 3x224x224: se mide solo el modelo,
sin lectura de imágenes ni preprocesamiento. cuDNN se usa en modo benchmark
(elige el algoritmo más rápido), como en un despliegue.

Protecciones (se saltan con --force):
- No arranca si la cola local sigue viva (logs/cola_local.pid) o si falta alguna
  corrida de rosa/yaiza, para no alterar los tiempos de entrenamiento.
- En Windows exige que el portátil esté conectado al cargador y registra el
  plan de energía activo.

Salidas: OUT/inferencia_rtx3050.csv (datos crudos con GPU, versiones y energía)
y OUT/tabla_inferencia.{csv,tex}.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .models.factory import build_model, count_parameters
from .runs import MAIN_CONFIGS, result_path, runs_for
from .utils import CONFIG_DIR, REPO_ROOT, environment_info, load_config, run_name


# ---------------------------------------------------------------------------
# Protecciones y entorno
# ---------------------------------------------------------------------------
def queue_alive(pid_file: Path) -> bool:
    if not pid_file.exists():
        return False
    pid = pid_file.read_text(encoding="ascii", errors="ignore").strip()
    if not pid.isdigit():
        return False
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
        return pid in out
    try:
        import os
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def power_status() -> dict:
    """Estado de energía en Windows: conectado a la red y plan activo."""
    if sys.platform != "win32":
        return {"ac_power": None, "power_plan": None}
    import ctypes

    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    status = SystemPowerStatus()
    ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status))
    plan = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout.strip()
    return {"ac_power": status.ACLineStatus == 1, "power_plan": plan}


def check_ready(results_root: Path, force: bool) -> None:
    problems = []
    if queue_alive(REPO_ROOT / "logs" / "cola_local.pid"):
        problems.append("la cola local de entrenamiento sigue corriendo")
    missing = [r["name"] for r in runs_for("rosa,yaiza") if not result_path(results_root, r).exists()]
    if missing:
        problems.append(f"faltan {len(missing)} corridas locales (p. ej. {missing[0]})")
    power = power_status()
    if power["ac_power"] is False:
        problems.append("el portátil está con batería; conéctalo al cargador")
    if problems and not force:
        raise SystemExit("No se lanza el benchmark: " + "; ".join(problems) + ". (--force para ignorar)")


# ---------------------------------------------------------------------------
# Medición
# ---------------------------------------------------------------------------
def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def latency_ms(model, device, fp16: bool, iters: int, warmup: int) -> np.ndarray:
    x = torch.randn(1, 3, 224, 224, device=device, generator=torch.Generator(device).manual_seed(0))
    times = []
    for i in range(warmup + iters):
        _sync(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            model(x)
        _sync(device)
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return np.array(times)


@torch.inference_mode()
def throughput(model, device, fp16: bool, batch_size: int, batches: int, warmup: int) -> float:
    x = torch.randn(batch_size, 3, 224, 224, device=device, generator=torch.Generator(device).manual_seed(0))
    for _ in range(warmup):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            model(x)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(batches):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16):
            model(x)
    _sync(device)
    return batch_size * batches / (time.perf_counter() - t0)


def bench_config(config: str, ckpt: Path, device, iters: int, warmup: int, batch_size: int,
                 batches: int) -> dict:
    cfg = load_config(CONFIG_DIR / f"{config}.yaml", ["model.pretrained=false"])
    model = build_model(cfg)
    model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.to(device).eval()
    row = {"config": config, "checkpoint": str(ckpt.name), "params": count_parameters(model)["total"]}
    for name, fp16 in [("fp32", False), ("fp16", True)]:
        if fp16 and device.type != "cuda":
            continue
        lat = latency_ms(model, device, fp16, iters, warmup)
        row[f"latency_{name}_median_ms"] = float(np.median(lat))
        row[f"latency_{name}_p90_ms"] = float(np.percentile(lat, 90))
        row[f"throughput_{name}_img_s"] = throughput(model, device, fp16, batch_size, batches, max(2, batches // 5))
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "tablas")
    parser.add_argument("--seed", type=int, default=0, help="Semilla de los checkpoints de E1")
    parser.add_argument("--iters", type=int, default=200, help="Pasadas cronometradas de latencia (>= 100)")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--configs", nargs="*", default=MAIN_CONFIGS)
    parser.add_argument("--force", action="store_true", help="Ignorar las protecciones")
    args = parser.parse_args(argv)

    check_ready(args.results_root, args.force)
    device = torch.device(args.device)
    torch.backends.cudnn.benchmark = True
    env, power = environment_info(), power_status()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = []
    for config in args.configs:
        ckpt = args.results_root / "checkpoints" / "E1" / f"{run_name(config, 1.0, args.seed)}.pt"
        if not ckpt.exists():
            raise SystemExit(f"Falta el checkpoint {ckpt}")
        row = bench_config(config, ckpt, device, args.iters, args.warmup, args.batch_size, args.batches)
        row.update({"gpu": env.get("gpu", "cpu"), "torch": env["torch"], "cuda": env.get("cuda"),
                    "cudnn": env.get("cudnn"), "latency_iters": args.iters, "warmup": args.warmup,
                    "batch_size": args.batch_size, "throughput_batches": args.batches,
                    "ac_power": power["ac_power"], "power_plan": power["power_plan"], "measured_at": started})
        rows.append(row)
        print(f"{config:<18} FP32 {row['latency_fp32_median_ms']:6.2f} ms  {row['throughput_fp32_img_s']:7.0f} img/s"
              + (f" | FP16 {row['latency_fp16_median_ms']:6.2f} ms  {row['throughput_fp16_img_s']:7.0f} img/s"
                 if "latency_fp16_median_ms" in row else ""), flush=True)

    from .report import save_table, table_inference

    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "inferencia_rtx3050.csv", index=False)
    if "latency_fp16_median_ms" in df:
        save_table(table_inference(df), args.out, "tabla_inferencia",
                   f"Inferencia en {env.get('gpu', 'CPU')} (checkpoints de la semilla {args.seed}): latencia "
                   "con lote 1 (mediana de 200 pasadas) y throughput con lote 32.", "tab:inferencia")
    print(f"Guardado en {args.out}")


if __name__ == "__main__":
    main()
