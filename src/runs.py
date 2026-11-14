"""Lista global de corridas, selección por experimento y semillas, y cola reanudable.

Uso:
    python -m src.runs --status                                  # avance por experimento
    python -m src.runs --experiment E1 --seeds 0-3               # lista esas corridas
    python -m src.runs --experiment E1 --seeds 0-3 --execute     # las ejecuta (omite las hechas)
    python -m src.runs --experiment E1,E3 --execute              # varios experimentos
    python -m src.runs --execute                                 # todas las pendientes (69)
    python -m src.runs --benchmark                               # mide s/época en esta GPU y estima totales

Cada corrida se ejecuta en un subproceso (python -m src.train ...). Si una
falla, se registra y la cola sigue. Como train.py omite las corridas cuyo JSON
ya existe, basta volver a lanzar el mismo comando tras una interrupción.

En E3 se igualan los pasos de optimización, así que cada corrida de E3 cuesta
lo mismo que su equivalente de E1.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

from .utils import CONFIG_DIR, REPO_ROOT, resolve_dir, run_name

MAIN_CONFIGS = ["cnn_scratch", "cnn_cbam_scratch", "deit_scratch", "mbv2_pre", "mbv2_cbam_pre", "deit_pre"]
E2_CONFIGS = ["cnn_cbam_channel_scratch", "cnn_cbam_spatial_scratch", "cnn_cbam_last2_scratch"]
E1_SEEDS = [0, 1, 2, 3, 4]
E3_SEEDS = [0, 1]
E3_FRACTIONS = [0.25, 0.5]

# Minutos estimados por corrida en T4 (100 % de los datos o pasos equivalentes).
# Se reemplazan por los medidos en el benchmark del notebook 01.
EST_MINUTES = {"cnn_scratch": 8, "cnn_cbam_scratch": 9, "deit_scratch": 10, "mbv2_pre": 4,
               "mbv2_cbam_pre": 4, "deit_pre": 4, **{c: 8 for c in E2_CONFIGS}}


def all_runs() -> list[dict]:
    runs = []
    # Orden: E1 primero (configuraciones intercaladas por semilla), luego E2 y E3.
    for seed in E1_SEEDS:
        runs += [dict(experiment="E1", config=c, seed=seed, fraction=1.0) for c in MAIN_CONFIGS]
    for seed in E1_SEEDS:
        runs += [dict(experiment="E2", config=c, seed=seed, fraction=1.0) for c in E2_CONFIGS]
    for seed in E3_SEEDS:
        for frac in E3_FRACTIONS:
            runs += [dict(experiment="E3", config=c, seed=seed, fraction=frac) for c in MAIN_CONFIGS]
    for r in runs:
        r["name"] = run_name(r["config"], r["fraction"], r["seed"])
    return runs


EXPERIMENTS = ["E1", "E2", "E3"]


def parse_seeds(text: str | None) -> list[int] | None:
    """"0-3" -> [0, 1, 2, 3]; "0,2,4" -> [0, 2, 4]; "4" -> [4]; None -> None (todas)."""
    if text is None:
        return None
    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        try:
            if "-" in part:
                lo, hi = (int(x) for x in part.split("-", 1))
                if lo > hi:
                    raise ValueError
                seeds += range(lo, hi + 1)
            else:
                seeds.append(int(part))
        except ValueError:
            raise SystemExit(f"Semillas no válidas: {text!r} (ejemplos: 0-3, 0,2,4, 4)") from None
    return sorted(set(seeds))


def select_runs(experiments: str | list[str] | None = None, seeds: list[int] | None = None) -> list[dict]:
    """Corridas filtradas por experimento ("E1", "E1,E3" o lista) y por semillas (None = todas)."""
    if isinstance(experiments, str):
        experiments = [e.strip().upper() for e in experiments.split(",") if e.strip()]
    unknown = sorted(set(experiments or []) - set(EXPERIMENTS))
    if unknown:
        raise SystemExit(f"Experimento desconocido: {unknown}. Opciones: {EXPERIMENTS}")
    runs = [r for r in all_runs() if (not experiments or r["experiment"] in experiments)
            and (seeds is None or r["seed"] in seeds)]
    if not runs:
        raise SystemExit(f"Ninguna corrida con experimento {experiments or 'todos'} y semillas {seeds}")
    return runs


def result_path(results_root: Path, run: dict) -> Path:
    return results_root / "runs" / run["experiment"] / f"{run['name']}.json"


def estimated_minutes(runs: list[dict]) -> float:
    return sum(EST_MINUTES[r["config"]] for r in runs)


def execute(runs: list[dict], data_root: Path, results_root: Path, extra: list[str]) -> None:
    pending = [r for r in runs if not result_path(results_root, r).exists()]
    print(f"{len(runs) - len(pending)} hechas, {len(pending)} pendientes (~{estimated_minutes(pending):.0f} min estimados)")
    failures = []
    for i, r in enumerate(pending, 1):
        if result_path(results_root, r).exists():  # otra sesión pudo terminarla
            continue
        cmd = [sys.executable, "-m", "src.train", "--config", str(CONFIG_DIR / f"{r['config']}.yaml"),
               "--seed", str(r["seed"]), "--fraction", str(r["fraction"]), "--experiment", r["experiment"],
               "--data-root", str(data_root), "--results-root", str(results_root), *extra]
        print(f"\n=== [{i}/{len(pending)}] {r['experiment']} {r['name']} ===", flush=True)
        t0 = time.time()
        code = subprocess.call(cmd, cwd=REPO_ROOT)
        print(f"=== terminó con código {code} en {(time.time() - t0) / 60:.1f} min ===", flush=True)
        if code != 0:
            failures.append(r["name"])
    if failures:
        print("\nCorridas con error (vuelve a lanzar para reintentar):", *failures, sep="\n  ")
    else:
        print("\nTodas las corridas asignadas están completas.")


def benchmark(data_root: Path, out_root: Path, epochs: int = 2) -> dict[str, float]:
    """Entrena cada configuración principal `epochs` épocas y estima minutos por corrida.

    La época 1 incluye el arranque (cuDNN, primera compilación de kernels), por
    eso la estimación usa la última época (régimen estable), más ~1 min fijo
    por corrida para prueba, interpretabilidad y escritura. Estimación =
    s/época x épocas completas (60 o 30), sin early stopping: cota superior.
    E2 se estima con cnn_cbam_scratch y E3 cuesta lo mismo que E1 porque se
    igualan los pasos de optimización.
    """
    import json

    from .utils import load_config

    minutes = {}
    print(f"{'configuración':<18} {'época 1 (s)':>11} {'época N (s)':>11} {'GPU pico asignada':>18} "
          f"{'GPU pico reservada':>19} {'min/corrida':>12}")
    for config in MAIN_CONFIGS:
        cmd = [sys.executable, "-m", "src.train", "--config", str(CONFIG_DIR / f"{config}.yaml"), "--seed", "0",
               "--experiment", "bench", "--data-root", str(data_root), "--results-root", str(out_root),
               "--override", f"train.epochs={epochs}", "--override", "train.warmup_epochs=1",
               "--override", "interpret.enabled=false", "--no-checkpoint", "--force"]
        code = subprocess.call(cmd, cwd=REPO_ROOT, stdout=subprocess.DEVNULL)
        path = out_root / "runs" / "bench" / f"{run_name(config, 1.0, 0)}.json"
        if code != 0 or not path.exists():
            print(f"{config:<18} FALLÓ (código {code}); revisa memoria de GPU")
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        hist, mem = data["history"], data["timing"].get("gpu_memory_mb", {})
        sec = hist[-1]["epoch_seconds"]
        full_epochs = load_config(CONFIG_DIR / f"{config}.yaml")["train"]["epochs"]
        minutes[config] = sec * full_epochs / 60 + 1.0
        print(f"{config:<18} {hist[0]['epoch_seconds']:>11.1f} {sec:>11.1f} "
              f"{mem.get('max_allocated', float('nan')):>15.0f} MB {mem.get('max_reserved', float('nan')):>16.0f} MB "
              f"{minutes[config]:>12.1f}")
    if len(minutes) < len(MAIN_CONFIGS):
        return minutes
    for c in E2_CONFIGS:
        minutes[c] = minutes["cnn_cbam_scratch"]
    total = 0.0
    for exp in EXPERIMENTS:
        m = sum(minutes[r["config"]] for r in select_runs(exp))
        total += m
        print(f"{exp}: ~{m / 60:.1f} h")
    print(f"Total: ~{total / 60:.1f} h de GPU (cota superior, sin early stopping)")
    return minutes


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", help="E1, E2, E3 o varios separados por coma (por defecto, todos)")
    parser.add_argument("--seeds", help="Semillas: rango 0-3, lista 0,2,4 o una sola (por defecto, todas)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status", action="store_true", help="Avance de las corridas seleccionadas")
    parser.add_argument("--benchmark", action="store_true", help="Medir tiempos en esta GPU")
    parser.add_argument("--bench-epochs", type=int, default=2)
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    args, extra = parser.parse_known_args(argv)

    if args.benchmark:
        benchmark(resolve_dir(args.data_root, "ROCOLE_DATA_ROOT", "datos"),
                  Path(args.results_root or "bench_results"), args.bench_epochs)
        return

    runs = select_runs(args.experiment, parse_seeds(args.seeds))
    if args.execute:
        execute(runs, resolve_dir(args.data_root, "ROCOLE_DATA_ROOT", "datos"),
                resolve_dir(args.results_root, "ROCOLE_RESULTS_ROOT", "resultados"), extra)
        return

    results_root = None
    if args.status or args.results_root:
        results_root = resolve_dir(args.results_root, "ROCOLE_RESULTS_ROOT", "resultados")
    for exp in EXPERIMENTS:
        mine = [r for r in runs if r["experiment"] == exp]
        if not mine:
            continue
        done = sum(result_path(results_root, r).exists() for r in mine) if results_root else 0
        print(f"{exp}: {len(mine)} corridas, ~{estimated_minutes(mine) / 60:.1f} h estimadas"
              + (f", {done} hechas" if results_root else ""))
    if args.experiment or args.seeds:
        for r in runs:
            mark = "x" if results_root and result_path(results_root, r).exists() else " "
            print(f"  [{mark}] {r['experiment']} {r['name']}")


if __name__ == "__main__":
    main()
