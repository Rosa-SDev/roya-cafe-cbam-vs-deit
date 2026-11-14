"""Lista global de corridas, reparto entre integrantes y cola reanudable.

Uso:
    python -m src.runs --status                       # avance global
    python -m src.runs --member rosa                  # lista las corridas de Rosa
    python -m src.runs --member rosa --execute        # las ejecuta (omite las hechas)
    python -m src.runs --member daniel --execute --experiment E1   # solo E1
    python -m src.runs --benchmark                    # mide s/época en esta GPU y estima totales

Cada corrida se ejecuta en un subproceso (python -m src.train ...). Si una
falla, se registra y la cola sigue. Como train.py omite las corridas cuyo JSON
ya existe, basta volver a lanzar el mismo comando tras una desconexión.

Reparto (estimación previa en T4; ver docs/decisiones.md). En E3 se igualan
los pasos de optimización, así que cada corrida de E3 cuesta lo mismo que su
equivalente de E1:
    rosa:   E1 semillas 0-1 + E3 semilla 0   (~2,5 h)
    yaiza:  E1 semillas 2-3 + E3 semilla 1   (~2,5 h)
    daniel: E1 semilla 4   + E2 completo    (~2,6 h)
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


ASSIGNMENT = {
    "rosa": lambda r: (r["experiment"] == "E1" and r["seed"] in (0, 1)) or (r["experiment"] == "E3" and r["seed"] == 0),
    "yaiza": lambda r: (r["experiment"] == "E1" and r["seed"] in (2, 3)) or (r["experiment"] == "E3" and r["seed"] == 1),
    "daniel": lambda r: (r["experiment"] == "E1" and r["seed"] == 4) or r["experiment"] == "E2",
}


def runs_for(member: str) -> list[dict]:
    if member not in ASSIGNMENT:
        raise SystemExit(f"Integrante desconocido: {member}. Opciones: {sorted(ASSIGNMENT)}")
    return [r for r in all_runs() if ASSIGNMENT[member](r)]


def result_path(results_root: Path, run: dict) -> Path:
    return results_root / "runs" / run["experiment"] / f"{run['name']}.json"


def estimated_minutes(runs: list[dict]) -> float:
    return sum(EST_MINUTES[r["config"]] for r in runs)


def check_assignment() -> None:
    """Cada corrida asignada exactamente a un integrante."""
    for r in all_runs():
        owners = [m for m, f in ASSIGNMENT.items() if f(r)]
        assert len(owners) == 1, f"{r['name']} ({r['experiment']}) asignada a {owners}"


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


def benchmark(data_root: Path, out_root: Path, epochs: int = 3) -> dict[str, float]:
    """Entrena cada configuración principal unas pocas épocas y estima minutos por corrida.

    La primera época incluye arranque (cuDNN, workers), por eso se usa la
    mediana de las épocas. Estimación = s/época x épocas completas (60 o 30),
    sin early stopping: es una cota superior. E2 se estima con cnn_cbam_scratch
    y E3 cuesta lo mismo que E1 porque se igualan los pasos.
    """
    import json
    import statistics

    from .utils import load_config

    minutes = {}
    for config in MAIN_CONFIGS:
        cmd = [sys.executable, "-m", "src.train", "--config", str(CONFIG_DIR / f"{config}.yaml"), "--seed", "0",
               "--experiment", "bench", "--data-root", str(data_root), "--results-root", str(out_root),
               "--override", f"train.epochs={epochs}", "--override", "train.warmup_epochs=1",
               "--override", "interpret.enabled=false", "--no-checkpoint", "--force"]
        subprocess.call(cmd, cwd=REPO_ROOT)
        path = out_root / "runs" / "bench" / f"{run_name(config, 1.0, 0)}.json"
        with open(path, encoding="utf-8") as fh:
            hist = json.load(fh)["history"]
        sec = statistics.median(h["epoch_seconds"] for h in hist[1:] or hist)
        full_epochs = load_config(CONFIG_DIR / f"{config}.yaml")["train"]["epochs"]
        minutes[config] = sec * full_epochs / 60
        print(f"{config}: {sec:.1f} s/época -> ~{minutes[config]:.1f} min por corrida")
    for c in E2_CONFIGS:
        minutes[c] = minutes["cnn_cbam_scratch"]
    total = 0.0
    for member in sorted(ASSIGNMENT):
        m = sum(minutes[r["config"]] for r in runs_for(member))
        total += m
        print(f"{member}: ~{m / 60:.1f} h")
    print(f"Total: ~{total / 60:.1f} h de GPU (cota superior, sin early stopping)")
    return minutes


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--member", help="rosa | yaiza | daniel")
    parser.add_argument("--experiment", help="Filtrar por experimento (E1, E2, E3)")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--status", action="store_true", help="Avance de todas las corridas")
    parser.add_argument("--benchmark", action="store_true", help="Medir tiempos en esta GPU")
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    args, extra = parser.parse_known_args(argv)
    check_assignment()

    runs = runs_for(args.member) if args.member else all_runs()
    if args.experiment:
        runs = [r for r in runs if r["experiment"] == args.experiment]

    if args.benchmark:
        benchmark(resolve_dir(args.data_root, "ROCOLE_DATA_ROOT", "datos"),
                  Path(args.results_root or "bench_results"))
        return

    results_root = Path(args.results_root) if args.results_root else None
    if args.execute:
        if not args.member:
            raise SystemExit("--execute requiere --member")
        execute(runs, resolve_dir(args.data_root, "ROCOLE_DATA_ROOT", "datos"),
                resolve_dir(args.results_root, "ROCOLE_RESULTS_ROOT", "resultados"), extra)
        return

    if args.status:
        results_root = resolve_dir(args.results_root, "ROCOLE_RESULTS_ROOT", "resultados")
    for member in ([args.member] if args.member else sorted(ASSIGNMENT)):
        mine = [r for r in runs if ASSIGNMENT[member](r)]
        done = sum(result_path(results_root, r).exists() for r in mine) if results_root else 0
        print(f"{member}: {len(mine)} corridas, ~{estimated_minutes(mine) / 60:.1f} h estimadas"
              + (f", {done} hechas" if results_root else ""))
        if args.member:
            for r in mine:
                mark = "x" if results_root and result_path(results_root, r).exists() else " "
                print(f"  [{mark}] {r['experiment']} {r['name']}")


if __name__ == "__main__":
    main()
