"""Agrega los JSON de todas las corridas en tablas CSV.

Uso:
    python -m src.aggregate --results-root RESULTADOS [--out results/] [--reps 2000]

Salidas (en --out, por defecto RESULTADOS/aggregate):
    runs.csv         una fila por corrida con las métricas principales
    summary.csv      media, desviación y n por (experimento, configuración, fracción)
    bootstrap.csv    IC95 por bootstrap por planta de la media entre semillas
    comparisons.csv  diferencias pareadas que responden a H1-H4 (con IC95)

Se omiten las corridas de depuración (--limit) y la carpeta 'debug'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import bootstrap_ci, mean_stat, metric_stat, plant_bootstrap_indices
from .utils import CLASSES, cuda_version_of, resolve_dir

METRICS = ["macro_f1", "accuracy", "binary_f1"]
SUMMARY_COLS = ["accuracy", "macro_f1", "binary_accuracy", "binary_f1",
                *[f"recall_{c}" for c in CLASSES], *[f"precision_{c}" for c in CLASSES],
                *[f"f1_{c}" for c in CLASSES],
                "params_total", "params_cbam", "train_minutes", "epochs_run", "best_epoch",
                "infer_ms", "leaf_area"]

# (etiqueta, hipótesis, experimento/config/fracción A, B). Se calcula A - B.
COMPARISONS = [
    ("cnn_cbam - cnn (desde cero)", "H1", ("E1", "cnn_cbam_scratch", 1.0), ("E1", "cnn_scratch", 1.0)),
    ("mbv2_cbam - mbv2 (preentrenado)", "H1", ("E1", "mbv2_cbam_pre", 1.0), ("E1", "mbv2_pre", 1.0)),
    ("deit - cnn (desde cero)", "H2", ("E1", "deit_scratch", 1.0), ("E1", "cnn_scratch", 1.0)),
    ("deit - cnn_cbam (desde cero)", "H2", ("E1", "deit_scratch", 1.0), ("E1", "cnn_cbam_scratch", 1.0)),
    ("deit - mbv2 (preentrenado)", "H2", ("E1", "deit_pre", 1.0), ("E1", "mbv2_pre", 1.0)),
    ("deit - mbv2_cbam (preentrenado)", "H2", ("E1", "deit_pre", 1.0), ("E1", "mbv2_cbam_pre", 1.0)),
    ("deit_pre - deit_scratch", "H2", ("E1", "deit_pre", 1.0), ("E1", "deit_scratch", 1.0)),
    ("cbam_channel - cnn", "E2", ("E2", "cnn_cbam_channel_scratch", 1.0), ("E1", "cnn_scratch", 1.0)),
    ("cbam_spatial - cnn", "E2", ("E2", "cnn_cbam_spatial_scratch", 1.0), ("E1", "cnn_scratch", 1.0)),
    ("cbam_last2 - cnn", "E2", ("E2", "cnn_cbam_last2_scratch", 1.0), ("E1", "cnn_scratch", 1.0)),
    ("cbam_channel - cbam_both", "E2", ("E2", "cnn_cbam_channel_scratch", 1.0), ("E1", "cnn_cbam_scratch", 1.0)),
    ("cbam_spatial - cbam_both", "E2", ("E2", "cnn_cbam_spatial_scratch", 1.0), ("E1", "cnn_cbam_scratch", 1.0)),
    ("cbam_last2 - cbam_both", "E2", ("E2", "cnn_cbam_last2_scratch", 1.0), ("E1", "cnn_cbam_scratch", 1.0)),
]
for _frac in (0.25, 0.5):
    _pct = int(_frac * 100)
    COMPARISONS += [
        (f"deit - cnn (desde cero, {_pct} %)", "H3", ("E3", "deit_scratch", _frac), ("E3", "cnn_scratch", _frac)),
        (f"deit - mbv2 (preentrenado, {_pct} %)", "H3", ("E3", "deit_pre", _frac), ("E3", "mbv2_pre", _frac)),
    ]
# H4: fracción de Grad-CAM dentro de la hoja (CNN) y rollout de DeiT contra la CNN base.
H4_COMPARISONS = [
    ("in_leaf gradcam: cnn_cbam - cnn", ("E1", "cnn_cbam_scratch", 1.0), ("E1", "cnn_scratch", 1.0), "gradcam", "gradcam"),
    ("in_leaf gradcam: mbv2_cbam - mbv2", ("E1", "mbv2_cbam_pre", 1.0), ("E1", "mbv2_pre", 1.0), "gradcam", "gradcam"),
    ("in_leaf rollout deit_scratch - gradcam cnn", ("E1", "deit_scratch", 1.0), ("E1", "cnn_scratch", 1.0), "rollout", "gradcam"),
    ("in_leaf rollout deit_pre - gradcam mbv2", ("E1", "deit_pre", 1.0), ("E1", "mbv2_pre", 1.0), "rollout", "gradcam"),
    ("in_leaf cbam_last cnn_cbam - gradcam cnn", ("E1", "cnn_cbam_scratch", 1.0), ("E1", "cnn_scratch", 1.0), "cbam_last", "gradcam"),
]


def load_runs(results_root: Path) -> list[dict]:
    runs = []
    for path in sorted((results_root / "runs").glob("*/*.json")):
        if path.parent.name == "debug":
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("debug_limit"):
            continue
        runs.append(data)
    return runs


def stable_epoch_seconds(history: list[dict]) -> float:
    """Mediana de los segundos por época sin la época 1.

    La época 1 incluye el arranque (workers del DataLoader, cuDNN). Cada valor
    cubre entrenamiento + validación de esa época. En E3 solo se registran las
    épocas validadas, que son representativas del resto.
    """
    secs = [h["epoch_seconds"] for h in history if h.get("epoch", 0) > 1]
    if not secs:
        secs = [h["epoch_seconds"] for h in history]
    return float(np.median(secs)) if secs else float("nan")


def run_row(r: dict) -> dict:
    t = r["test"]
    row = {
        "experiment": r["experiment"], "config": r["config_name"], "arch": r["arch"],
        "pretrained": r["pretrained"], "seed": r["seed"], "fraction": r["fraction"],
        "accuracy": t["accuracy"], "macro_f1": t["macro_f1"],
        "binary_accuracy": t["binary"]["accuracy"], "binary_f1": t["binary"]["f1"],
        "params_total": r["params"]["total"], "params_cbam": r["params"]["cbam"],
        "train_minutes": r["timing"]["train_seconds"] / 60, "epochs_run": r["epochs_run"],
        "best_epoch": r["best"]["epoch"], "best_val_macro_f1": r["best"]["val_macro_f1"],
        "infer_ms": r["timing"]["inference"]["median_ms"], "infer_device": r["timing"]["inference"]["device"],
        "commit": r["code_version"]["commit"][:10], "gpu": r["env"].get("gpu", r["env"]["device"]),
        "torch": r["env"].get("torch"), "cuda": cuda_version_of(r["env"]),
        "epoch_seconds_stable": stable_epoch_seconds(r.get("history") or []),
    }
    for c in CLASSES:
        pc = t["per_class"][c]
        row.update({f"recall_{c}": pc["recall"], f"precision_{c}": pc["precision"], f"f1_{c}": pc["f1"]})
    interp = r.get("interpretation") or {}
    for key, value in (interp.get("summary") or {}).items():
        if key.endswith("_mean"):
            row[key[:-len("_mean")]] = value
    return row


def summarize(runs_df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in SUMMARY_COLS if c in runs_df] + sorted(c for c in runs_df if c.startswith("in_leaf_"))
    g = runs_df.groupby(["experiment", "config", "fraction"])
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out.insert(0, "n_seeds", g.size())
    return out.reset_index()


class Index:
    """Predicciones y valores por imagen de cada corrida, alineados por imagen."""

    def __init__(self, runs: list[dict]):
        self.by_key: dict[tuple, list[dict]] = {}
        ref = None
        for r in runs:
            preds = sorted(r["predictions"], key=lambda p: p["image"])
            images = [p["image"] for p in preds]
            if ref is None:
                ref = images
                self.images = np.array(images)
                self.plants = np.array([p["plant"] for p in preds])
                self.y_true = np.array([p["y_true"] for p in preds])
            elif images != ref:
                raise ValueError(f"{r['run']}: el conjunto de prueba no coincide con el de otras corridas")
            entry = {"seed": r["seed"], "y_pred": np.array([p["y_pred"] for p in preds]), "in_leaf": {}}
            per_image = (r.get("interpretation") or {}).get("per_image") or []
            if per_image:
                lut = {row["image"]: row for row in per_image}
                for meth in r["interpretation"]["methods"]:
                    entry["in_leaf"][meth] = np.array([lut[i][f"in_leaf_{meth}"] for i in images], dtype=float)
            key = (r["experiment"], r["config_name"], float(r["fraction"]))
            self.by_key.setdefault(key, []).append(entry)

    def metric_stats(self, key, metric):
        return [metric_stat(self.y_true, e["y_pred"], metric) for e in self.by_key.get(key, [])]

    def leaf_stats(self, key, method):
        return [mean_stat(e["in_leaf"][method]) for e in self.by_key.get(key, []) if method in e["in_leaf"]]


def bootstrap_tables(runs: list[dict], reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not runs:
        return pd.DataFrame(), pd.DataFrame()
    idx = Index(runs)
    samples = plant_bootstrap_indices(idx.plants, reps)
    boot_rows = []
    for key in sorted(idx.by_key):
        for metric in METRICS:
            ci = bootstrap_ci(idx.metric_stats(key, metric), samples)
            boot_rows.append({"experiment": key[0], "config": key[1], "fraction": key[2], "metric": metric,
                              "n_seeds": len(idx.by_key[key]), **ci})
        methods = idx.by_key[key][0]["in_leaf"].keys()
        for meth in methods:
            ci = bootstrap_ci(idx.leaf_stats(key, meth), samples)
            boot_rows.append({"experiment": key[0], "config": key[1], "fraction": key[2],
                              "metric": f"in_leaf_{meth}", "n_seeds": len(idx.by_key[key]), **ci})

    comp_rows = []
    for label, hyp, a, b in COMPARISONS:
        if a in idx.by_key and b in idx.by_key:
            for metric in METRICS:
                ci = bootstrap_ci(idx.metric_stats(a, metric), samples, idx.metric_stats(b, metric))
                comp_rows.append({"comparison": label, "hypothesis": hyp, "metric": metric,
                                  "a": "/".join(map(str, a)), "b": "/".join(map(str, b)),
                                  "n_a": len(idx.by_key[a]), "n_b": len(idx.by_key[b]), **ci})
    for label, a, b, ma, mb in H4_COMPARISONS:
        sa, sb = idx.leaf_stats(a, ma), idx.leaf_stats(b, mb)
        if sa and sb:
            ci = bootstrap_ci(sa, samples, sb)
            comp_rows.append({"comparison": label, "hypothesis": "H4", "metric": f"in_leaf_{ma}-{mb}",
                              "a": "/".join(map(str, a)), "b": "/".join(map(str, b)),
                              "n_a": len(sa), "n_b": len(sb), **ci})
    return pd.DataFrame(boot_rows), pd.DataFrame(comp_rows)


def aggregate(results_root: Path, out_dir: Path, reps: int = 2000) -> dict[str, pd.DataFrame]:
    runs = load_runs(results_root)
    if not runs:
        raise SystemExit(f"No hay corridas en {results_root / 'runs'}")
    runs_df = pd.DataFrame([run_row(r) for r in runs]).sort_values(["experiment", "config", "fraction", "seed"])
    tables = {"runs": runs_df, "summary": summarize(runs_df)}
    tables["bootstrap"], tables["comparisons"] = bootstrap_tables(runs, reps)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    print(f"{len(runs)} corridas agregadas -> {out_dir}")
    return tables


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-root")
    parser.add_argument("--out")
    parser.add_argument("--reps", type=int, default=2000)
    args = parser.parse_args(argv)
    root = resolve_dir(args.results_root, "ROCOLE_RESULTS_ROOT", "resultados")
    aggregate(root, Path(args.out) if args.out else root / "aggregate", args.reps)


if __name__ == "__main__":
    main()
