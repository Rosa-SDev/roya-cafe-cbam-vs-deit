"""Exploración: distribución de clases por lote (C) y por resolución.

Uso:
    python -m src.explore --data-root DATA_ROOT [--out docs/exploracion]

Objetivo: saber si la cámara (que se refleja en la resolución) o el lote
podrían servir de atajo al modelo. Se reporta; NO se cambia la partición.

Para cada factor se calcula una tabla de contingencia con la clase, la prueba
chi-cuadrado de independencia y la V de Cramér (0 = independiente, 1 = el
factor determina la clase). Ojo: todas las imágenes se reducen a 256x256 antes
de entrenar, así que la resolución original no llega directo al modelo; pero
la cámara también cambia color, nitidez y compresión, que sí podrían llegar.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .utils import CLASSES


def add_resolution(meta: pd.DataFrame) -> pd.DataFrame:
    meta = meta.copy()
    meta["resolution"] = meta.width.astype(str) + "x" + meta.height.astype(str)
    return meta


def contingency(meta: pd.DataFrame, factor: str) -> pd.DataFrame:
    return pd.crosstab(meta[factor], meta.label).reindex(columns=CLASSES, fill_value=0)


def association(table: pd.DataFrame) -> dict:
    chi2, p, dof, _ = chi2_contingency(table.to_numpy())
    n = table.to_numpy().sum()
    v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"chi2": float(chi2), "dof": int(dof), "p_value": float(p), "cramers_v": float(v)}


def row_percent(table: pd.DataFrame) -> pd.DataFrame:
    return (100 * table.div(table.sum(axis=1), axis=0)).round(1)


def report(meta: pd.DataFrame, split: pd.DataFrame | None = None) -> dict[str, pd.DataFrame | dict]:
    meta = add_resolution(meta)
    out = {}
    for factor in ["lot", "resolution"]:
        tab = contingency(meta, factor)
        out[f"class_by_{factor}"] = tab
        out[f"class_by_{factor}_pct"] = row_percent(tab)
        out[f"assoc_{factor}"] = association(tab)
    out["resolution_by_lot"] = pd.crosstab(meta.lot, meta.resolution)
    if split is not None:
        s = split.merge(meta[["image", "resolution"]], on="image")
        out["resolution_by_split"] = pd.crosstab(s.split, s.resolution)
        out["lot_by_split"] = pd.crosstab(s.split, s.lot)
    return out


def to_markdown(results: dict) -> str:
    lines = ["# Exploración: clase por lote y por resolución", ""]
    for key, value in results.items():
        lines.append(f"## {key}")
        lines.append("")
        if isinstance(value, pd.DataFrame):
            lines.append(value.to_markdown() if hasattr(value, "to_markdown") else value.to_string())
        else:
            lines.append(", ".join(f"{k} = {v:.4g}" if isinstance(v, float) else f"{k} = {v}" for k, v in value.items()))
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default=None, help="split.csv (opcional)")
    parser.add_argument("--out", default=None, help="Carpeta donde guardar CSV")
    args = parser.parse_args(argv)
    meta = pd.read_csv(Path(args.data_root) / "metadata.csv")
    split = pd.read_csv(args.split) if args.split else None
    res = report(meta, split)
    for key, value in res.items():
        print(f"\n== {key} ==")
        print(value.to_string() if isinstance(value, pd.DataFrame) else value)
        if args.out and isinstance(value, pd.DataFrame):
            Path(args.out).mkdir(parents=True, exist_ok=True)
            value.to_csv(Path(args.out) / f"{key}.csv")


if __name__ == "__main__":
    main()
