"""Partición por planta, estratificada por clase en la medida de lo posible.

Uso:
    python -m src.split --data-root DATA_ROOT        # crea data/splits/split.csv

Problema: la clase es por imagen y no por planta (379 de 390 plantas tienen
fotos de clases distintas). No se puede estratificar exactamente sin romper la
regla de que una planta va a un solo conjunto. Solución: búsqueda aleatoria
con semilla fija. Se prueban muchas permutaciones de plantas, se corta en
70/15/15 (por número de plantas) y se elige la que minimiza la mayor
desviación entre la proporción de cada clase en cada conjunto y la objetivo.

Subconjuntos de E3: del 25 % y del 50 % de las plantas de entrenamiento,
anidados (las del 25 % están dentro del 50 %), elegidos con el mismo criterio.
Validación y prueba no cambian.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import CLASSES, SPLIT_CSV

SEED = 42
N_TRIALS = 20_000
SPLIT_FRACS = {"train": 0.70, "val": 0.15, "test": 0.15}
SUBSET_FRACS = (0.25, 0.50)


def plant_class_counts(meta: pd.DataFrame) -> pd.DataFrame:
    """Matriz plantas x clases con el número de imágenes."""
    return pd.crosstab(meta.plant, meta.label).reindex(columns=CLASSES, fill_value=0).sort_index()


def _cost(counts: np.ndarray, groups: list[np.ndarray], targets: list[float]) -> float:
    """Mayor desviación absoluta entre la proporción de cada clase y la objetivo.

    Penaliza fuerte que algún grupo se quede sin alguna clase.
    """
    total = counts.sum(0)
    cost = 0.0
    for idx, target in zip(groups, targets):
        sub = counts[idx].sum(0)
        cost = max(cost, float(np.abs(sub / total - target).max()))
        if (sub == 0).any():
            cost += 1.0
    return cost


def split_plants(counts: pd.DataFrame, seed: int = SEED, n_trials: int = N_TRIALS) -> dict[str, list[str]]:
    plants = counts.index.to_numpy()
    mat = counts.to_numpy()
    n = len(plants)
    n_train = round(SPLIT_FRACS["train"] * n)
    n_val = round(SPLIT_FRACS["val"] * n)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_trials):
        perm = rng.permutation(n)
        groups = [perm[:n_train], perm[n_train:n_train + n_val], perm[n_train + n_val:]]
        cost = _cost(mat, groups, list(SPLIT_FRACS.values()))
        if best is None or cost < best[0]:
            best = (cost, groups)
    cost, groups = best
    print(f"Partición: desviación máxima de proporción por clase = {cost:.4f}")
    return {name: sorted(plants[g]) for name, g in zip(SPLIT_FRACS, groups)}


def nested_subsets(counts: pd.DataFrame, train_plants: list[str], seed: int = SEED,
                   n_trials: int = N_TRIALS) -> dict[float, list[str]]:
    """Subconjuntos anidados de plantas de entrenamiento (25 % dentro de 50 %)."""
    sub = counts.loc[train_plants]
    plants = sub.index.to_numpy()
    mat = sub.to_numpy()
    sizes = [round(f * len(plants)) for f in SUBSET_FRACS]
    rng = np.random.default_rng(seed + 1)
    best = None
    for _ in range(n_trials):
        perm = rng.permutation(len(plants))
        groups = [perm[:s] for s in sizes]
        cost = _cost(mat, groups, list(SUBSET_FRACS))
        if best is None or cost < best[0]:
            best = (cost, groups)
    cost, groups = best
    print(f"Subconjuntos E3: desviación máxima de proporción por clase = {cost:.4f}")
    return {f: sorted(plants[g]) for f, g in zip(SUBSET_FRACS, groups)}


def make_split(meta: pd.DataFrame, seed: int = SEED, n_trials: int = N_TRIALS) -> pd.DataFrame:
    counts = plant_class_counts(meta)
    parts = split_plants(counts, seed, n_trials)
    plant_to_split = {p: name for name, ps in parts.items() for p in ps}
    subsets = nested_subsets(counts, parts["train"], seed, n_trials)

    out = meta[["image", "plant", "lot", "view", "label", "width", "height"]].copy()
    out["split"] = out.plant.map(plant_to_split)
    for frac, plants in subsets.items():
        out[f"in_f{int(frac * 100)}"] = out.plant.isin(plants).astype(int)
    out["in_f100"] = (out.split == "train").astype(int)
    return out.sort_values("image").reset_index(drop=True)


def check_split(df: pd.DataFrame) -> None:
    """Verificaciones que también corren en los tests."""
    per_plant = df.groupby("plant").split.nunique()
    assert (per_plant == 1).all(), "Hay plantas en más de un conjunto"
    train = df[df.split == "train"]
    assert df.loc[df.split != "train", ["in_f25", "in_f50"]].to_numpy().sum() == 0
    assert (train.in_f25 <= train.in_f50).all(), "El 25 % no está dentro del 50 %"
    for col in ["in_f25", "in_f50"]:
        assert train[train[col] == 1].groupby("plant").size().eq(4).all()
        assert set(train[train[col] == 1].label) == set(CLASSES), f"{col} no tiene todas las clases"


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Plantas e imágenes por conjunto y por clase (para el artículo)."""
    rows = []
    groups = [("train", df.split == "train"), ("val", df.split == "val"), ("test", df.split == "test"),
              ("train_f50", df.in_f50 == 1), ("train_f25", df.in_f25 == 1)]
    for name, sel in groups:
        sub = df[sel]
        row = {"set": name, "plants": sub.plant.nunique(), "images": len(sub)}
        row.update(sub.label.value_counts().reindex(CLASSES, fill_value=0).to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Carpeta con metadata.csv (de prepare_data)")
    parser.add_argument("--out", default=str(SPLIT_CSV))
    args = parser.parse_args(argv)

    meta = pd.read_csv(Path(args.data_root) / "metadata.csv")
    df = make_split(meta)
    check_split(df)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(summary_table(df).to_string(index=False))
    print(f"Guardado en {args.out}")


if __name__ == "__main__":
    main()
