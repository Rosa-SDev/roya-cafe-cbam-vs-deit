"""Fixtures compartidas: datos sintéticos para que los tests no necesiten el dataset."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import CACHE_FILENAME, CLASSES  # noqa: E402


@pytest.fixture
def synthetic_data(tmp_path):
    """Caché y split.csv sintéticos: 12 plantas x 4 vistas = 48 imágenes, 6 clases."""
    rng = np.random.default_rng(0)
    rows = []
    for p in range(12):
        for k, view in enumerate(["H1", "H2", "E1", "E2"]):
            split = "train" if p < 8 else ("val" if p < 10 else "test")
            rows.append({"image": f"C1P{p + 1}{view}.jpg", "plant": f"C1P{p + 1}", "lot": 1, "view": view,
                         "label": CLASSES[(p + k) % len(CLASSES)], "width": 1280, "height": 720, "split": split})
    df = pd.DataFrame(rows).sort_values("image").reset_index(drop=True)
    df["in_f100"] = (df.split == "train").astype(int)
    df["in_f50"] = df.plant.isin([f"C1P{i}" for i in range(1, 5)]).astype(int)
    df["in_f25"] = df.plant.isin(["C1P1", "C1P2"]).astype(int)
    n = len(df)
    images = rng.integers(0, 255, size=(n, 64, 64, 3), dtype=np.uint8)
    masks = np.zeros((n, 64, 64), dtype=np.uint8)
    masks[:, 16:48, 8:56] = 1
    data_root = tmp_path / "data"
    data_root.mkdir()
    np.savez(data_root / CACHE_FILENAME, images=images, masks=masks, names=np.array(df.image.tolist()))
    split_csv = tmp_path / "split.csv"
    df.to_csv(split_csv, index=False)
    return {"data_root": data_root, "split_csv": split_csv, "df": df, "results_root": tmp_path / "results"}
