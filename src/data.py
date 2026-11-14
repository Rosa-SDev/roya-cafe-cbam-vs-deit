"""Lectura de RoCoLe: etiquetas, identificador de planta, máscaras y Dataset.

Hechos verificados sobre los datos (ver docs/decisiones.md):
- Las anotaciones de Labelbox están en Annotations/RoCoLE-json.json: un
  registro por imagen con la clase (Label.classification) y un polígono de la
  hoja (Label.Leaf[0].geometry) en píxeles.
- El identificador de planta está en el nombre: C<lote>P<planta><H|E><1|2>.jpg.
  Hay una anomalía, C11P11HE.jpg, que es la foto H2 de la planta C11P11.
- Todas las fotos traen la etiqueta EXIF de orientación (6 u 8), pero los
  polígonos están en las coordenadas del JPEG SIN rotar. Por eso aquí nunca se
  aplica ImageOps.exif_transpose: hacerlo desalinearía imagen y máscara.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import v2

from .utils import CACHE_FILENAME, CLASS_TO_IDX, CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FILENAME_RE = re.compile(r"^C(\d+)P(\d+)([HE])([0-9A-Z]+)\.jpg$")
# Nombres anómalos conocidos -> vista que les corresponde.
KNOWN_ANOMALIES = {"C11P11HE.jpg": "H2"}


def parse_filename(name: str) -> dict:
    """Extrae lote, planta y vista del nombre de archivo."""
    match = FILENAME_RE.match(name)
    if match is None:
        raise ValueError(f"Nombre de archivo inesperado: {name}")
    lot, plant_num, side, idx = match.groups()
    view = KNOWN_ANOMALIES.get(name, side + idx)
    if view not in {"H1", "H2", "E1", "E2"}:
        raise ValueError(f"Vista desconocida en {name}: {view}")
    return {
        "lot": int(lot),
        "plant": f"C{int(lot)}P{int(plant_num)}",
        "view": view,
    }


def load_annotations(annotations_dir: str | Path) -> pd.DataFrame:
    """Lee RoCoLE-json.json y devuelve una tabla con una fila por imagen."""
    path = Path(annotations_dir) / "RoCoLE-json.json"
    with open(path, encoding="utf-8") as fh:
        records = json.load(fh)
    rows = []
    for rec in records:
        name = rec["External ID"]
        label = rec["Label"]["classification"]
        leaves = rec["Label"]["Leaf"]
        if len(leaves) != 1:
            raise ValueError(f"{name}: se esperaba 1 polígono y hay {len(leaves)}")
        polygon = [(float(p["x"]), float(p["y"])) for p in leaves[0]["geometry"]]
        rows.append({"image": name, "label": label, "polygon": polygon, **parse_filename(name)})
    df = pd.DataFrame(rows).sort_values("image").reset_index(drop=True)
    unknown = set(df.label) - set(CLASSES)
    if unknown:
        raise ValueError(f"Clases desconocidas: {unknown}")
    return df


def open_raw_image(path: str | Path) -> Image.Image:
    """Abre el JPEG sin aplicar la rotación EXIF (a propósito, ver arriba)."""
    img = Image.open(path)
    img.load()
    return img.convert("RGB")


def polygon_to_mask(polygon, width: int, height: int, out_size: tuple[int, int] | None = None) -> np.ndarray:
    """Rasteriza el polígono de la hoja como máscara binaria uint8 (1 = hoja).

    Si se da `out_size` (ancho, alto), las coordenadas se escalan a ese tamaño,
    que equivale a redimensionar la imagen deformándola igual que la foto.
    """
    out_w, out_h = out_size or (width, height)
    sx, sy = out_w / width, out_h / height
    pts = [(x * sx, y * sy) for x, y in polygon]
    mask = Image.new("L", (out_w, out_h), 0)
    ImageDraw.Draw(mask).polygon(pts, outline=1, fill=1)
    return np.asarray(mask, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Caché
# ---------------------------------------------------------------------------
def load_cache(data_root: str | Path) -> dict:
    """Carga data_root/rocole_256.npz (imágenes, máscaras y nombres)."""
    path = Path(data_root) / CACHE_FILENAME
    if not path.exists():
        raise SystemExit(
            f"No existe {path}. Genéralo con: python -m src.prepare_data --data-root {data_root}"
        )
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


# ---------------------------------------------------------------------------
# Transformaciones (idénticas para todos los modelos)
# ---------------------------------------------------------------------------
def build_transforms(cfg: dict, train: bool) -> v2.Compose:
    """Aumento solo en entrenamiento; no se toca el tono (hue).

    Las imágenes llegan como tensores uint8 de 256x256 (ya deformadas de 16:9 a
    cuadrado). En evaluación solo se redimensiona a 224 y se normaliza.
    """
    size = cfg["data"]["image_size"]
    aug = cfg["augment"]
    final = [v2.ToDtype(torch.float32, scale=True), v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train:
        return v2.Compose([v2.Resize((size, size), antialias=True), *final])
    return v2.Compose(
        [
            v2.RandomResizedCrop(size, scale=tuple(aug["rrc_scale"]), antialias=True),
            v2.RandomHorizontalFlip(),
            v2.RandomVerticalFlip(),
            v2.RandomRotation(aug["rotation_deg"]),
            v2.ColorJitter(brightness=aug["brightness"], contrast=aug["contrast"]),
            *final,
        ]
    )


class RoCoLeDataset(Dataset):
    """Dataset en memoria sobre la caché.

    `images` es un arreglo uint8 (N, H, W, 3) compartido por todos los
    subconjuntos; `indices` selecciona las filas de este subconjunto.
    """

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices, transform):
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)  # vista N,3,H,W sin copia
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        return self.transform(self.images[idx]), self.labels[idx], idx


def labels_from_names(names, split_df: pd.DataFrame) -> np.ndarray:
    """Etiquetas numéricas en el orden de `names` usando split.csv."""
    lut = dict(zip(split_df.image, split_df.label))
    return np.array([CLASS_TO_IDX[lut[n]] for n in names], dtype=np.int64)
