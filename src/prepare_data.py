"""Descarga RoCoLe y genera la caché de entrenamiento.

Uso:
    # Descargar de Mendeley (2,25 GB) y generar la caché en DATA_ROOT
    python -m src.prepare_data --data-root DATA_ROOT --download

    # Si ya tienes las carpetas Photos/ y Annotations/ (descarga manual)
    python -m src.prepare_data --data-root DATA_ROOT --raw-dir RUTA/AL/DATASET

Salidas en DATA_ROOT:
    rocole_256.npz  imágenes uint8 (N,256,256,3), máscaras de hoja (N,256,256)
                    y nombres de archivo, en orden alfabético.
    metadata.csv    una fila por imagen: clase, lote, planta, vista, tamaño
                    original, orientación EXIF y fracción de área de la hoja.

Por qué una caché: decodificar JPEG de hasta 4128x2322 en cada época con los
2 vCPU de Colab sería el cuello de botella. Se decodifica una sola vez, se
redimensiona a 256x256 (deformando 16:9 a cuadrado para no cortar la hoja) y
todo cabe en RAM (~300 MB).
"""
from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from PIL import Image

from .data import load_annotations, polygon_to_mask
from .utils import CACHE_FILENAME

MENDELEY_ZIP_URL = "https://data.mendeley.com/public-api/zip/c5yvn32dzg/download/2"
EXPECTED_ZIP_BYTES = 2_245_588_288
EXPECTED_COUNTS = {
    "healthy": 791,
    "red_spider_mite": 167,
    "rust_level_1": 344,
    "rust_level_2": 166,
    "rust_level_3": 62,
    "rust_level_4": 30,
}
CACHE_SIZE = 256


def download_zip(dest: Path) -> Path:
    """Descarga el zip completo del repositorio oficial (versión 2)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == EXPECTED_ZIP_BYTES:
        print(f"Zip ya descargado: {dest}")
        return dest
    print(f"Descargando {MENDELEY_ZIP_URL} -> {dest} (2,25 GB)...")
    with requests.get(MENDELEY_ZIP_URL, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        done = 0
        with open(dest, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=8 << 20):
                fh.write(chunk)
                done += len(chunk)
                if done % (256 << 20) < (8 << 20):
                    print(f"  {done / 1e9:.2f} GB")
    size = dest.stat().st_size
    if size != EXPECTED_ZIP_BYTES:
        raise SystemExit(f"Tamaño inesperado del zip ({size} bytes). Descárgalo a mano.")
    return dest


def extract_needed(zip_path: Path, out_dir: Path) -> Path:
    """Extrae solo Photos/ y las anotaciones JSON/XLSX (omite el VOC de 700 MB)."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [
            m for m in zf.namelist()
            if "/Photos/" in m or m.endswith((".json", ".xlsx", ".csv"))
        ]
        zf.extractall(out_dir, members)
    roots = {Path(m).parts[0] for m in members}
    return out_dir / roots.pop()


def find_raw_dir(base: Path) -> Path:
    """Busca la carpeta que contiene Photos/ y Annotations/."""
    for candidate in [base, *base.rglob("*")]:
        if (candidate / "Photos").is_dir() and (candidate / "Annotations").is_dir():
            return candidate
    raise SystemExit(f"No encontré Photos/ y Annotations/ dentro de {base}")


def build_cache(raw_dir: Path, data_root: Path) -> pd.DataFrame:
    ann = load_annotations(raw_dir / "Annotations")
    counts = ann.label.value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"Conteos por clase distintos a los esperados: {counts}")
    plants = ann.groupby("plant").size()
    if len(plants) != 390 or (plants != 4).any():
        raise SystemExit("Se esperaban 390 plantas con 4 imágenes cada una.")

    n = len(ann)
    images = np.zeros((n, CACHE_SIZE, CACHE_SIZE, 3), dtype=np.uint8)
    masks = np.zeros((n, CACHE_SIZE, CACHE_SIZE), dtype=np.uint8)
    meta = []
    for i, row in ann.iterrows():
        path = raw_dir / "Photos" / row.image
        with Image.open(path) as probe:
            width, height = probe.size
            orientation = probe.getexif().get(274)
        img = Image.open(path)
        # draft acelera la decodificación JPEG reduciendo por potencias de 2,
        # siempre a un tamaño >= al pedido; luego se redimensiona con antialias.
        img.draft("RGB", (2 * CACHE_SIZE, 2 * CACHE_SIZE))
        img.load()
        img = img.convert("RGB")  # sin exif_transpose, a propósito (ver data.py)
        images[i] = np.asarray(img.resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC))
        mask_full = polygon_to_mask(row.polygon, width, height)
        masks[i] = polygon_to_mask(row.polygon, width, height, (CACHE_SIZE, CACHE_SIZE))
        meta.append({
            "image": row.image, "label": row.label, "lot": row.lot, "plant": row.plant,
            "view": row.view, "width": width, "height": height,
            "exif_orientation": orientation, "leaf_area_frac": round(float(mask_full.mean()), 4),
        })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n} imágenes procesadas")

    data_root.mkdir(parents=True, exist_ok=True)
    np.savez(
        data_root / CACHE_FILENAME,
        images=images, masks=masks, names=np.array(ann.image.tolist()),
    )
    meta_df = pd.DataFrame(meta)
    meta_df.to_csv(data_root / "metadata.csv", index=False)
    print(f"Caché guardada en {data_root / CACHE_FILENAME}")
    return meta_df


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Carpeta donde se guarda la caché")
    parser.add_argument("--raw-dir", help="Carpeta con Photos/ y Annotations/ (descarga manual)")
    parser.add_argument("--download", action="store_true", help="Descargar el zip de Mendeley")
    args = parser.parse_args(argv)

    data_root = Path(args.data_root).expanduser()
    if args.raw_dir:
        raw_dir = find_raw_dir(Path(args.raw_dir).expanduser())
    elif args.download:
        zip_path = download_zip(data_root / "raw" / "rocole.zip")
        raw_dir = extract_needed(zip_path, data_root / "raw")
    else:
        raw_dir = find_raw_dir(data_root / "raw")
    print(f"Datos crudos: {raw_dir}")
    build_cache(raw_dir, data_root)


if __name__ == "__main__":
    main()
