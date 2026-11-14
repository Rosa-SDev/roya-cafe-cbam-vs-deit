"""Une en results/runs/ los JSON de corridas hechas en otra máquina (p. ej. Colab).

Uso:
    python -m src.merge_runs RUTA [RUTA ...] [--dest results] [--dry-run]

RUTA puede ser una carpeta (se busca recursivamente) o el .zip que genera
Google Drive al descargar una carpeta. Para cada JSON de corrida:
- se identifica por su contenido (experimento y nombre de corrida), no por la
  ruta, así da igual cómo quedaron las carpetas al descargar;
- si no existe en destino, se copia a DEST/runs/<experimento>/<corrida>.json,
  junto con su <corrida>_maps.npz si viene en la misma carpeta;
- si ya existe y es idéntico, se omite;
- si ya existe y es distinto, NO se sobrescribe: se reporta como conflicto
  (la misma corrida hecha dos veces). Con --overwrite se reemplaza.
Se ignoran las corridas de depuración (--limit) y los experimentos 'bench' y
'debug'. Nunca se copian pesos (.pt).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from .utils import REPO_ROOT

SKIP_EXPERIMENTS = {"bench", "debug", "smoke"}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _iter_run_jsons(root: Path):
    for path in sorted(root.rglob("*.json")):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if isinstance(data, dict) and {"run", "experiment", "config_name", "test"} <= data.keys():
            yield path, data


def merge(sources: list[Path], dest: Path, overwrite: bool = False, dry_run: bool = False) -> dict:
    stats = {"copied": [], "identical": [], "conflicts": [], "skipped": []}
    with tempfile.TemporaryDirectory() as tmp:
        roots = []
        for i, src in enumerate(sources):
            if src.is_file() and src.suffix == ".zip":
                target = Path(tmp) / f"zip{i}"
                with zipfile.ZipFile(src) as zf:
                    zf.extractall(target, [m for m in zf.namelist() if m.endswith((".json", ".npz"))])
                roots.append(target)
            elif src.is_dir():
                roots.append(src)
            else:
                raise SystemExit(f"No es carpeta ni .zip: {src}")
        for root in roots:
            for path, data in _iter_run_jsons(root):
                name, exp = data["run"], data["experiment"]
                if exp in SKIP_EXPERIMENTS or data.get("debug_limit"):
                    stats["skipped"].append(f"{exp}/{name}")
                    continue
                target = dest / "runs" / exp / f"{name}.json"
                if target.exists() and _digest(target) == _digest(path):
                    stats["identical"].append(f"{exp}/{name}")
                    continue
                if target.exists() and not overwrite:
                    stats["conflicts"].append(f"{exp}/{name}  (origen: {path})")
                    continue
                stats["copied"].append(f"{exp}/{name}")
                if not dry_run:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target)
                    maps = path.with_name(f"{name}_maps.npz")
                    if maps.exists():
                        shutil.copy2(maps, target.with_name(maps.name))
    return stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--dest", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--overwrite", action="store_true", help="Reemplazar corridas en conflicto")
    parser.add_argument("--dry-run", action="store_true", help="Solo mostrar qué se haría")
    args = parser.parse_args(argv)
    stats = merge(args.sources, args.dest, args.overwrite, args.dry_run)
    prefix = "[simulación] " if args.dry_run else ""
    print(f"{prefix}copiadas: {len(stats['copied'])} | ya estaban (idénticas): {len(stats['identical'])} | "
          f"conflictos: {len(stats['conflicts'])} | ignoradas (depuración): {len(stats['skipped'])}")
    for item in stats["copied"]:
        print("  +", item)
    if stats["conflicts"]:
        print("Conflictos (no se sobrescribieron; usa --overwrite si corresponde):")
        for item in stats["conflicts"]:
            print("  !", item)


if __name__ == "__main__":
    main()
