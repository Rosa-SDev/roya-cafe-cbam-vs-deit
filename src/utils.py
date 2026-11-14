"""Utilidades comunes: rutas, configuración, semillas, determinismo y E/S segura.

Ninguna ruta está fija en el código: la raíz de datos y la de resultados se
toman de argumentos de línea de comandos o de las variables de entorno
ROCOLE_DATA_ROOT y ROCOLE_RESULTS_ROOT.
"""
from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Raíz del repositorio (carpeta que contiene src/, configs/, data/splits/).
REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
SPLIT_CSV = REPO_ROOT / "data" / "splits" / "split.csv"

# Orden fijo de clases: el índice de cada clase es su etiqueta numérica.
CLASSES = [
    "healthy",
    "red_spider_mite",
    "rust_level_1",
    "rust_level_2",
    "rust_level_3",
    "rust_level_4",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
HEALTHY_IDX = CLASS_TO_IDX["healthy"]

CACHE_FILENAME = "rocole_256.npz"


# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
def resolve_dir(cli_value: str | None, env_var: str, purpose: str) -> Path:
    """Devuelve la ruta dada por argumento o, si falta, por variable de entorno."""
    value = cli_value or os.environ.get(env_var)
    if not value:
        raise SystemExit(
            f"Falta la ruta de {purpose}: pásala como argumento o define {env_var}."
        )
    return Path(value).expanduser()


def fraction_tag(fraction: float) -> str:
    """1.0 -> 'f100', 0.5 -> 'f50', 0.25 -> 'f25' (porcentaje entero)."""
    return f"f{int(round(fraction * 100))}"


def run_name(config_name: str, fraction: float, seed: int) -> str:
    return f"{config_name}_{fraction_tag(fraction)}_s{seed}"


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
def deep_update(base: dict, other: dict) -> dict:
    """Mezcla recursiva: los valores de `other` reemplazan a los de `base`."""
    out = copy.deepcopy(base)
    for key, value in other.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """Carga configs/base.yaml y encima el YAML de la configuración.

    `overrides` admite pares 'seccion.clave=valor' (el valor se interpreta
    como YAML, así 'train.epochs=1' queda como entero).
    """
    path = Path(path)
    with open(CONFIG_DIR / "base.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    with open(path, encoding="utf-8") as fh:
        cfg = deep_update(cfg, yaml.safe_load(fh) or {})
    cfg.setdefault("name", path.stem)
    for item in overrides or []:
        key, _, raw = item.partition("=")
        node = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return cfg


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    """Fija semillas de Python, NumPy y PyTorch y activa el modo determinista.

    El determinismo es "en la medida de lo posible": con warn_only=True las
    operaciones de CUDA sin versión determinista solo emiten un aviso. Aun así
    no se garantiza igualdad bit a bit entre máquinas distintas de Colab.
    """
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """Semilla de cada worker del DataLoader derivada de la semilla de PyTorch."""
    import torch

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ---------------------------------------------------------------------------
# Información del entorno
# ---------------------------------------------------------------------------
def git_commit() -> dict:
    """Commit actual del repositorio y si hay cambios sin confirmar."""
    def _git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()

    try:
        sha = _git("rev-parse", "HEAD")
        if len(sha) != 40:  # repositorio sin commits o git ausente
            sha = "unknown"
        dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    except Exception:  # git no disponible
        sha, dirty = "unknown", None
    return {"commit": sha, "dirty": dirty}


def environment_info() -> dict:
    import platform

    import timm
    import torch
    import torchvision

    info = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    return info


# ---------------------------------------------------------------------------
# E/S
# ---------------------------------------------------------------------------
def write_json_atomic(path: Path, data: Any) -> None:
    """Escribe primero un temporal y luego lo renombra.

    Si Colab se desconecta a mitad de escritura no queda un JSON corrupto, que
    haría creer que la corrida terminó.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
