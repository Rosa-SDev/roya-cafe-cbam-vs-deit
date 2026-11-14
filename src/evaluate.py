"""Métricas de evaluación, bootstrap por planta y tiempo de inferencia.

Bootstrap por planta: las 4 fotos de una planta no son independientes. Por eso
se remuestrean plantas con reemplazo, no imágenes; cada planta aporta todas
sus imágenes de prueba. Con los mismos índices remuestreados para todas las
semillas y configuraciones se obtienen:
- el IC95 de la media entre semillas de cada configuración, y
- el IC95 de diferencias pareadas entre configuraciones (bootstrap_ci).
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from .utils import CLASSES, HEALTHY_IDX

K = len(CLASSES)


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def to_binary(y: np.ndarray) -> np.ndarray:
    """Colapsa las 6 clases: 0 = sana, 1 = enferma (ácaro y roya cuentan como enferma)."""
    return (np.asarray(y) != HEALTHY_IDX).astype(int)


def classification_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(K))
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    bt, bp = to_binary(y_true), to_binary(y_pred)
    bprec, brec, bf1, _ = precision_recall_fscore_support(bt, bp, average="binary", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            c: {"precision": float(prec[i]), "recall": float(rec[i]), "f1": float(f1[i]), "support": int(support[i])}
            for i, c in enumerate(CLASSES)
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "binary": {
            "accuracy": float(accuracy_score(bt, bp)),
            "f1": float(bf1),
            "precision": float(bprec),
            "recall": float(brec),
            "confusion_matrix": confusion_matrix(bt, bp, labels=[0, 1]).tolist(),
        },
    }


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int = K) -> float:
    """F1 macro vectorizado (igual a sklearn con zero_division=0), para el bootstrap."""
    cm = np.bincount(y_true * k + y_pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1.mean())


def fast_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


METRIC_FNS = {
    "macro_f1": fast_macro_f1,
    "accuracy": fast_accuracy,
    # F1 de la clase "enferma" (igual que sklearn con average="binary").
    "binary_f1": lambda t, p: _f1_class(to_binary(t), to_binary(p), 1),
}


def _f1_class(y_true, y_pred, cls) -> float:
    tp = np.sum((y_true == cls) & (y_pred == cls))
    denom = np.sum(y_true == cls) + np.sum(y_pred == cls)
    return float(2 * tp / denom) if denom else 0.0


# ---------------------------------------------------------------------------
# Bootstrap por planta
# ---------------------------------------------------------------------------
def plant_bootstrap_indices(plants: np.ndarray, reps: int, seed: int = 12345) -> list[np.ndarray]:
    """Lista de índices de imágenes, una por réplica, remuestreando plantas."""
    plants = np.asarray(plants)
    uniq, inverse = np.unique(plants, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        chosen = rng.integers(0, len(uniq), size=len(uniq))
        out.append(np.concatenate([members[c] for c in chosen]))
    return out


def metric_stat(y_true: np.ndarray, y_pred: np.ndarray, metric: str):
    """Función idx -> valor de la métrica sobre esas imágenes (idx=None: todas)."""
    fn = METRIC_FNS[metric]
    return lambda idx: fn(y_true, y_pred) if idx is None else fn(y_true[idx], y_pred[idx])


def mean_stat(values: np.ndarray):
    """Función idx -> media de valores por imagen (p. ej. fracción en la hoja)."""
    values = np.asarray(values, dtype=float)
    return lambda idx: float(np.nanmean(values if idx is None else values[idx]))


def bootstrap_ci(stats_a: list, samples: list[np.ndarray], stats_b: list | None = None,
                 alpha: float = 0.05) -> dict:
    """IC por bootstrap de la media entre semillas (o de la diferencia A - B).

    `stats_a`/`stats_b`: una función idx -> valor por semilla. Todas usan los
    mismos remuestreos (`samples`), así la diferencia es pareada.
    """
    def value(stats, idx):
        return float(np.mean([f(idx) for f in stats]))

    point = value(stats_a, None) - (value(stats_b, None) if stats_b else 0.0)
    boot = np.array([value(stats_a, s) - (value(stats_b, s) if stats_b else 0.0) for s in samples])
    lo, hi = np.nanquantile(boot, [alpha / 2, 1 - alpha / 2])
    out = {"point": point, "ci_low": float(lo), "ci_high": float(hi)}
    if stats_b:
        out["p_le_0"] = float(np.mean(boot <= 0))  # p unilateral aproximada de A > B
    return out


# ---------------------------------------------------------------------------
# Predicción y tiempos
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader, device, amp: bool = False) -> dict:
    model.eval()
    all_idx, all_true, all_prob = [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp and device.type == "cuda"):
            logits = model(x)
        all_prob.append(torch.softmax(logits.float(), dim=1).cpu())
        all_true.append(y)
        all_idx.append(idx)
    prob = torch.cat(all_prob).numpy()
    return {
        "index": torch.cat(all_idx).numpy(),
        "y_true": torch.cat(all_true).numpy(),
        "y_pred": prob.argmax(1),
        "prob": prob,
    }


@torch.no_grad()
def inference_time_ms(model, device, image_size: int = 224, iters: int = 100, warmup: int = 20) -> dict:
    """Latencia por imagen con batch 1 (mediana), en el dispositivo de entrenamiento."""
    model.eval()
    x = torch.randn(1, 3, image_size, image_size, device=device)
    if device.type == "cpu":
        iters, warmup = min(iters, 20), min(warmup, 3)
    times = []
    for i in range(warmup + iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append((time.perf_counter() - t0) * 1000)
    return {"median_ms": float(np.median(times)), "mean_ms": float(np.mean(times)),
            "device": device.type, "batch_size": 1, "iters": iters}
