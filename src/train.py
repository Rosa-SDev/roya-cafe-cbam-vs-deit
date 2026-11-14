"""Entrena y evalúa una corrida: una configuración, una semilla y una fracción de datos.

Uso:
    python -m src.train --config configs/cnn_cbam_scratch.yaml --seed 0 \
        --fraction 1.0 --experiment E1 --data-root DATA --results-root RESULTADOS

Rutas: --data-root / --results-root o las variables ROCOLE_DATA_ROOT /
ROCOLE_RESULTS_ROOT.

Salidas (reanudable: si el JSON ya existe, la corrida se omite):
    RESULTADOS/runs/<exp>/<config>_f<pct>_s<semilla>.json   métricas y predicciones
    RESULTADOS/runs/<exp>/<config>_f<pct>_s<semilla>_maps.npz  mapas de ejemplo
    RESULTADOS/checkpoints/<exp>/<...>.pt                    mejor modelo (no va a git)

Fracciones < 1 (E3): se iguala el número de pasos de optimización, no las
épocas. Épocas, warmup, paciencia y frecuencia de validación se escalan por
1/fracción (p. ej. 25 % desde cero: 240 épocas, warmup 20, paciencia 60,
validación cada 4 épocas). Así H3 mide el efecto de tener menos datos y no el
de entrenar menos, y el número de validaciones (oportunidades de elegir
checkpoint) es el mismo que con el 100 %.
"""
from __future__ import annotations

import argparse
import copy
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import RoCoLeDataset, build_transforms, labels_from_names, load_cache
from .evaluate import classification_metrics, inference_time_ms, predict
from .interpret import choose_examples, run_interpretation
from .models.factory import build_model, count_parameters, param_groups
from .utils import (CLASSES, SPLIT_CSV, Timer, environment_info, git_commit, load_config,
                    resolve_dir, run_name, seed_worker, set_seed, write_json_atomic)


# ---------------------------------------------------------------------------
# Piezas
# ---------------------------------------------------------------------------
def scaled_schedule(cfg: dict, fraction: float, steps_per_epoch: int) -> dict:
    t = cfg["train"]
    factor = 1.0 / fraction
    epochs = max(1, round(t["epochs"] * factor))
    return {
        "epochs": epochs,
        "warmup_epochs": min(epochs, round(t["warmup_epochs"] * factor)),
        "patience": round(t["patience"] * factor),
        "eval_every": max(1, round(factor)),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": epochs * steps_per_epoch,
    }


def class_weights(train_labels: np.ndarray, k: int = len(CLASSES)) -> torch.Tensor:
    """w_c = N / (K * n_c), calculado solo con entrenamiento (como sklearn 'balanced')."""
    counts = np.bincount(train_labels, minlength=k).astype(float)
    counts = np.maximum(counts, 1.0)  # evita división por cero en pruebas con pocas imágenes
    return torch.tensor(len(train_labels) / (k * counts), dtype=torch.float32)


def lr_lambda_factory(warmup_steps: int, total_steps: int, min_ratio: float):
    """Warmup lineal y luego coseno hasta min_ratio * lr."""
    def fn(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return fn


def select_indices(split_df: pd.DataFrame, names: np.ndarray, fraction: float, limit: int | None):
    """Índices de la caché para entrenamiento (según la fracción), validación y prueba."""
    pos = {n: i for i, n in enumerate(names)}
    col = f"in_f{int(round(fraction * 100))}"
    if col not in split_df:
        raise SystemExit(f"Fracción no soportada: {fraction} (no existe la columna {col} en split.csv)")
    sets = {
        "train": split_df[(split_df.split == "train") & (split_df[col] == 1)],
        "val": split_df[split_df.split == "val"],
        "test": split_df[split_df.split == "test"],
    }
    out = {}
    for name, sub in sets.items():
        idx = np.array(sorted(pos[n] for n in sub.image))
        if limit:  # solo para pruebas rápidas en CPU
            n = limit if name == "train" else max(4, limit // 3)
            idx = np.random.default_rng(0).permutation(idx)[:n]
            idx.sort()
        out[name] = idx
    return out


def make_loader(dataset, cfg, shuffle: bool, seed: int, device) -> DataLoader:
    workers = cfg["data"]["num_workers"]
    if sys.platform == "win32":
        workers = 0  # en Windows los workers copiarían la caché completa (spawn)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return DataLoader(
        dataset, batch_size=cfg["train"]["batch_size"], shuffle=shuffle, num_workers=workers,
        pin_memory=device.type == "cuda", worker_init_fn=seed_worker, generator=gen,
        persistent_workers=workers > 0, drop_last=False,
    )


def run_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, amp, grad_clip) -> float:
    model.train()
    total, n = 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        if grad_clip:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
    return total / max(1, n)


@torch.no_grad()
def validate(model, loader, criterion, device, amp) -> dict:
    out = predict(model, loader, device, amp)
    logits_like = torch.log(torch.from_numpy(out["prob"]).clamp_min(1e-12))
    loss = criterion.cpu()(logits_like, torch.from_numpy(out["y_true"])).item()
    criterion.to(device)
    m = classification_metrics(out["y_true"], out["y_pred"])
    return {"loss": loss, "macro_f1": m["macro_f1"], "accuracy": m["accuracy"]}


# ---------------------------------------------------------------------------
# Corrida completa
# ---------------------------------------------------------------------------
def train_run(cfg: dict, seed: int, fraction: float, experiment: str, data_root: Path,
              results_root: Path, limit: int | None = None, force: bool = False,
              save_checkpoint: bool = True, device_name: str | None = None) -> Path:
    name = run_name(cfg["name"], fraction, seed)
    out_json = results_root / "runs" / experiment / f"{name}.json"
    if out_json.exists() and not force:
        print(f"[omitida] {out_json} ya existe")
        return out_json

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    set_seed(seed, cfg["train"].get("deterministic", True))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg["train"]["amp"]) and device.type == "cuda"

    # Datos
    cache = load_cache(data_root)
    split_df = pd.read_csv(SPLIT_CSV)
    names = cache["names"]
    labels = labels_from_names(names, split_df)
    idx = select_indices(split_df, names, fraction, limit)
    tr_tf, ev_tf = build_transforms(cfg, True), build_transforms(cfg, False)
    loaders = {
        "train": make_loader(RoCoLeDataset(cache["images"], labels, idx["train"], tr_tf), cfg, True, seed, device),
        "val": make_loader(RoCoLeDataset(cache["images"], labels, idx["val"], ev_tf), cfg, False, seed, device),
        "test": make_loader(RoCoLeDataset(cache["images"], labels, idx["test"], ev_tf), cfg, False, seed, device),
    }
    sched = scaled_schedule(cfg, fraction, len(loaders["train"]))
    if limit:
        sched.update(epochs=cfg["train"]["epochs"], warmup_epochs=min(cfg["train"]["warmup_epochs"], cfg["train"]["epochs"]),
                     eval_every=1, total_steps=cfg["train"]["epochs"] * len(loaders["train"]))

    # Modelo y optimización
    model = build_model(cfg).to(device)
    params = count_parameters(model)
    weights = class_weights(labels[idx["train"]])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    t = cfg["train"]
    optimizer = torch.optim.AdamW(param_groups(model, t["weight_decay"]), lr=t["lr"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(sched["warmup_epochs"] * sched["steps_per_epoch"],
                                     sched["total_steps"], t["min_lr_ratio"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    print(f"[{name}] {experiment} | {device} | parámetros {params['total']:,} | "
          f"train {len(idx['train'])} val {len(idx['val'])} test {len(idx['test'])} | "
          f"épocas {sched['epochs']} (warmup {sched['warmup_epochs']}, paciencia {sched['patience']}, "
          f"validación cada {sched['eval_every']})")

    history, best = [], {"epoch": -1, "val_macro_f1": -1.0}
    best_state = None
    stopped_early = False
    with Timer() as train_timer:
        for epoch in range(1, sched["epochs"] + 1):
            t0 = time.perf_counter()
            train_loss = run_epoch(model, loaders["train"], criterion, optimizer, scheduler, scaler,
                                   device, amp, t.get("grad_clip"))
            if epoch % sched["eval_every"] and epoch != sched["epochs"]:
                continue
            val = validate(model, loaders["val"], criterion, device, amp)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val["loss"],
                            "val_macro_f1": val["macro_f1"], "val_accuracy": val["accuracy"],
                            "lr": optimizer.param_groups[0]["lr"], "epoch_seconds": time.perf_counter() - t0})
            if val["macro_f1"] > best["val_macro_f1"]:
                best = {"epoch": epoch, "val_macro_f1": val["macro_f1"]}
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            print(f"  época {epoch:3d}  pérdida {train_loss:.4f}  val F1 {val['macro_f1']:.4f}  "
                  f"(mejor {best['val_macro_f1']:.4f} @ {best['epoch']})", flush=True)
            # Paciencia contada después del warmup
            since = epoch - max(best["epoch"], sched["warmup_epochs"])
            if epoch > sched["warmup_epochs"] and since >= sched["patience"]:
                stopped_early = True
                print(f"  early stopping en la época {epoch}")
                break

    # Evaluación en prueba con el mejor checkpoint
    model.load_state_dict(best_state)
    with Timer() as test_timer:
        test_out = predict(model, loaders["test"], device, amp)
    test_metrics = classification_metrics(test_out["y_true"], test_out["y_pred"])
    timing_inf = inference_time_ms(model, device, cfg["data"]["image_size"], cfg["eval"]["inference_timing_iters"])

    plant_of = dict(zip(split_df.image, split_df.plant))
    predictions = [
        {"image": str(names[i]), "plant": plant_of[str(names[i])], "y_true": int(yt), "y_pred": int(yp),
         "prob": [round(float(p), 5) for p in pr]}
        for i, yt, yp, pr in zip(test_out["index"], test_out["y_true"], test_out["y_pred"], test_out["prob"])
    ]

    interp = None
    if cfg["interpret"]["enabled"]:
        examples = set(np.flatnonzero(np.isin(names, choose_examples(split_df, cfg["interpret"]["examples_per_class"]))))
        with Timer() as interp_timer:
            interp = run_interpretation(model, cfg["model"]["arch"], loaders["test"], cache["masks"], names,
                                        examples, device, cfg["interpret"]["map_size"])
        if interp["examples"]:
            maps_path = out_json.with_name(f"{name}_maps.npz")
            maps_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(maps_path, **{f"{img}|{meth}": arr for img, d in interp["examples"].items()
                                             for meth, arr in d.items()})
        interp = {k: v for k, v in interp.items() if k != "examples"}
        interp["seconds"] = interp_timer.seconds

    if save_checkpoint:
        ckpt = results_root / "checkpoints" / experiment / f"{name}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, ckpt)

    result = {
        "run": name, "experiment": experiment, "config_name": cfg["name"], "seed": seed,
        "fraction": fraction, "arch": cfg["model"]["arch"], "pretrained": bool(cfg["model"].get("pretrained")),
        "debug_limit": limit, "config": cfg, "schedule": sched, "params": params,
        "class_weights": [round(float(w), 5) for w in weights],
        "n_images": {k: int(len(v)) for k, v in idx.items()},
        "best": best, "stopped_early": stopped_early, "epochs_run": history[-1]["epoch"],
        "history": history,
        "timing": {"train_seconds": train_timer.seconds, "test_predict_seconds": test_timer.seconds,
                   "inference": timing_inf},
        "test": test_metrics, "predictions": predictions, "interpretation": interp,
        "code_version": git_commit(), "env": environment_info(),
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json_atomic(out_json, result)
    print(f"[{name}] prueba: F1 macro {test_metrics['macro_f1']:.4f}, exactitud {test_metrics['accuracy']:.4f} "
          f"| entrenamiento {train_timer.seconds / 60:.1f} min -> {out_json}")
    return out_json


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--experiment", required=True, help="E1, E2, E3 (o debug)")
    parser.add_argument("--data-root")
    parser.add_argument("--results-root")
    parser.add_argument("--override", action="append", default=[], help="clave.subclave=valor (repetible)")
    parser.add_argument("--limit", type=int, help="Solo pruebas: N imágenes de entrenamiento")
    parser.add_argument("--force", action="store_true", help="Repetir aunque el JSON exista")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.override)
    train_run(
        cfg, args.seed, args.fraction, args.experiment,
        resolve_dir(args.data_root, "ROCOLE_DATA_ROOT", "datos"),
        resolve_dir(args.results_root, "ROCOLE_RESULTS_ROOT", "resultados"),
        limit=args.limit, force=args.force, save_checkpoint=not args.no_checkpoint, device_name=args.device,
    )


if __name__ == "__main__":
    main()
