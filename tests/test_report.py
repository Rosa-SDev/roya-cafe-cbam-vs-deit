"""report.py genera todas las tablas y figuras de 02_resultados con corridas sintéticas."""
import numpy as np

from src import report
from src.aggregate import aggregate, load_runs
from src.evaluate import classification_metrics
from src.runs import all_runs
from src.utils import CLASSES, write_json_atomic


def _fake(root, r, rng):
    images = [f"C1P{p}{v}.jpg" for p in range(1, 13) for v in ["H1", "H2", "E1", "E2"]]
    y_true = np.arange(len(images)) % len(CLASSES)
    y_pred = np.where(rng.random(len(images)) < 0.7, y_true, rng.integers(0, 6, len(images)))
    arch = r["config"].split("_")[0]
    methods = ["rollout"] if arch == "deit" else (["gradcam", "cbam_last"] if "cbam" in r["config"] else ["gradcam"])
    name = r["name"]
    write_json_atomic(root / "runs" / r["experiment"] / f"{name}.json", {
        "run": name, "experiment": r["experiment"], "config_name": r["config"], "seed": r["seed"],
        "fraction": r["fraction"], "arch": arch, "pretrained": r["config"].endswith("pre"), "debug_limit": None,
        "params": {"total": 1_000_000, "trainable": 1_000_000, "cbam": 1000 if "cbam" in r["config"] else 0},
        "best": {"epoch": 3, "val_macro_f1": 0.5}, "epochs_run": 10,
        "timing": {"train_seconds": 300.0, "inference": {"median_ms": 2.0, "device": "cuda"}},
        "test": classification_metrics(y_true, y_pred),
        "predictions": [{"image": im, "plant": im[:-6], "y_true": int(t), "y_pred": int(p)}
                        for im, t, p in zip(images, y_true, y_pred)],
        "interpretation": {
            "methods": methods,
            "summary": {"leaf_area_mean": 0.3, **{f"in_leaf_{m}_mean": 0.5 for m in methods},
                        **{f"in_leaf_{m}_std": 0.1 for m in methods}},
            "per_image": [{"image": im, "leaf_area": 0.3, **{f"in_leaf_{m}": float(rng.random()) for m in methods}}
                          for im in images]},
        "code_version": {"commit": "a" * 40}, "env": {"device": "cuda", "gpu": "Tesla T4"},
    })
    if r["experiment"] == "E1":
        np.savez_compressed(root / "runs" / "E1" / f"{name}_maps.npz",
                            **{f"{im}|{m}": rng.random((56, 56)).astype(np.float16)
                               for im in images[:6] for m in methods})
    return images


def test_full_report(tmp_path):
    rng = np.random.default_rng(0)
    root = tmp_path / "res"
    runs = [r for r in all_runs() if r["seed"] <= 1]  # 2 semillas bastan para la prueba
    for r in runs:
        images = _fake(root, r, rng)
    tables = aggregate(root, tmp_path / "agg", reps=50)
    summary, boot = tables["summary"], tables["bootstrap"]
    out = tmp_path / "fig"
    report.setup_style()
    for name, df in [("tabla_e1", report.table_e1(summary, boot)), ("tabla_e2", report.table_e2(summary, boot)),
                     ("tabla_parametros", report.table_params_times(summary))]:
        assert len(df) > 0
        report.save_table(df, out, name, "Prueba", f"tab:{name}")
        assert "\\toprule" in (out / f"{name}.tex").read_text(encoding="utf-8")
    loaded = load_runs(root)
    report.fig_f1_bars(summary, out)
    report.fig_data_curves(summary, out)
    report.fig_confusions(loaded, summary, out)
    report.fig_recall_per_class(summary, out)
    report.fig_in_leaf(summary, out)
    imgs = rng.integers(0, 255, (len(images), 256, 256, 3), dtype=np.uint8)
    report.fig_attention_examples(root, loaded, imgs, np.array(images), out)
    for fig in ["f1_macro_por_configuracion", "curvas_datos_f1", "matrices_confusion_mejor_por_familia",
                "sensibilidad_por_clase", "fraccion_atencion_hoja", "ejemplos_mapas_atencion"]:
        assert (out / f"{fig}.pdf").exists() and (out / f"{fig}.png").exists()
