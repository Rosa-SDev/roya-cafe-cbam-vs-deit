"""Tablas (CSV + LaTeX booktabs) y figuras (PDF + PNG 300 dpi) para el artículo.

Lo usa notebooks/02_resultados.ipynb. Todo se calcula a partir de las tablas
de aggregate.py y de los JSON/NPZ de las corridas; nada se escribe a mano.

Tamaños IEEE: una columna = 3,5 in; doble columna = 7,16 in.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .utils import CLASSES  # noqa: E402

COL_W, DBL_W = 3.5, 7.16
MAIN_ORDER = ["cnn_scratch", "cnn_cbam_scratch", "deit_scratch", "mbv2_pre", "mbv2_cbam_pre", "deit_pre"]
LABELS = {
    "cnn_scratch": "CNN", "cnn_cbam_scratch": "CNN+CBAM", "deit_scratch": "DeiT-Ti",
    "mbv2_pre": "MBV2", "mbv2_cbam_pre": "MBV2+CBAM", "deit_pre": "DeiT-Ti",
    "cnn_cbam_channel_scratch": "CBAM canal", "cnn_cbam_spatial_scratch": "CBAM espacial",
    "cnn_cbam_last2_scratch": "CBAM bloques 3-4",
}
FAMILY = {"cnn_scratch": "CNN base", "mbv2_pre": "CNN base", "cnn_cbam_scratch": "CNN + CBAM",
          "mbv2_cbam_pre": "CNN + CBAM", "deit_scratch": "Transformer", "deit_pre": "Transformer"}
FAMILY_COLOR = {"CNN base": "#4C72B0", "CNN + CBAM": "#DD8452", "Transformer": "#55A868"}
CLASS_SHORT = ["Sana", "Ácaro", "R1", "R2", "R3", "R4"]
METHOD_LABEL = {"gradcam": "Grad-CAM", "cbam_last": "CBAM espacial", "cbam_mean": "CBAM (media)",
                "rollout": "Rollout"}


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "serif",
        "savefig.dpi": 300, "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    })


def pretty(config: str) -> str:
    base = LABELS.get(config, config)
    if config in MAIN_ORDER:
        base += " (pre)" if config.endswith("_pre") else " (cero)"
    return base


def save_fig(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------
def pm(mean, std, digits: int = 3) -> str:
    if pd.isna(mean):
        return "--"
    return f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def to_latex_booktabs(df: pd.DataFrame, caption: str, label: str, align: str | None = None) -> str:
    def esc(x):
        s = str(x)
        return s if "$" in s else s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    align = align or "l" + "c" * (df.shape[1] - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{" + align + "}", r"\toprule",
             " & ".join(esc(c) for c in df.columns) + r" \\", r"\midrule"]
    lines += [" & ".join(esc(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, out_dir: Path, name: str, caption: str, label: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / f"{name}.tex").write_text(to_latex_booktabs(df, caption, label), encoding="utf-8")


def _ci(boot: pd.DataFrame, exp, config, frac, metric) -> str:
    sel = boot[(boot.experiment == exp) & (boot.config == config) & (boot.fraction == frac) & (boot.metric == metric)]
    if sel.empty:
        return "--"
    r = sel.iloc[0]
    return f"[{r.ci_low:.3f}, {r.ci_high:.3f}]"


def table_e1(summary: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    s = summary[(summary.experiment == "E1") & (summary.fraction == 1.0)].set_index("config")
    rows = []
    for c in MAIN_ORDER:
        if c not in s.index:
            continue
        r = s.loc[c]
        rows.append({
            "Modelo": pretty(c), "Parámetros (M)": f"{r.params_total_mean / 1e6:.2f}",
            "Exactitud": pm(r.accuracy_mean, r.accuracy_std), "F1 macro": pm(r.macro_f1_mean, r.macro_f1_std),
            "IC95 F1 macro": _ci(boot, "E1", c, 1.0, "macro_f1"),
            "F1 sana/enferma": pm(r.binary_f1_mean, r.binary_f1_std), "n": int(r.n_seeds),
        })
    return pd.DataFrame(rows)


def table_e2(summary: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    order = [("E1", "cnn_scratch", "Sin CBAM"), ("E1", "cnn_cbam_scratch", "CBAM canal+espacial (4 bloques)"),
             ("E2", "cnn_cbam_channel_scratch", "Solo canal (4 bloques)"),
             ("E2", "cnn_cbam_spatial_scratch", "Solo espacial (4 bloques)"),
             ("E2", "cnn_cbam_last2_scratch", "Canal+espacial (bloques 3-4)")]
    rows = []
    for exp, c, name in order:
        sel = summary[(summary.experiment == exp) & (summary.config == c) & (summary.fraction == 1.0)]
        if sel.empty:
            continue
        r = sel.iloc[0]
        rows.append({"Variante": name, "Parámetros CBAM": f"{int(r.params_cbam_mean):,}",
                     "Exactitud": pm(r.accuracy_mean, r.accuracy_std), "F1 macro": pm(r.macro_f1_mean, r.macro_f1_std),
                     "IC95 F1 macro": _ci(boot, exp, c, 1.0, "macro_f1"), "n": int(r.n_seeds)})
    return pd.DataFrame(rows)


TIMING_GPU = "RTX 3050"        # los tiempos del artículo se miden solo en esta GPU
TIMING_SEEDS = [0, 1, 2, 3]   # E1 semillas locales (la 4 y E2 corren en T4)


def table_train_times(runs_df: pd.DataFrame, gpu: str = TIMING_GPU, seeds=TIMING_SEEDS) -> pd.DataFrame:
    """Parámetros y tiempos de entrenamiento: solo E1, solo la GPU indicada.

    Tiempo total = entrenamiento completo con early stopping (sin prueba ni
    interpretabilidad). Época estable = mediana de s/época sin la época 1
    (entrenamiento + validación). Falla si entra una semilla distinta de las
    esperadas, para no mezclar GPU por accidente.
    """
    sel = runs_df[(runs_df.experiment == "E1") & (runs_df.fraction == 1.0)
                  & runs_df.gpu.astype(str).str.contains(gpu, regex=False)]
    extra = sorted(set(sel.seed) - set(seeds))
    if extra:
        raise ValueError(f"Semillas inesperadas en la tabla de tiempos ({gpu}): {extra}")
    rows = []
    for c in MAIN_ORDER:
        d = sel[sel.config == c]
        if d.empty:
            continue
        rows.append({
            "Modelo": pretty(c), "Parámetros": f"{int(d.params_total.iloc[0]):,}",
            "Parámetros CBAM": f"{int(d.params_cbam.iloc[0]):,}",
            "Entrenamiento (min)": pm(d.train_minutes.mean(), d.train_minutes.std(), 1),
            "Época estable (s)": pm(d.epoch_seconds_stable.mean(), d.epoch_seconds_stable.std(), 1),
            "Épocas": pm(d.epochs_run.mean(), d.epochs_run.std(), 0),
            "Semillas": ",".join(str(s) for s in sorted(d.seed)),
        })
    return pd.DataFrame(rows)


def table_inference(bench: pd.DataFrame) -> pd.DataFrame:
    """Tabla de inferencia a partir del CSV de src/bench_inference.py."""
    rows = []
    for c in MAIN_ORDER:
        d = bench[bench.config == c]
        if d.empty:
            continue
        r = d.iloc[0]
        rows.append({
            "Modelo": pretty(c),
            "Latencia FP32 (ms)": f"{r.latency_fp32_median_ms:.2f}",
            "Throughput FP32 (img/s)": f"{r.throughput_fp32_img_s:.0f}",
            "Latencia FP16 (ms)": f"{r.latency_fp16_median_ms:.2f}",
            "Throughput FP16 (img/s)": f"{r.throughput_fp16_img_s:.0f}",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_f1_bars(summary: pd.DataFrame, out_dir: Path) -> None:
    s = summary[(summary.experiment == "E1") & (summary.fraction == 1.0)].set_index("config")
    configs = [c for c in MAIN_ORDER if c in s.index]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    x = np.arange(len(configs)) + np.array([0 if c.endswith("scratch") else 0.6 for c in configs])
    for xi, c in zip(x, configs):
        ax.bar(xi, s.loc[c, "macro_f1_mean"], yerr=s.loc[c, "macro_f1_std"], color=FAMILY_COLOR[FAMILY[c]],
               capsize=2, width=0.8, edgecolor="black", linewidth=0.4,
               hatch="" if c.endswith("scratch") else "///")
    ax.set_xticks(x, [LABELS[c] for c in configs], rotation=30, ha="right")
    ax.set_ylabel("F1 macro (prueba)")
    ax.set_ylim(0, 1)
    ax.text(np.mean(x[:3]) if len(x) >= 3 else 0, 0.95, "Desde cero", ha="center", fontsize=7)
    if len(x) > 3:
        ax.text(np.mean(x[3:]), 0.95, "Preentrenado", ha="center", fontsize=7)
    save_fig(fig, out_dir, "f1_macro_por_configuracion")


def fig_data_curves(summary: pd.DataFrame, out_dir: Path) -> None:
    curve = summary[((summary.experiment == "E1") & (summary.fraction == 1.0)) | (summary.experiment == "E3")]
    fig, axes = plt.subplots(1, 2, figsize=(COL_W, 1.9), sharey=True)
    for ax, kind, title in [(axes[0], "scratch", "Desde cero"), (axes[1], "pre", "Preentrenado")]:
        for c in [c for c in MAIN_ORDER if c.endswith(kind)]:
            d = curve[curve.config == c].sort_values("fraction")
            if d.empty:
                continue
            ax.errorbar(d.fraction * 100, d.macro_f1_mean, yerr=d.macro_f1_std, marker="o", ms=3, capsize=2,
                        color=FAMILY_COLOR[FAMILY[c]], label=LABELS[c], lw=1)
        ax.set_title(title)
        ax.set_xticks([25, 50, 100])
        ax.set_xlabel("% del entrenamiento")
        ax.legend(frameon=False)
    axes[0].set_ylabel("F1 macro (prueba)")
    save_fig(fig, out_dir, "curvas_datos_f1")


def best_per_family(summary: pd.DataFrame) -> dict[str, str]:
    s = summary[(summary.experiment == "E1") & (summary.fraction == 1.0)]
    out = {}
    for fam in FAMILY_COLOR:
        cands = s[s.config.map(FAMILY.get) == fam]
        if not cands.empty:
            out[fam] = cands.sort_values("macro_f1_mean").iloc[-1].config
    return out


def fig_confusions(runs: list[dict], summary: pd.DataFrame, out_dir: Path) -> None:
    best = best_per_family(summary)
    fig, axes = plt.subplots(1, len(best), figsize=(DBL_W, 2.4))
    axes = np.atleast_1d(axes)
    for ax, (fam, config) in zip(axes, best.items()):
        cms = [np.array(r["test"]["confusion_matrix"]) for r in runs
               if r["experiment"] == "E1" and r["config_name"] == config]
        cm = np.sum(cms, axis=0).astype(float)
        cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > 0.6 else "black")
        ax.set_xticks(range(len(CLASSES)), CLASS_SHORT)
        ax.set_yticks(range(len(CLASSES)), CLASS_SHORT)
        ax.set_title(f"{fam}: {pretty(config)}")
        ax.set_xlabel("Predicha")
    axes[0].set_ylabel("Real")
    save_fig(fig, out_dir, "matrices_confusion_mejor_por_familia")


def fig_recall_per_class(summary: pd.DataFrame, out_dir: Path) -> None:
    s = summary[(summary.experiment == "E1") & (summary.fraction == 1.0)].set_index("config")
    configs = [c for c in MAIN_ORDER if c in s.index]
    fig, ax = plt.subplots(figsize=(DBL_W, 2.0))
    width = 0.8 / max(1, len(configs))
    for k, c in enumerate(configs):
        means = [s.loc[c, f"recall_{cl}_mean"] for cl in CLASSES]
        stds = [s.loc[c, f"recall_{cl}_std"] for cl in CLASSES]
        ax.bar(np.arange(len(CLASSES)) + k * width, means, width, yerr=stds, capsize=1.5, label=pretty(c),
               color=FAMILY_COLOR[FAMILY[c]], alpha=1.0 if c.endswith("scratch") else 0.55,
               edgecolor="black", linewidth=0.3)
    ax.set_xticks(np.arange(len(CLASSES)) + 0.4 - width / 2, CLASS_SHORT)
    ax.set_ylabel("Sensibilidad (prueba)")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    save_fig(fig, out_dir, "sensibilidad_por_clase")


def fig_in_leaf(summary: pd.DataFrame, out_dir: Path) -> None:
    s = summary[(summary.experiment == "E1") & (summary.fraction == 1.0)].set_index("config")
    configs = [c for c in MAIN_ORDER if c in s.index]
    methods = ["gradcam", "cbam_last", "rollout"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    xt = []
    pos = 0
    for c in configs:
        for m in methods:
            col = f"in_leaf_{m}_mean"
            if col in s and not pd.isna(s.loc[c, col]):
                ax.bar(pos, s.loc[c, col], yerr=s.loc[c, f"in_leaf_{m}_std"], capsize=2,
                       color=FAMILY_COLOR[FAMILY[c]], hatch={"gradcam": "", "cbam_last": "...", "rollout": "xx"}[m],
                       edgecolor="black", linewidth=0.4)
                xt.append((pos, f"{LABELS[c]}{' (pre)' if c.endswith('pre') else ''}\n{METHOD_LABEL[m]}"))
                pos += 1
        pos += 0.4
    if "leaf_area_mean" in s:
        ax.axhline(s["leaf_area_mean"].mean(), ls="--", lw=0.8, color="gray", label="Área de la hoja (azar)")
        ax.legend(frameon=False, loc="upper right")
    ax.set_xticks([p for p, _ in xt], [t for _, t in xt], rotation=60, ha="right", fontsize=5.5)
    ax.set_ylabel("Fracción de atención en la hoja")
    ax.set_ylim(0, 1)
    save_fig(fig, out_dir, "fraccion_atencion_hoja")


def fig_attention_examples(results_root: Path, runs: list[dict], images: np.ndarray, names: np.ndarray,
                           out_dir: Path, seed: int = 0,
                           panels=(("cnn_scratch", "gradcam"), ("cnn_cbam_scratch", "gradcam"),
                                   ("cnn_cbam_scratch", "cbam_last"), ("mbv2_cbam_pre", "gradcam"),
                                   ("deit_scratch", "rollout"), ("deit_pre", "rollout"))) -> None:
    """Una fila por clase (la primera imagen de ejemplo); columnas: foto y mapas.

    Título verde = acierto, rojo = error de ese modelo en esa imagen.
    """
    preds, maps = {}, {}
    for r in runs:
        if r["experiment"] == "E1" and r["seed"] == seed:
            preds[r["config_name"]] = {p["image"]: (p["y_true"], p["y_pred"]) for p in r["predictions"]}
            path = results_root / "runs" / "E1" / f"{r['run']}_maps.npz"
            if path.exists():
                with np.load(path) as z:
                    maps[r["config_name"]] = {k: z[k] for k in z.files}
    panels = [p for p in panels if p[0] in maps]
    if not panels:
        print("No hay mapas de ejemplo para la figura.")
        return
    example_imgs = sorted({k.split("|")[0] for k in next(iter(maps.values()))})
    by_class = {}
    for img in example_imgs:
        yt = next(iter(preds.values()))[img][0]
        by_class.setdefault(yt, img)
    rows = [by_class[k] for k in sorted(by_class)]
    pos = {n: i for i, n in enumerate(names)}
    fig, axes = plt.subplots(len(rows), len(panels) + 1, figsize=(DBL_W, 1.05 * len(rows)))
    for i, img in enumerate(rows):
        photo = images[pos[img]]
        axes[i, 0].imshow(photo)
        axes[i, 0].set_ylabel(CLASS_SHORT[preds[panels[0][0]][img][0]], rotation=0, labelpad=14, va="center")
        for j, (config, meth) in enumerate(panels, start=1):
            ax = axes[i, j]
            ax.imshow(photo)
            key = f"{img}|{meth}"
            if key in maps[config]:
                m = maps[config][key].astype(float)
                ax.imshow(m, cmap="jet", alpha=0.45, interpolation="bilinear",
                          extent=(0, photo.shape[1], photo.shape[0], 0))
            yt, yp = preds[config][img]
            ax.set_title(f"pred: {CLASS_SHORT[yp]}", color="green" if yt == yp else "red", fontsize=6, pad=1)
            if i == 0:
                origin = "pre" if config.endswith("_pre") else "cero"
                ax.text(0.5, 1.25, f"{LABELS[config]} ({origin})\n{METHOD_LABEL[meth]}", transform=ax.transAxes,
                        ha="center", va="bottom", fontsize=5.5)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[0, 0].text(0.5, 1.25, "Imagen", transform=axes[0, 0].transAxes, ha="center", va="bottom", fontsize=6)
    save_fig(fig, out_dir, "ejemplos_mapas_atencion")
