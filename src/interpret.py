"""Interpretabilidad (H4): mapas de atención y fracción dentro de la hoja.

Mapas por familia:
- gradcam:    Grad-CAM (Selvaraju et al., 2017) sobre la última capa
              convolucional de las CNN (CNN propia: bloque 4, 28x28;
              MobileNetV2: features[18], 7x7). Clase objetivo = la predicha.
- cbam_last:  mapa de atención espacial del CBAM más profundo.
- cbam_mean:  promedio de los mapas espaciales de todos los CBAM (cada uno
              reescalado a 224 y normalizado).
- rollout:    attention rollout (Abnar y Zuidema, 2020) para DeiT: se
              promedian las cabezas, se suma la identidad (conexión residual),
              se renormalizan las filas y se multiplican las capas; se toma la
              fila del token CLS hacia los 196 parches (14x14).

Métrica: cada mapa se normaliza a [0, 1] (min-max), se lleva a 224x224 y
    fracción_en_hoja = suma(mapa * máscara) / suma(mapa).
Referencia: un mapa uniforme daría exactamente el área relativa de la hoja,
que también se guarda (leaf_area) para comparar contra el azar.

Advertencia: Grad-CAM, CBAM y rollout son mapas de naturaleza distinta; la
comparación entre familias es indicativa, no estrictamente equivalente.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .models.factory import cbam_modules, gradcam_layer


def _normalize(maps: torch.Tensor) -> torch.Tensor:
    """Min-max por imagen; maps (B, H, W) -> [0, 1]."""
    flat = maps.flatten(1)
    lo = flat.min(1, keepdim=True).values
    hi = flat.max(1, keepdim=True).values
    out = (flat - lo) / (hi - lo).clamp_min(1e-12)
    return out.view_as(maps)


def _to_size(maps: torch.Tensor, size: int) -> torch.Tensor:
    return F.interpolate(maps.unsqueeze(1).float(), size=(size, size), mode="bilinear",
                         align_corners=False).squeeze(1)


def fraction_in_mask(maps: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """maps y masks (B, H, W) del mismo tamaño; maps ya normalizados."""
    total = maps.flatten(1).sum(1)
    inside = (maps * masks).flatten(1).sum(1)
    return torch.where(total > 0, inside / total.clamp_min(1e-12), torch.full_like(total, float("nan")))


class GradCAM:
    def __init__(self, model, layer):
        self.model = model
        self.acts = None
        self.grads = None
        self._h = layer.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self.acts = out
        out.register_hook(lambda g: setattr(self, "grads", g))

    def __call__(self, x: torch.Tensor):
        """Devuelve (logits, cam (B, h, w)) para la clase predicha."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        target = logits.argmax(1)
        logits.gather(1, target[:, None]).sum().backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.acts).sum(1))
        return logits.detach(), cam.detach()

    def remove(self):
        self._h.remove()


class AttentionRecorder:
    """Captura las matrices de atención de cada bloque de DeiT.

    timm usa por defecto la atención "fused" (scaled_dot_product_attention),
    que no expone las matrices; se desactiva y se engancha attn_drop, cuya
    salida (en modo eval) es exactamente la matriz softmax(QK^T / sqrt(d)).
    """

    def __init__(self, model):
        self.maps = []
        self.handles = []
        for blk in model.blocks:
            blk.attn.fused_attn = False
            self.handles.append(blk.attn.attn_drop.register_forward_hook(
                lambda m, i, o: self.maps.append(o.detach())))

    def rollout(self) -> torch.Tensor:
        """(B, 14, 14) a partir de las atenciones capturadas en el último forward."""
        result = None
        for attn in self.maps:
            a = attn.mean(1)  # promedio de cabezas: (B, T, T)
            a = a + torch.eye(a.size(-1), device=a.device)
            a = a / a.sum(-1, keepdim=True)
            result = a if result is None else a @ result
        cls_to_patches = result[:, 0, 1:]
        side = int(cls_to_patches.size(1) ** 0.5)
        self.maps = []
        return cls_to_patches.reshape(-1, side, side)

    def remove(self):
        for h in self.handles:
            h.remove()


def compute_maps(model, arch: str, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Calcula todos los mapas aplicables, cada uno (B, 224, 224) en [0, 1]."""
    size = x.shape[-1]
    maps = {}
    if arch == "deit":
        rec = AttentionRecorder(model)
        try:
            with torch.no_grad():
                logits = model(x)
            maps["rollout"] = _normalize(_to_size(rec.rollout(), size))
        finally:
            rec.remove()
        return logits, maps

    cam = GradCAM(model, gradcam_layer(model, arch))
    try:
        with torch.enable_grad():
            logits, raw = cam(x)
    finally:
        cam.remove()
    maps["gradcam"] = _normalize(_to_size(raw, size))
    spatial = [m.last_spatial_map for m in cbam_modules(model) if m.last_spatial_map is not None]
    if spatial:
        ups = [_normalize(_to_size(s[:, 0], size)) for s in spatial]
        maps["cbam_last"] = ups[-1]
        maps["cbam_mean"] = _normalize(torch.stack(ups).mean(0))
    return logits, maps


def run_interpretation(model, arch: str, loader, masks_256: np.ndarray, names: np.ndarray,
                       example_idx: set[int], device, map_size: int = 56) -> dict:
    """Recorre el conjunto (de prueba) y calcula la fracción de atención en la hoja.

    Devuelve un resumen, los valores por imagen y los mapas de las imágenes de
    ejemplo (reducidos a map_size x map_size, float16) para las figuras.
    """
    model.eval()
    per_image = []
    examples = {}
    for x, y, idx in loader:
        x = x.to(device)
        logits, maps = compute_maps(model, arch, x)
        size = x.shape[-1]
        m = torch.from_numpy(masks_256[idx.numpy()]).float().to(device)
        m = F.interpolate(m.unsqueeze(1), size=(size, size), mode="nearest").squeeze(1)
        fracs = {k: fraction_in_mask(v, m).cpu().numpy() for k, v in maps.items()}
        leaf_area = m.flatten(1).mean(1).cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
        for j, i in enumerate(idx.numpy()):
            row = {"image": str(names[i]), "leaf_area": float(leaf_area[j]),
                   "y_true": int(y[j]), "y_pred": int(pred[j])}
            row.update({f"in_leaf_{k}": float(v[j]) for k, v in fracs.items()})
            per_image.append(row)
            if int(i) in example_idx:
                small = {k: _to_size(v[j:j + 1], map_size)[0].cpu().numpy().astype(np.float16)
                         for k, v in maps.items()}
                examples[str(names[i])] = small
    methods = sorted({k[len("in_leaf_"):] for r in per_image for k in r if k.startswith("in_leaf_")})
    summary = {"leaf_area_mean": float(np.mean([r["leaf_area"] for r in per_image]))}
    for meth in methods:
        vals = np.array([r[f"in_leaf_{meth}"] for r in per_image], dtype=float)
        summary[f"in_leaf_{meth}_mean"] = float(np.nanmean(vals))
        summary[f"in_leaf_{meth}_std"] = float(np.nanstd(vals))
    return {"summary": summary, "methods": methods, "per_image": per_image, "examples": examples}


def choose_examples(split_df, per_class: int = 2) -> list[str]:
    """Imágenes de ejemplo fijas del conjunto de prueba: las primeras `per_class`
    de cada clase en orden alfabético (las mismas para todas las corridas)."""
    test = split_df[split_df.split == "test"].sort_values("image")
    return test.groupby("label").head(per_class).image.tolist()
