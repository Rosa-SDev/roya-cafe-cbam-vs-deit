"""Construcción de modelos a partir de la configuración YAML.

Arquitecturas (`model.arch`):
- cnn:   CNN propia (src/models/cnn.py), con o sin CBAM.
- mbv2:  MobileNetV2 de torchvision, con o sin CBAM, desde cero o con pesos
         de ImageNet (IMAGENET1K_V1).
- deit:  deit_tiny_patch16_224 de timm, desde cero o con pesos de ImageNet.

MobileNetV2 + CBAM: CBAM va DENTRO de la rama residual (antes de la suma) del
último bloque de las etapas de 32, 64, 96 y 160 canales: features[6, 10, 13,
16]. Así lo integran Woo et al. en ResNet. El camino identidad sigue llevando
las características preentrenadas intactas, de modo que el CBAM recién
inicializado (aleatorio) no destruye el preentrenamiento al empezar. Se omite
la etapa de 24 canales porque con r = 16 su MLP tendría 1 sola neurona.
"""
from __future__ import annotations

import timm
import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.mobilenetv2 import InvertedResidual

from .cbam import CBAM
from .cnn import SimpleCNN

MBV2_CBAM_BLOCKS = (6, 10, 13, 16)


def build_model(cfg: dict, num_classes: int = 6) -> nn.Module:
    m = cfg["model"]
    arch = m["arch"]
    cbam_cfg = m.get("cbam") or {}
    cbam_mode = cbam_cfg.get("mode")
    reduction = cbam_cfg.get("reduction", 16)

    if arch == "cnn":
        model = SimpleCNN(
            num_classes=num_classes,
            channels=tuple(m.get("channels", (32, 64, 128, 256))),
            dropout=m.get("dropout", 0.3),
            cbam_mode=cbam_mode,
            cbam_blocks=cbam_cfg.get("blocks"),
            cbam_reduction=reduction,
        )
    elif arch == "mbv2":
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if m.get("pretrained") else None
        model = mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        if cbam_mode:
            for idx in cbam_cfg.get("blocks") or MBV2_CBAM_BLOCKS:
                block = model.features[idx]
                if not (isinstance(block, InvertedResidual) and block.use_res_connect):
                    raise ValueError(f"features[{idx}] no es un bloque residual")
                # InvertedResidual.forward hace x + self.conv(x): envolver conv
                # coloca CBAM en la rama residual, antes de la suma.
                block.conv = nn.Sequential(block.conv, CBAM(block.out_channels, reduction, cbam_mode))
    elif arch == "deit":
        model = timm.create_model(
            "deit_tiny_patch16_224",
            pretrained=bool(m.get("pretrained")),
            num_classes=num_classes,
            drop_path_rate=m.get("drop_path_rate", 0.1),
        )
    else:
        raise ValueError(f"Arquitectura desconocida: {arch}")
    return model


def gradcam_layer(model: nn.Module, arch: str) -> nn.Module | None:
    """Última capa convolucional para Grad-CAM (None para DeiT)."""
    if arch == "cnn":
        return model.gradcam_layer
    if arch == "mbv2":
        return model.features[-1]  # Conv 1x1 a 1280 canales, 7x7
    return None


def cbam_modules(model: nn.Module) -> list[CBAM]:
    return [mod for mod in model.modules() if isinstance(mod, CBAM)]


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cbam = sum(p.numel() for mod in cbam_modules(model) for p in mod.parameters())
    return {"total": total, "trainable": trainable, "cbam": cbam}


def param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Sin weight decay en sesgos, normalizaciones, tokens y embeddings de posición.

    Regla práctica: tensores con ndim <= 1 (sesgos, BN/LN) y los parámetros
    especiales de DeiT (cls_token, pos_embed) no se decaen.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith((".bias", "cls_token", "pos_embed", "dist_token")):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


@torch.no_grad()
def describe(cfg: dict) -> dict:
    """Parámetros de una configuración sin descargar pesos."""
    cfg = {**cfg, "model": {**cfg["model"], "pretrained": False}}
    return count_parameters(build_model(cfg))
