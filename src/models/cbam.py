"""CBAM: Convolutional Block Attention Module (Woo et al., ECCV 2018).

Implementación propia y fiel al artículo:
- Atención de canal: un MLP compartido (C -> C/r -> C, ReLU en medio) se
  aplica al average-pool y al max-pool espaciales; se suman y pasan por una
  sigmoide. Mc(F) = sigma(MLP(AvgPool(F)) + MLP(MaxPool(F))).
- Atención espacial: se concatenan la media y el máximo a lo largo de los
  canales (2 mapas), conv 7x7 con padding 3 y sigmoide.
  Ms(F) = sigma(conv7x7([AvgPool_c(F); MaxPool_c(F)])).
- Orden secuencial: primero canal y luego espacial (el mejor orden del artículo).

`mode` permite la ablación: 'channel', 'spatial' o 'both'.
El último mapa espacial queda en `self.last_spatial_map` para interpretar (H4).
"""
from __future__ import annotations

import torch
from torch import nn

VALID_MODES = ("channel", "spatial", "both")


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        # Las convoluciones 1x1 equivalen a capas lineales sobre el vector (C,1,1).
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        return torch.sigmoid(self.mlp(avg) + self.mlp(mx))  # (B, C, 1, 1)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))  # (B, 1, H, W)


class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, mode: str = "both", kernel_size: int = 7):
        super().__init__()
        if mode not in VALID_MODES:
            raise ValueError(f"mode debe ser uno de {VALID_MODES}, no {mode!r}")
        self.mode = mode
        self.channel = ChannelAttention(channels, reduction) if mode in ("channel", "both") else None
        self.spatial = SpatialAttention(kernel_size) if mode in ("spatial", "both") else None
        self.last_spatial_map: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channel is not None:
            x = x * self.channel(x)
        if self.spatial is not None:
            attn = self.spatial(x)
            self.last_spatial_map = attn.detach()
            x = x * attn
        return x
