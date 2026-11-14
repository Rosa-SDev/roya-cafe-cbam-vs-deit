"""CNN propia pequeña (~1,17 M de parámetros), con CBAM opcional.

Arquitectura: 4 bloques [Conv3x3 -> BN -> ReLU] x2, con canales 32, 64, 128 y
256; CBAM opcional al final de cada bloque (antes del MaxPool, para que el mapa
espacial tenga más resolución); MaxPool 2x2; Global Average Pooling; Dropout
0,3; capa lineal.

Con entrada 224: los bloques trabajan a 224, 112, 56 y 28 píxeles. Grad-CAM se
calcula sobre la salida del bloque 4 (28x28).
"""
from __future__ import annotations

from torch import nn

from .cbam import CBAM


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cbam: CBAM | None = None):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam = cbam

    def forward(self, x):
        x = self.convs(x)
        if self.cbam is not None:
            x = self.cbam(x)
        return x


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 6, channels=(32, 64, 128, 256), dropout: float = 0.3,
                 cbam_mode: str | None = None, cbam_blocks=None, cbam_reduction: int = 16):
        """`cbam_blocks`: índices (desde 0) de los bloques que llevan CBAM.
        None con cbam_mode definido significa todos los bloques."""
        super().__init__()
        if cbam_mode is not None and cbam_blocks is None:
            cbam_blocks = list(range(len(channels)))
        cbam_blocks = set(cbam_blocks or [])
        blocks = []
        in_ch = 3
        for i, out_ch in enumerate(channels):
            cbam = CBAM(out_ch, cbam_reduction, cbam_mode) if i in cbam_blocks else None
            blocks.append(ConvBlock(in_ch, out_ch, cbam))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)
        self.pool = nn.MaxPool2d(2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(in_ch, num_classes)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        for block in self.blocks:
            x = self.pool(block(x))
        return self.head(x)

    @property
    def gradcam_layer(self) -> nn.Module:
        """Capa objetivo de Grad-CAM: salida del último bloque (antes del pool)."""
        return self.blocks[-1]
