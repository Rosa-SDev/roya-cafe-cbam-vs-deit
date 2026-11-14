"""CBAM conserva la forma del tensor y agrega pocos parámetros."""
import pytest
import torch

from src.models.cbam import CBAM
from src.models.cnn import SimpleCNN
from src.models.factory import MBV2_CBAM_BLOCKS, build_model, cbam_modules, count_parameters


@pytest.mark.parametrize("mode", ["channel", "spatial", "both"])
@pytest.mark.parametrize("shape", [(2, 32, 56, 56), (1, 256, 7, 7), (3, 16, 9, 13)])
def test_cbam_preserves_shape(mode, shape):
    x = torch.randn(*shape)
    assert CBAM(shape[1], 16, mode)(x).shape == x.shape


def test_cbam_spatial_map_and_range():
    m = CBAM(64, 16, "both")
    m(torch.randn(2, 64, 14, 14))
    assert m.last_spatial_map.shape == (2, 1, 14, 14)
    assert 0 <= m.last_spatial_map.min() and m.last_spatial_map.max() <= 1


def test_cbam_param_count_matches_paper():
    # Canal: MLP compartido C -> C/r -> C con sesgos; espacial: conv 7x7 de 2 -> 1 canal sin sesgo.
    c, r = 256, 16
    expected = (c * (c // r) + c // r) + ((c // r) * c + c) + 2 * 7 * 7
    assert sum(p.numel() for p in CBAM(c, r, "both").parameters()) == expected


def test_invalid_mode():
    with pytest.raises(ValueError):
        CBAM(32, 16, "nada")


def test_cnn_params_and_cbam_overhead():
    base = count_parameters(SimpleCNN())
    cbam = count_parameters(SimpleCNN(cbam_mode="both"))
    assert 1.0e6 < base["total"] < 1.5e6
    assert cbam["cbam"] == cbam["total"] - base["total"]
    assert cbam["cbam"] / base["total"] < 0.02  # aumento mínimo de parámetros (H1)


@pytest.mark.parametrize("blocks,expected", [(None, 4), ([2, 3], 2)])
def test_cnn_cbam_blocks(blocks, expected):
    model = SimpleCNN(cbam_mode="both", cbam_blocks=blocks)
    assert len(cbam_modules(model)) == expected
    assert model(torch.randn(2, 3, 224, 224)).shape == (2, 6)


def test_mbv2_cbam_in_residual_branch():
    cfg = {"model": {"arch": "mbv2", "pretrained": False,
                     "cbam": {"mode": "both", "blocks": list(MBV2_CBAM_BLOCKS)}}}
    model = build_model(cfg).eval()
    for idx in MBV2_CBAM_BLOCKS:
        block = model.features[idx]
        assert block.use_res_connect
        assert isinstance(block.conv[-1], CBAM)
    assert len(cbam_modules(model)) == len(MBV2_CBAM_BLOCKS)
    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 6)


def test_deit_builds():
    model = build_model({"model": {"arch": "deit", "pretrained": False}})
    assert model(torch.randn(1, 3, 224, 224)).shape == (1, 6)
