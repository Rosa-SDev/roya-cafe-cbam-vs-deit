"""Nombres de archivo, máscaras y la regla de no aplicar la rotación EXIF."""
import pytest
from PIL import Image

from src.data import open_raw_image, parse_filename, polygon_to_mask


def test_parse_filename():
    assert parse_filename("C10P3E2.jpg") == {"lot": 10, "plant": "C10P3", "view": "E2"}


def test_known_anomaly():
    assert parse_filename("C11P11HE.jpg") == {"lot": 11, "plant": "C11P11", "view": "H2"}


@pytest.mark.parametrize("bad", ["foto.jpg", "C1P2X1.jpg", "C1P2H9.jpg"])
def test_bad_names(bad):
    with pytest.raises(ValueError):
        parse_filename(bad)


def test_polygon_mask_and_scaling():
    poly = [(100, 100), (300, 100), (300, 200), (100, 200)]
    full = polygon_to_mask(poly, 400, 400)
    assert full.shape == (400, 400) and full[150, 200] == 1 and full[50, 50] == 0
    small = polygon_to_mask(poly, 400, 400, (100, 100))
    assert small.shape == (100, 100)
    assert abs(small.mean() - full.mean()) < 0.02


def test_exif_not_applied(tmp_path):
    """Una foto con EXIF orientación 6 debe leerse con su tamaño crudo (apaisado)."""
    img = Image.new("RGB", (64, 32), (255, 0, 0))
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "x.jpg"
    img.save(path, exif=exif.tobytes())
    assert open_raw_image(path).size == (64, 32)
