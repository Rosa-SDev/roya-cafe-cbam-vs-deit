"""merge_runs une corridas de otra máquina sin duplicar ni sobrescribir."""
import json
import zipfile

from src.merge_runs import merge
from src.utils import write_json_atomic


def _run(path, exp, name, value, debug=None):
    write_json_atomic(path, {"run": name, "experiment": exp, "config_name": name.split("_f")[0],
                             "test": {"macro_f1": value}, "debug_limit": debug})


def test_merge_copies_skips_and_detects_conflicts(tmp_path):
    dest, colab = tmp_path / "results", tmp_path / "descarga"
    _run(dest / "runs" / "E1" / "cnn_scratch_f100_s0.json", "E1", "cnn_scratch_f100_s0", 0.5)
    _run(colab / "x" / "cnn_scratch_f100_s0.json", "E1", "cnn_scratch_f100_s0", 0.5)       # idéntica
    _run(colab / "y" / "deit_pre_f100_s4.json", "E1", "deit_pre_f100_s4", 0.7)             # nueva
    _run(colab / "z" / "cnn_scratch_f100_s0.json", "E1", "cnn_scratch_f100_s0", 0.9)       # conflicto
    _run(colab / "cnn_cbam_last2_scratch_f100_s1.json", "E2", "cnn_cbam_last2_scratch_f100_s1", 0.6)
    _run(colab / "dbg.json", "debug", "cnn_scratch_f100_s9", 0.1, debug=24)                # ignorada
    (colab / "y" / "deit_pre_f100_s4_maps.npz").write_bytes(b"npz")

    stats = merge([colab], dest)
    assert sorted(stats["copied"]) == ["E1/deit_pre_f100_s4", "E2/cnn_cbam_last2_scratch_f100_s1"]
    assert stats["identical"] == ["E1/cnn_scratch_f100_s0"]
    assert len(stats["conflicts"]) == 1
    assert len(stats["skipped"]) == 1
    assert (dest / "runs" / "E1" / "deit_pre_f100_s4_maps.npz").exists()
    original = json.loads((dest / "runs" / "E1" / "cnn_scratch_f100_s0.json").read_text(encoding="utf-8"))
    assert original["test"]["macro_f1"] == 0.5  # el conflicto no sobrescribe

    again = merge([colab], dest)  # idempotente: no duplica
    assert again["copied"] == []


def test_merge_from_drive_zip(tmp_path):
    src = tmp_path / "src"
    _run(src / "results" / "runs" / "E2" / "a_f100_s0.json", "E2", "a_f100_s0", 0.4)
    zpath = tmp_path / "drive.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in src.rglob("*.json"):
            zf.write(f, f.relative_to(src))
    stats = merge([zpath], tmp_path / "dest")
    assert stats["copied"] == ["E2/a_f100_s0"]
