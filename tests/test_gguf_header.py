"""scripts/gguf_header.py parses a GGUF header correctly.

The README makes claims about the shipped weights (architecture, parameter count,
quantization mix) and points readers at that script to check them. If the parser is
wrong the claims are wrong, so it gets a test rather than trust.

Built against a synthetic GGUF so the test needs no 806 MB download.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gguf_header.py"

spec = importlib.util.spec_from_file_location("gguf_header", SCRIPT)
gguf_header = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gguf_header)

U32, STRING, U64 = gguf_header.U32, gguf_header.STRING, gguf_header.U64

Q4_K, Q5_0, F32 = 12, 6, 0


def _string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _tensor(name: str, dims, ttype: int, offset: int = 0) -> bytes:
    out = _string(name) + struct.pack("<I", len(dims))
    for d in dims:
        out += struct.pack("<Q", d)
    return out + struct.pack("<I", ttype) + struct.pack("<Q", offset)


@pytest.fixture
def synthetic(tmp_path):
    """Two 2-D tensors: one K-quantizable row length, one not."""
    body = b"GGUF" + struct.pack("<I", 3)
    body += struct.pack("<Q", 3)  # tensors
    body += struct.pack("<Q", 2)  # metadata keys

    body += _string("general.architecture") + struct.pack("<I", STRING) + _string("gemma3")
    body += _string("general.size_label") + struct.pack("<I", STRING) + _string("1000M")

    body += _tensor("blk.0.attn_output.weight", [1024, 1152], Q4_K)  # 1024 % 256 == 0
    body += _tensor("blk.0.ffn_up.weight", [1152, 6912], Q5_0)       # 1152 % 256 != 0
    body += _tensor("output_norm.weight", [1152], F32)               # 1-D, not counted

    path = tmp_path / "synthetic.gguf"
    path.write_bytes(body)
    return path


def test_parses_metadata_and_parameter_count(synthetic, capsys):
    assert gguf_header.main(str(synthetic)) == 0
    out = capsys.readouterr().out

    assert "general.architecture = 'gemma3'" in out
    assert "general.size_label = '1000M'" in out
    # 1024*1152 + 1152*6912 + 1152
    assert f"{1024 * 1152 + 1152 * 6912 + 1152:,}" in out


def test_reports_the_quantization_mix(synthetic, capsys):
    gguf_header.main(str(synthetic))
    out = capsys.readouterr().out
    assert "Q4_K" in out and "Q5_0" in out and "F32" in out


def test_reports_the_k_quant_row_length_fallback(synthetic, capsys):
    """The finding the README's design note rests on."""
    gguf_header.main(str(synthetic))
    out = capsys.readouterr().out
    assert "row length divisible by 256: True    K-quant: True " in out
    assert "row length divisible by 256: False   K-quant: False" in out


def test_rejects_a_file_that_is_not_gguf(tmp_path, capsys):
    bad = tmp_path / "not.gguf"
    bad.write_bytes(b"NOPE" + b"\x00" * 64)
    assert gguf_header.main(str(bad)) == 1
    assert "not a GGUF file" in capsys.readouterr().err


def test_truncated_file_raises_rather_than_reporting_nonsense(tmp_path):
    short = tmp_path / "short.gguf"
    short.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 99))
    with pytest.raises(EOFError):
        gguf_header.main(str(short))


@pytest.mark.skipif(
    not (SCRIPT.parent.parent / "models" / "docker_gemma3_4b_q4km.gguf").exists(),
    reason="shipped GGUF not present",
)
def test_shipped_gguf_matches_what_the_readme_claims(capsys):
    """Runs only where the weights are on disk. Pins the README's numbers."""
    model = SCRIPT.parent.parent / "models" / "docker_gemma3_4b_q4km.gguf"
    assert gguf_header.main(str(model)) == 0
    out = capsys.readouterr().out

    assert "general.architecture = 'gemma3'" in out
    assert "general.size_label = '1000M'" in out
    assert "999,885,952" in out


def test_missing_file_reports_cleanly(tmp_path, capsys):
    """The README points readers at this script; a traceback is not an answer."""
    assert gguf_header.main(str(tmp_path / "nope.gguf")) == 1
    assert "cannot read" in capsys.readouterr().err
