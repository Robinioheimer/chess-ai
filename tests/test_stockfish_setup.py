"""Tests for Stockfish discovery, probing and archive extraction."""
import io
import os
import sys
import zipfile
from pathlib import Path

import pytest

from chess_ai.engine import stockfish_setup
from chess_ai.engine.uci import UciEngineError

FAKE_ENGINE = Path(__file__).parent / "fake_engine" / "fake_uci_engine.py"


def test_find_stockfish_via_environment(monkeypatch, tmp_path):
    binary = tmp_path / ("stockfish.exe" if os.name == "nt" else "stockfish")
    binary.write_text("dummy")
    if binary.suffix != ".exe":
        binary.chmod(0o755)
    monkeypatch.setenv("STOCKFISH_PATH", str(binary))
    info = stockfish_setup.find_stockfish()
    assert info is not None
    assert info.path == str(binary)
    assert info.source == "STOCKFISH_PATH environment variable"


def test_find_stockfish_explicit_path(tmp_path):
    binary = tmp_path / "custom-sf.exe"
    binary.write_text("dummy")
    info = stockfish_setup.find_stockfish(str(binary))
    assert info is not None and info.path == str(binary)


def test_find_stockfish_missing_returns_none(monkeypatch):
    monkeypatch.delenv("STOCKFISH_PATH", raising=False)
    monkeypatch.setattr(stockfish_setup.shutil, "which", lambda _name: None)
    monkeypatch.setattr(stockfish_setup, "_candidate_paths", lambda: [])
    assert stockfish_setup.find_stockfish() is None


def test_probe_fake_engine_reports_id_name():
    name = stockfish_setup.probe_stockfish(f"{sys.executable} {FAKE_ENGINE}")
    assert "FakeStockfish" in name


def test_extract_zip_archive_locates_binary(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("stockfish/stockfish.exe", b"binary-bytes")
    binary = stockfish_setup._extract_archive(buffer.getvalue(), tmp_path)
    assert binary.name == "stockfish.exe"
    assert binary.read_bytes() == b"binary-bytes"


def test_ensure_stockfish_without_engine_and_no_download(monkeypatch):
    monkeypatch.setattr(stockfish_setup, "find_stockfish", lambda *a, **k: None)
    with pytest.raises(UciEngineError):
        stockfish_setup.ensure_stockfish(auto_download=False)


def test_ensure_stockfish_uses_downloader(monkeypatch, tmp_path):
    sentinel = stockfish_setup.StockfishInfo(str(tmp_path / "sf"), "sf", "automatic download")
    monkeypatch.setattr(stockfish_setup, "find_stockfish", lambda *a, **k: None)
    monkeypatch.setattr(stockfish_setup, "download_stockfish", lambda *a, **k: sentinel)
    info = stockfish_setup.ensure_stockfish(auto_download=True)
    assert info is sentinel
