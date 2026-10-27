"""Shared pytest fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

FAKE_ENGINE = Path(__file__).parent / "fake_engine" / "fake_uci_engine.py"


@pytest.fixture
def fake_engine_command() -> str:
    """Command line for the deterministic fake UCI engine subprocess."""
    return f"{sys.executable} {FAKE_ENGINE}"
