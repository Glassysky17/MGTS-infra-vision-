"""Общие фикстуры тестового набора."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def cfg():
    from mgts.config import load_config
    return load_config()


@pytest.fixture(scope="session")
def classes():
    from mgts.config import load_classes
    return load_classes()


@pytest.fixture
def rng():
    import numpy as np
    return np.random.default_rng(0)
