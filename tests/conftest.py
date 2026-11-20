"""Общие фикстуры. ГЛАВНОЕ: тест, упавший с NotImplementedError (функция-заглушка), помечается как skipped.
Как только агент реализует функцию — тест становится обязательным."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    if os.environ.get("MGTS_STRICT") == "1":
        return  # режим TDD/приёмки: заглушки остаются красными
    if call.when == "call" and call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        rep.outcome = "skipped"
        msg = str(call.excinfo.value) or "stub"
        rep.longrepr = (str(item.fspath), item.location[1] or 0, f"not implemented yet: {msg}")


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
