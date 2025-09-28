"""Pytest configuration for the matching service tests."""

import os
import sys
from pathlib import Path


os.environ.setdefault("DATABASE_URI", "sqlite:///:memory:")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage import init_db, reset_database


import pytest


@pytest.fixture(scope="session", autouse=True)
def _initialize_database():
    init_db()
    reset_database()
    yield


@pytest.fixture(autouse=True)
def _clean_database():
    reset_database()
    yield
