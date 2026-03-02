from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

from thinking_buildings.config import (
    AlerterConfig,
    AppConfig,
    CameraConfig,
    DetectorConfig,
    DisplayConfig,
    FaceRecConfig,
    LoggingConfig,
)
from thinking_buildings.events import Detection


@pytest.fixture
def make_detection():
    """Factory fixture for creating Detection instances with sensible defaults."""

    def _make(**overrides: Any) -> Detection:
        defaults = dict(
            label="person",
            confidence=0.85,
            bbox=(100, 100, 200, 300),
            timestamp=time.time(),
        )
        defaults.update(overrides)
        return Detection(**defaults)

    return _make


@pytest.fixture
def sample_config() -> AppConfig:
    """AppConfig with test-safe defaults."""
    return AppConfig(
        camera=CameraConfig(),
        detector=DetectorConfig(),
        alerter=AlerterConfig(cooldown_seconds=30, desktop_notifications=False),
        display=DisplayConfig(show_window=False),
        logging=LoggingConfig(level="DEBUG", log_file="/dev/null"),
        face_recognition=FaceRecConfig(
            enabled=False,
            db_path=":memory:",
        ),
    )


@pytest.fixture
def mock_frame() -> np.ndarray:
    """A dummy BGR frame (480x640x3)."""
    return np.zeros((480, 640, 3), dtype=np.uint8)
