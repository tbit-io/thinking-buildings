from __future__ import annotations

import logging
from typing import List, Tuple

import cv2

from thinking_buildings.config import CameraConfig

logger = logging.getLogger("thinking_buildings")

_DEFAULT_FALLBACKS: List[Tuple[int, int]] = [
    (1280, 720),
    (640, 480),
    (320, 240),
]


def enumerate_cameras(max_index: int = 8) -> List[int]:
    """Probe camera indices and return those that open successfully."""
    available: List[int] = []
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available


def negotiate_resolution(
    cap: cv2.VideoCapture,
    preferred: Tuple[int, int],
    fallbacks: List[Tuple[int, int]] | None = None,
) -> Tuple[int, int]:
    """Try to set the preferred resolution, fall back to alternatives.

    Returns the actual resolution achieved.
    """
    candidates = [preferred] + (fallbacks or _DEFAULT_FALLBACKS)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for res in candidates:
        if res not in seen:
            seen.add(res)
            unique.append(res)

    for w, h in unique:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_w == w and actual_h == h:
            return (actual_w, actual_h)

    # Return whatever the camera settled on
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.warning(
        "Could not set any preferred resolution, using %dx%d",
        actual_w, actual_h,
    )
    return (actual_w, actual_h)


def auto_select_camera(cfg: CameraConfig) -> int | str:
    """Return camera index or RTSP URL. If source == -1, auto-detect first available."""
    if isinstance(cfg.source, str):
        return cfg.source
    if cfg.source != -1:
        return cfg.source
    available = enumerate_cameras()
    if not available:
        raise RuntimeError("No cameras found during auto-detection")
    logger.info("Auto-detected cameras: %s, using index %d", available, available[0])
    return available[0]
