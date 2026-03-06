from __future__ import annotations

import logging
import platform
import threading

import cv2
import numpy as np

from thinking_buildings.camera_probe import auto_select_camera, negotiate_resolution
from thinking_buildings.config import CameraConfig

logger = logging.getLogger("thinking_buildings")


class VideoCapture:
    def __init__(self, cfg: CameraConfig) -> None:
        source = auto_select_camera(cfg)
        self._is_rtsp = isinstance(source, str) and source.startswith("rtsp://")

        if self._is_rtsp:
            self.cap = cv2.VideoCapture(source)
        else:
            backend = self._pick_backend()
            if backend is not None:
                self.cap = cv2.VideoCapture(source, backend)
            else:
                self.cap = cv2.VideoCapture(source)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source {source}")

        if self._is_rtsp:
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            actual_w, actual_h = negotiate_resolution(
                self.cap, (cfg.width, cfg.height)
            )
        logger.info("Camera resolution: %dx%d", actual_w, actual_h)

    @staticmethod
    def _pick_backend() -> int | None:
        system = platform.system()
        if system == "Linux":
            return cv2.CAP_V4L2
        if system == "Windows":
            return cv2.CAP_DSHOW
        return None

    def read(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        self.cap.release()


class ThreadedCapture:
    """Latest-frame-wins threaded capture. Drops stale frames."""

    def __init__(self, cfg: CameraConfig) -> None:
        self._capture = VideoCapture(cfg)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> ThreadedCapture:
        self._running = True
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()
        return self

    def _reader(self) -> None:
        while self._running:
            frame = self._capture.read()
            if frame is not None:
                with self._lock:
                    self._frame = frame

    def read(self) -> np.ndarray | None:
        with self._lock:
            return self._frame

    def release(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._capture.release()
