from __future__ import annotations

import cv2
import numpy as np

from thinking_buildings.config import CameraConfig


class VideoCapture:
    def __init__(self, cfg: CameraConfig) -> None:
        self.cap = cv2.VideoCapture(cfg.source, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera source {cfg.source}")

    def read(self) -> np.ndarray | None:
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def release(self) -> None:
        self.cap.release()
