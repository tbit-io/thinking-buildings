from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np

from thinking_buildings.config import DisplayConfig
from thinking_buildings.events import Detection

# BGR colors
COLOR_RED = (0, 0, 255)
COLOR_ORANGE = (0, 140, 255)
COLOR_GREEN = (0, 255, 0)


class Display:
    def __init__(self, cfg: DisplayConfig) -> None:
        self.show_fps = cfg.show_fps
        self.default_color = cfg.box_color
        self.box_thickness = cfg.box_thickness
        self._prev_time = time.time()
        self._fps = 0.0
        self._latest_detections: List[Detection] = []

    def handle(self, detections: List[Detection]) -> None:
        self._latest_detections = detections

    def _box_color(self, det: Detection) -> Tuple[int, int, int]:
        if det.identity is None:
            return self.default_color
        if det.occlusion:
            return COLOR_RED
        if det.identity == "unknown_person":
            return COLOR_ORANGE
        return COLOR_GREEN

    def render(self, frame: np.ndarray) -> np.ndarray:
        for det in self._latest_detections:
            color = self._box_color(det)
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.box_thickness)

            label = self._build_label(det)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Draw face bounding box if available
            if det.face_bbox:
                fx1, fy1, fx2, fy2 = det.face_bbox
                cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), color, 1)

        if self.show_fps:
            now = time.time()
            dt = now - self._prev_time
            self._prev_time = now
            self._fps = 1.0 / dt if dt > 0 else 0
            cv2.putText(frame, f"FPS: {self._fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        return frame

    def _build_label(self, det: Detection) -> str:
        if det.identity and det.identity not in ("unknown_person", "occluded_face"):
            return f"{det.identity} {det.confidence:.0%}"
        if det.identity == "unknown_person":
            return f"unknown {det.confidence:.0%}"
        if det.occlusion:
            return f"OCCLUDED {det.confidence:.0%}"
        return f"{det.label} {det.confidence:.0%}"

    @staticmethod
    def show(frame: np.ndarray) -> int:
        cv2.imshow("Thinking Buildings", frame)
        return cv2.waitKey(1) & 0xFF

    @staticmethod
    def cleanup() -> None:
        cv2.destroyAllWindows()
