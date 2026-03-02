from __future__ import annotations

import time
from typing import List, Set

import numpy as np

from thinking_buildings.backends import get_backend
from thinking_buildings.config import DetectorConfig
from thinking_buildings.events import Detection


class Detector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.backend = get_backend(cfg.backend)
        self.backend.load(cfg.model)
        self.confidence = cfg.confidence
        self.target_classes: Set[str] = set(cfg.classes)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        raw_dets = self.backend.infer(frame, self.confidence)
        detections: List[Detection] = []
        now = time.time()

        for raw in raw_dets:
            if raw.label not in self.target_classes:
                continue
            detections.append(Detection(
                label=raw.label,
                confidence=raw.confidence,
                bbox=raw.bbox,
                timestamp=now,
            ))

        return detections
