from __future__ import annotations

import time
from typing import List, Set

import numpy as np
from ultralytics import YOLO

from thinking_buildings.config import DetectorConfig
from thinking_buildings.events import Detection


class Detector:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.model = YOLO(cfg.model)
        self.confidence = cfg.confidence
        self.target_classes: Set[str] = set(cfg.classes)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        results = self.model(frame, conf=self.confidence, verbose=False)
        detections: List[Detection] = []
        now = time.time()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self.model.names[class_id]
                if label not in self.target_classes:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(Detection(
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    timestamp=now,
                ))

        return detections
