from __future__ import annotations

from typing import Dict, List

import numpy as np

from thinking_buildings.backends.base import InferenceBackend, RawDetection


class UltralyticsBackend(InferenceBackend):
    def __init__(self) -> None:
        self._model = None
        self._class_names: Dict[int, str] = {}

    def load(self, model_path: str, device: str = "cpu") -> None:
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._class_names = dict(self._model.names)

    def infer(self, frame: np.ndarray, confidence: float) -> List[RawDetection]:
        if self._model is None:
            raise RuntimeError("Model not loaded — call load() first")
        results = self._model(frame, conf=confidence, verbose=False)
        detections: List[RawDetection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                label = self._class_names.get(class_id, str(class_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                detections.append(RawDetection(
                    class_id=class_id,
                    label=label,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                ))
        return detections

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    @staticmethod
    def is_available() -> bool:
        try:
            import ultralytics  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def priority() -> int:
        return 50
