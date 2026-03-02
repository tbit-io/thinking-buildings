from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple
import time


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float
    identity: Optional[str] = None
    face_confidence: Optional[float] = None
    face_bbox: Optional[Tuple[int, int, int, int]] = None
    occlusion: bool = False


class EventBus:
    def __init__(self) -> None:
        self._subscribers: List[Callable[[List[Detection]], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, callback: Callable[[List[Detection]], None]) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def publish(self, detections: List[Detection]) -> None:
        with self._lock:
            subscribers = list(self._subscribers)
        for callback in subscribers:
            callback(detections)
