from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple
import time


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float


class EventBus:
    def __init__(self) -> None:
        self._subscribers: List[Callable[[List[Detection]], None]] = []

    def subscribe(self, callback: Callable[[List[Detection]], None]) -> None:
        self._subscribers.append(callback)

    def publish(self, detections: List[Detection]) -> None:
        for callback in self._subscribers:
            callback(detections)
