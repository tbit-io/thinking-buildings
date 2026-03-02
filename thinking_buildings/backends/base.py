from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class RawDetection:
    class_id: int
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2


class InferenceBackend(ABC):
    @abstractmethod
    def load(self, model_path: str, device: str = "cpu") -> None:
        ...

    @abstractmethod
    def infer(self, frame: np.ndarray, confidence: float) -> List[RawDetection]:
        ...

    @property
    @abstractmethod
    def class_names(self) -> Dict[int, str]:
        ...

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        ...

    @staticmethod
    @abstractmethod
    def priority() -> int:
        ...
