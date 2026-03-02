from unittest.mock import MagicMock, patch

import numpy as np

from thinking_buildings.backends.base import RawDetection
from thinking_buildings.config import DetectorConfig


class TestDetector:
    @patch("thinking_buildings.detector.get_backend")
    def test_filters_target_classes(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        from thinking_buildings.detector import Detector

        cfg = DetectorConfig(classes=["person", "cat"])
        detector = Detector(cfg)

        mock_backend.infer.return_value = [
            RawDetection(class_id=0, label="person", confidence=0.9, bbox=(10, 20, 100, 200)),
            RawDetection(class_id=1, label="bicycle", confidence=0.8, bbox=(50, 50, 150, 250)),
        ]

        dets = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(dets) == 1
        assert dets[0].label == "person"
        assert dets[0].confidence == 0.9

    @patch("thinking_buildings.detector.get_backend")
    def test_detection_fields_populated(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        from thinking_buildings.detector import Detector

        cfg = DetectorConfig(classes=["cat"])
        detector = Detector(cfg)

        mock_backend.infer.return_value = [
            RawDetection(class_id=15, label="cat", confidence=0.75, bbox=(10, 20, 80, 120)),
        ]

        dets = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert len(dets) == 1
        assert dets[0].label == "cat"
        assert dets[0].bbox == (10, 20, 80, 120)
        assert dets[0].identity is None

    @patch("thinking_buildings.detector.get_backend")
    def test_empty_results(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        from thinking_buildings.detector import Detector

        cfg = DetectorConfig(classes=["person"])
        detector = Detector(cfg)

        mock_backend.infer.return_value = []

        dets = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        assert dets == []

    @patch("thinking_buildings.detector.get_backend")
    def test_backend_receives_correct_config(self, mock_get_backend):
        mock_backend = MagicMock()
        mock_get_backend.return_value = mock_backend

        from thinking_buildings.detector import Detector

        cfg = DetectorConfig(model="yolo11n.pt", backend="ultralytics")
        Detector(cfg)

        mock_get_backend.assert_called_once_with("ultralytics")
        mock_backend.load.assert_called_once_with("yolo11n.pt")
