from unittest.mock import MagicMock, patch

import pytest

from thinking_buildings.backends import get_backend
from thinking_buildings.backends.base import InferenceBackend, RawDetection


class TestGetBackend:
    @patch("thinking_buildings.backends._REGISTRY", [])
    @patch("thinking_buildings.backends._register_backends")
    def test_no_backends_raises(self, mock_register):
        with pytest.raises(RuntimeError, match="No inference backends"):
            get_backend()

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("nonexistent")

    def test_auto_returns_backend(self):
        backend = get_backend("auto")
        assert isinstance(backend, InferenceBackend)

    def test_explicit_ultralytics(self):
        backend = get_backend("ultralytics")
        assert isinstance(backend, InferenceBackend)
        assert backend.__class__.__name__ == "UltralyticsBackend"


class TestUltralyticsBackend:
    def test_is_available(self):
        from thinking_buildings.backends.ultralytics_backend import UltralyticsBackend
        # Should be True in dev environment
        assert UltralyticsBackend.is_available() is True

    def test_priority(self):
        from thinking_buildings.backends.ultralytics_backend import UltralyticsBackend
        assert UltralyticsBackend.priority() == 50

    def test_load_and_infer(self):
        import numpy as np

        from thinking_buildings.backends.ultralytics_backend import UltralyticsBackend

        backend = UltralyticsBackend()
        with patch("ultralytics.YOLO") as MockYOLO:
            mock_model = MagicMock()
            mock_model.names = {0: "person"}
            MockYOLO.return_value = mock_model

            backend.load("yolo11n.pt")
            assert backend.class_names == {0: "person"}

            # Mock inference result
            box = MagicMock()
            box.cls = [0]
            box.conf = [0.9]
            box.xyxy = [np.array([10, 20, 100, 200])]
            result = MagicMock()
            result.boxes = [box]
            mock_model.return_value = [result]

            dets = backend.infer(np.zeros((480, 640, 3), dtype=np.uint8), 0.5)
            assert len(dets) == 1
            assert isinstance(dets[0], RawDetection)
            assert dets[0].label == "person"
            assert dets[0].confidence == 0.9

    def test_infer_before_load_raises(self):
        import numpy as np

        from thinking_buildings.backends.ultralytics_backend import UltralyticsBackend

        backend = UltralyticsBackend()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            backend.infer(np.zeros((480, 640, 3), dtype=np.uint8), 0.5)


class TestRawDetection:
    def test_fields(self):
        rd = RawDetection(class_id=0, label="person", confidence=0.9, bbox=(10, 20, 100, 200))
        assert rd.class_id == 0
        assert rd.label == "person"
        assert rd.confidence == 0.9
        assert rd.bbox == (10, 20, 100, 200)
