from unittest.mock import MagicMock, patch

import pytest

from thinking_buildings.camera_probe import (
    auto_select_camera,
    enumerate_cameras,
    negotiate_resolution,
)
from thinking_buildings.config import CameraConfig


class TestEnumerateCameras:
    @patch("thinking_buildings.camera_probe.cv2.VideoCapture")
    def test_finds_available_cameras(self, MockCap):
        caps = {}
        def make_cap(idx):
            cap = MagicMock()
            cap.isOpened.return_value = idx in (0, 2)
            caps[idx] = cap
            return cap
        MockCap.side_effect = make_cap

        result = enumerate_cameras(max_index=4)
        assert result == [0, 2]
        # All opened caps should be released
        caps[0].release.assert_called_once()
        caps[2].release.assert_called_once()

    @patch("thinking_buildings.camera_probe.cv2.VideoCapture")
    def test_no_cameras_returns_empty(self, MockCap):
        cap = MagicMock()
        cap.isOpened.return_value = False
        MockCap.return_value = cap

        result = enumerate_cameras(max_index=3)
        assert result == []


class TestNegotiateResolution:
    def test_preferred_resolution_accepted(self):
        cap = MagicMock()
        cap.get.side_effect = lambda prop: {
            3: 1280.0,  # CAP_PROP_FRAME_WIDTH
            4: 720.0,   # CAP_PROP_FRAME_HEIGHT
        }.get(prop, 0.0)

        w, h = negotiate_resolution(cap, (1280, 720))
        assert (w, h) == (1280, 720)

    def test_falls_back_when_preferred_rejected(self):
        cap = MagicMock()
        # First attempt (1920x1080) rejected, second (1280x720) accepted
        cap.get.side_effect = [
            640.0, 480.0,   # after setting 1920x1080 → camera gives 640x480
            640.0, 480.0,   # after setting 1280x720 → same
            640.0, 480.0,   # after setting 640x480 → matches!
        ]

        w, h = negotiate_resolution(cap, (1920, 1080))
        assert (w, h) == (640, 480)


class TestAutoSelectCamera:
    def test_explicit_source_passthrough(self):
        cfg = CameraConfig(source=2)
        assert auto_select_camera(cfg) == 2

    @patch("thinking_buildings.camera_probe.enumerate_cameras")
    def test_auto_detect_picks_first(self, mock_enum):
        mock_enum.return_value = [1, 3]
        cfg = CameraConfig(source=-1)
        assert auto_select_camera(cfg) == 1

    @patch("thinking_buildings.camera_probe.enumerate_cameras")
    def test_auto_detect_no_cameras_raises(self, mock_enum):
        mock_enum.return_value = []
        cfg = CameraConfig(source=-1)
        with pytest.raises(RuntimeError, match="No cameras found"):
            auto_select_camera(cfg)
