import time
from unittest.mock import MagicMock

import cv2
import numpy as np

from thinking_buildings.capture import ThreadedCapture, VideoCapture
from thinking_buildings.config import CameraConfig


class TestPickBackend:
    def test_linux_returns_v4l2(self, mocker):
        mocker.patch("thinking_buildings.capture.platform.system", return_value="Linux")
        assert VideoCapture._pick_backend() == cv2.CAP_V4L2

    def test_windows_returns_dshow(self, mocker):
        mocker.patch("thinking_buildings.capture.platform.system", return_value="Windows")
        assert VideoCapture._pick_backend() == cv2.CAP_DSHOW

    def test_darwin_returns_none(self, mocker):
        mocker.patch("thinking_buildings.capture.platform.system", return_value="Darwin")
        assert VideoCapture._pick_backend() is None


class TestThreadedCapture:
    def _mock_video_capture(self, mocker):
        """Mock VideoCapture so it doesn't touch real cameras."""
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (False, None)
        mock_cap.get.return_value = 640.0
        mocker.patch("thinking_buildings.capture.cv2.VideoCapture", return_value=mock_cap)
        mocker.patch("thinking_buildings.capture.auto_select_camera", return_value=0)
        mocker.patch("thinking_buildings.capture.negotiate_resolution", return_value=(640, 480))
        return mock_cap

    def test_start_returns_self(self, mocker):
        self._mock_video_capture(mocker)
        cfg = CameraConfig()
        tc = ThreadedCapture(cfg)
        result = tc.start()
        assert result is tc
        tc.release()

    def test_read_returns_none_before_start(self, mocker):
        self._mock_video_capture(mocker)
        cfg = CameraConfig()
        tc = ThreadedCapture(cfg)
        assert tc.read() is None

    def test_read_returns_latest_frame(self, mocker):
        mock_cap = self._mock_video_capture(mocker)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mock_cap.read.return_value = (True, frame)

        cfg = CameraConfig()
        tc = ThreadedCapture(cfg)
        tc.start()
        time.sleep(0.1)
        result = tc.read()
        assert result is not None
        assert result.shape == (480, 640, 3)
        tc.release()

    def test_release_stops_thread(self, mocker):
        self._mock_video_capture(mocker)
        cfg = CameraConfig()
        tc = ThreadedCapture(cfg)
        tc.start()
        assert tc._running is True
        tc.release()
        assert tc._running is False
        assert not tc._thread.is_alive()
