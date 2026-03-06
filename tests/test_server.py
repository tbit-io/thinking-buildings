import textwrap
import time
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from thinking_buildings.events import Detection
from thinking_buildings.server import (
    _detection_to_dict,
    _frame_to_jpeg,
    _read_upload_as_frame,
    create_app,
)

# --- Helpers ---

def _make_test_frame(w=640, h=480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_jpeg_bytes(w=640, h=480):
    frame = _make_test_frame(w, h)
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def _make_detection(**kw):
    defaults = dict(
        label="person", confidence=0.9, bbox=(10, 20, 100, 200), timestamp=time.time()
    )
    defaults.update(kw)
    return Detection(**defaults)


# --- Unit tests ---

class TestDetectionToDict:
    def test_basic(self):
        det = _make_detection(camera_id="cam1")
        d = _detection_to_dict(det)
        assert d["label"] == "person"
        assert d["camera_id"] == "cam1"
        assert isinstance(d["bbox"], list)

    def test_with_face_bbox(self):
        det = _make_detection(face_bbox=(5, 10, 50, 60))
        d = _detection_to_dict(det)
        assert isinstance(d["face_bbox"], list)
        assert d["face_bbox"] == [5, 10, 50, 60]

    def test_none_face_bbox(self):
        det = _make_detection()
        d = _detection_to_dict(det)
        assert d["face_bbox"] is None


class TestFrameToJpeg:
    def test_returns_bytes(self):
        frame = _make_test_frame()
        result = _frame_to_jpeg(frame)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_valid_jpeg(self):
        frame = _make_test_frame()
        jpeg = _frame_to_jpeg(frame)
        decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None


class TestReadUploadAsFrame:
    def test_valid_jpeg(self):
        jpeg = _make_jpeg_bytes()
        frame = _read_upload_as_frame(jpeg)
        assert frame.shape == (480, 640, 3)

    def test_invalid_bytes_raises(self):
        with pytest.raises(ValueError, match="Could not decode"):
            _read_upload_as_frame(b"not an image")


# --- Integration tests with mocked pipeline ---

@pytest.fixture
def client(tmp_path):
    """Create a test client with mocked capture and detector."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(textwrap.dedent("""\
        cameras:
          - id: test_cam
            source: 0
            location: "Test"
            width: 640
            height: 480
        face_recognition:
          enabled: false
    """))

    frame = _make_test_frame()
    mock_capture = MagicMock()
    mock_capture.read.return_value = frame
    mock_capture.start.return_value = mock_capture

    det = _make_detection()

    with (
        patch("thinking_buildings.server.ThreadedCapture", return_value=mock_capture),
        patch("thinking_buildings.server.Detector") as MockDetector,
    ):
        MockDetector.return_value.detect.return_value = [det]
        app = create_app(str(config_file))
        with TestClient(app) as tc:
            yield tc


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["cameras"] == 1
        assert data["face_recognition"] is False


class TestCamerasEndpoint:
    def test_list_cameras(self, client):
        resp = client.get("/cameras")
        assert resp.status_code == 200
        cameras = resp.json()
        assert len(cameras) == 1
        assert cameras[0]["id"] == "test_cam"
        assert cameras[0]["location"] == "Test"
        assert cameras[0]["source_type"] == "local"


class TestSnapshotEndpoint:
    def test_returns_jpeg(self, client):
        resp = client.get("/cameras/test_cam/snapshot")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/jpeg"
        decoded = cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)
        assert decoded is not None

    def test_unknown_camera_404(self, client):
        resp = client.get("/cameras/nonexistent/snapshot")
        assert resp.status_code == 404


class TestDetectCameraEndpoint:
    def test_detect_returns_detections(self, client):
        resp = client.post("/cameras/test_cam/detect")
        assert resp.status_code == 200
        data = resp.json()
        assert data["camera_id"] == "test_cam"
        assert len(data["detections"]) == 1
        assert data["detections"][0]["label"] == "person"

    def test_unknown_camera_404(self, client):
        resp = client.post("/cameras/nonexistent/detect")
        assert resp.status_code == 404


class TestDetectUploadEndpoint:
    def test_detect_uploaded_image(self, client):
        jpeg = _make_jpeg_bytes()
        resp = client.post("/detect", files={"file": ("test.jpg", jpeg, "image/jpeg")})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["detections"]) == 1

    def test_invalid_image_400(self, client):
        resp = client.post(
            "/detect", files={"file": ("bad.jpg", b"not an image", "image/jpeg")}
        )
        assert resp.status_code == 400


class TestEnrollEndpoint:
    def test_enroll_disabled_returns_400(self, client):
        jpeg = _make_jpeg_bytes()
        resp = client.post(
            "/enroll", params={"name": "alice"}, files={"file": ("face.jpg", jpeg)}
        )
        assert resp.status_code == 400
        assert "not enabled" in resp.json()["detail"]
