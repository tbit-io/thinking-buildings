from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from thinking_buildings.config import FaceRecConfig
from thinking_buildings.events import Detection
from thinking_buildings.face_recognizer import FaceRecognizer


@pytest.fixture
def recognizer(tmp_path):
    """Create a FaceRecognizer with mocked InsightFace app."""
    cfg = FaceRecConfig(
        db_path=str(tmp_path / "test.db"),
        recognition_threshold=0.4,
        occlusion_det_threshold=0.5,
    )
    with patch("thinking_buildings.face_recognizer.FaceAnalysis") as MockFA:
        mock_app = MagicMock()
        MockFA.return_value = mock_app
        rec = FaceRecognizer(cfg)
    yield rec
    rec.close()


class TestMatchEmbedding:
    def test_exact_match_high_similarity(self, recognizer):
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        recognizer.db.add_embedding("alice", emb)

        name, sim = recognizer._match_embedding(emb)
        assert name == "alice"
        assert sim is not None
        assert sim > 0.99

    def test_orthogonal_near_zero(self, recognizer):
        emb1 = np.zeros(512, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(512, dtype=np.float32)
        emb2[1] = 1.0
        recognizer.db.add_embedding("alice", emb1)

        name, sim = recognizer._match_embedding(emb2)
        assert sim is not None
        assert abs(sim) < 0.01

    def test_empty_db_returns_unknown(self, recognizer):
        emb = np.random.randn(512).astype(np.float32)
        name, sim = recognizer._match_embedding(emb)
        assert name == "unknown_person"
        assert sim is None

    def test_below_threshold_returns_unknown(self, recognizer):
        emb_stored = np.random.randn(512).astype(np.float32)
        emb_stored = emb_stored / np.linalg.norm(emb_stored)
        recognizer.db.add_embedding("alice", emb_stored)

        # Nearly orthogonal — should be below threshold
        emb_query = np.zeros(512, dtype=np.float32)
        emb_query[0] = 1.0
        name, sim = recognizer._match_embedding(emb_query)
        assert name == "unknown_person"

    def test_above_threshold_returns_name(self, recognizer):
        emb = np.random.randn(512).astype(np.float32)
        emb = emb / np.linalg.norm(emb)
        recognizer.db.add_embedding("bob", emb)

        # Slightly perturbed but still very similar
        noise = np.random.randn(512).astype(np.float32) * 0.01
        query = emb + noise
        name, sim = recognizer._match_embedding(query)
        assert name == "bob"
        assert sim is not None
        assert sim > 0.4


class TestLandmarksAnomaly:
    def _make_face(self, kps):
        face = MagicMock()
        face.landmark_2d_106 = None
        face.kps = np.array(kps, dtype=np.float32)
        return face

    def test_symmetric_landmarks_no_anomaly(self, recognizer):
        # left_eye, right_eye, nose, left_mouth, right_mouth
        kps = [[30, 50], [70, 50], [50, 70], [35, 90], [65, 90]]
        face = self._make_face(kps)
        assert not recognizer._landmarks_anomaly(face)

    def test_asymmetric_landmarks_anomaly(self, recognizer):
        # left eye very far from nose, right eye close — ratio > 2
        kps = [[10, 50], [50, 50], [49, 70], [35, 90], [65, 90]]
        face = self._make_face(kps)
        assert recognizer._landmarks_anomaly(face)

    def test_zero_distance_is_anomaly(self, recognizer):
        kps = [[50, 70], [70, 50], [50, 70], [35, 90], [65, 90]]  # left_eye == nose
        face = self._make_face(kps)
        assert recognizer._landmarks_anomaly(face)


class TestFaceBboxToFrame:
    def test_offsets_applied(self, recognizer):
        face_box = np.array([10, 20, 50, 60])
        result = recognizer._face_bbox_to_frame(face_box, offset_x=100, offset_y=200)
        assert result == (110, 220, 150, 260)


class TestRecognize:
    def test_non_person_passes_through(self, recognizer, mock_frame):
        det = Detection(
            label="cat", confidence=0.9, bbox=(10, 10, 50, 50), timestamp=1.0
        )
        result = recognizer.recognize(mock_frame, [det])
        assert len(result) == 1
        assert result[0].label == "cat"
        assert result[0].identity is None
