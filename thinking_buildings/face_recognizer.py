from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from thinking_buildings.config import FaceRecConfig
from thinking_buildings.events import Detection
from thinking_buildings.face_db import FaceDB

logger = logging.getLogger("thinking_buildings")


class FaceRecognizer:
    def __init__(self, cfg: FaceRecConfig) -> None:
        self.cfg = cfg
        self.db = FaceDB(cfg.db_path)
        self.app = FaceAnalysis(name=cfg.model_name, allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0, det_size=cfg.det_size)
        self._occlusion_counts: dict[str, int] = {}
        logger.info("FaceRecognizer initialized with model %s", cfg.model_name)

    def recognize(self, frame: np.ndarray, detections: List[Detection]) -> List[Detection]:
        enriched: List[Detection] = []
        active_keys = set()
        for det in detections:
            if det.label != "person":
                enriched.append(det)
                continue
            active_keys.add(self._region_key(det.bbox))
            enriched.append(self._process_person(frame, det))
        # Clean up counters for regions no longer detected
        stale = [k for k in self._occlusion_counts if k not in active_keys]
        for k in stale:
            del self._occlusion_counts[k]
        return enriched

    def _region_key(self, bbox: Tuple[int, int, int, int]) -> str:
        """Quantize bbox center to 50px grid for tracking continuity."""
        cx = (bbox[0] + bbox[2]) // 2 // 50
        cy = (bbox[1] + bbox[3]) // 2 // 50
        return f"{cx},{cy}"

    def _check_occlusion_grace(self, det: Detection) -> bool:
        """Increment counter for this region. Return True if grace period exceeded."""
        key = self._region_key(det.bbox)
        self._occlusion_counts[key] = self._occlusion_counts.get(key, 0) + 1
        return self._occlusion_counts[key] >= self.cfg.occlusion_grace_frames

    def _clear_occlusion(self, det: Detection) -> None:
        """Reset the grace counter when a face is successfully detected."""
        key = self._region_key(det.bbox)
        self._occlusion_counts.pop(key, None)

    def _process_person(self, frame: np.ndarray, det: Detection) -> Detection:
        x1, y1, x2, y2 = det.bbox
        h, w = frame.shape[:2]

        # Pad the crop by 20% for better face detection
        bw, bh = x2 - x1, y2 - y1
        pad_x, pad_y = int(bw * 0.2), int(bh * 0.2)
        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(w, x2 + pad_x)
        cy2 = min(h, y2 + pad_y)
        crop = frame[cy1:cy2, cx1:cx2]

        faces = self.app.get(crop)

        if not faces:
            # High-confidence person but no face → possible occlusion
            if det.confidence >= self.cfg.occlusion_det_threshold:
                if self._check_occlusion_grace(det):
                    return Detection(
                        label=det.label,
                        confidence=det.confidence,
                        bbox=det.bbox,
                        timestamp=det.timestamp,
                        identity="occluded_face",
                        occlusion=True,
                    )
            return det

        face = max(faces, key=lambda f: f.det_score)

        # Check for occlusion via low face detection score
        if face.det_score < self.cfg.occlusion_det_threshold:
            if self._check_occlusion_grace(det):
                face_bbox = self._face_bbox_to_frame(face.bbox, cx1, cy1)
                return Detection(
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    timestamp=det.timestamp,
                    identity="occluded_face",
                    face_confidence=float(face.det_score),
                    face_bbox=face_bbox,
                    occlusion=True,
                )
            return det

        # Check landmark geometry for partial mask/occlusion
        if self._landmarks_anomaly(face):
            if self._check_occlusion_grace(det):
                face_bbox = self._face_bbox_to_frame(face.bbox, cx1, cy1)
                return Detection(
                    label=det.label,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    timestamp=det.timestamp,
                    identity="occluded_face",
                    face_confidence=float(face.det_score),
                    face_bbox=face_bbox,
                    occlusion=True,
                )
            return det

        # Face found and clear — reset grace counter
        self._clear_occlusion(det)

        # Match against known faces
        face_bbox = self._face_bbox_to_frame(face.bbox, cx1, cy1)
        identity, similarity = self._match_embedding(face.embedding)

        return Detection(
            label=det.label,
            confidence=det.confidence,
            bbox=det.bbox,
            timestamp=det.timestamp,
            identity=identity,
            face_confidence=similarity,
            face_bbox=face_bbox,
            occlusion=False,
        )

    def _face_bbox_to_frame(
        self, face_box: np.ndarray, offset_x: int, offset_y: int
    ) -> Tuple[int, int, int, int]:
        fx1, fy1, fx2, fy2 = (int(v) for v in face_box[:4])
        return (fx1 + offset_x, fy1 + offset_y, fx2 + offset_x, fy2 + offset_y)

    def _landmarks_anomaly(self, face) -> bool:
        if not hasattr(face, "landmark_2d_106") or face.landmark_2d_106 is None:
            if not hasattr(face, "kps") or face.kps is None:
                return False
            kps = face.kps
            if len(kps) < 5:
                return False
            # 5-point landmarks: left_eye, right_eye, nose, left_mouth, right_mouth
            left_eye, right_eye, nose = kps[0], kps[1], kps[2]
            d_left = np.linalg.norm(left_eye - nose)
            d_right = np.linalg.norm(right_eye - nose)
            if d_left == 0 or d_right == 0:
                return True
            ratio = max(d_left, d_right) / min(d_left, d_right)
            return ratio > 2.0

        landmarks = face.landmark_2d_106
        left_eye = landmarks[33]
        right_eye = landmarks[87]
        nose = landmarks[86]
        d_left = np.linalg.norm(left_eye - nose)
        d_right = np.linalg.norm(right_eye - nose)
        if d_left == 0 or d_right == 0:
            return True
        ratio = max(d_left, d_right) / min(d_left, d_right)
        return ratio > 2.0

    def _match_embedding(self, embedding: np.ndarray) -> Tuple[str, Optional[float]]:
        all_embeddings = self.db.get_all_embeddings()
        if not all_embeddings:
            return "unknown_person", None

        best_name = "unknown_person"
        best_sim = -1.0
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-8)

        for name, embs in all_embeddings.items():
            for stored in embs:
                stored_norm = stored / (np.linalg.norm(stored) + 1e-8)
                sim = float(np.dot(emb_norm, stored_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name

        if best_sim >= self.cfg.recognition_threshold:
            return best_name, best_sim
        return "unknown_person", best_sim

    def enroll_from_directory(self, faces_dir: str) -> int:
        faces_path = Path(faces_dir)
        if not faces_path.exists():
            logger.warning("Faces directory %s does not exist", faces_dir)
            return 0

        count = 0
        for person_dir in sorted(faces_path.iterdir()):
            if not person_dir.is_dir():
                continue
            name = person_dir.name
            for img_path in sorted(person_dir.iterdir()):
                if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                    continue
                img = cv2.imread(str(img_path))
                if img is None:
                    logger.warning("Could not read image %s", img_path)
                    continue
                faces = self.app.get(img)
                if not faces:
                    logger.warning("No face found in %s", img_path)
                    continue
                face = max(faces, key=lambda f: f.det_score)
                self.db.add_embedding(name, face.embedding, source=str(img_path))
                count += 1
                logger.info("Enrolled face for '%s' from %s", name, img_path.name)

        logger.info("Enrolled %d face(s) total", count)
        return count

    def enroll_from_frame(self, name: str, frame: np.ndarray) -> bool:
        faces = self.app.get(frame)
        if not faces:
            logger.warning("No face found in frame for enrollment")
            return False
        face = max(faces, key=lambda f: f.det_score)
        self.db.add_embedding(name, face.embedding, source="camera_capture")
        logger.info("Enrolled face for '%s' from camera", name)
        return True

    def close(self) -> None:
        self.db.close()
