from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import subprocess
import tempfile

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse

from thinking_buildings.config import load_config
from thinking_buildings.detector import Detector
from thinking_buildings.face_recognizer import FaceRecognizer

logger = logging.getLogger("thinking_buildings")

CAPTURES_DIR = Path("captures")
SNAPSHOTS_DIR = Path("snapshots")
SAVE_LABELS = {"person", "car", "truck", "bus"}

RTSP_CAMERAS = {
    "sala": "rtsp://admin:Ijavier.27@192.168.1.19:554/cam/realmonitor?channel=1&subtype=0",
    "nvr_ch1": "rtsp://admin:admin211@192.168.1.14:554/cam/realmonitor?channel=1&subtype=0",
    "nvr_ch2": "rtsp://admin:admin211@192.168.1.14:554/cam/realmonitor?channel=2&subtype=0",
    "pc": 0,  # webcam del PC
}

app = FastAPI(title="Thinking Buildings Detection API")
_detector: Detector | None = None
_face_rec: FaceRecognizer | None = None


def _get_detector() -> Detector:
    global _detector
    if _detector is None:
        cfg = load_config()
        _detector = Detector(cfg.detector)
    return _detector


def _get_face_rec() -> FaceRecognizer | None:
    global _face_rec
    if _face_rec is None:
        cfg = load_config()
        if cfg.face_recognition.enabled:
            _face_rec = FaceRecognizer(cfg.face_recognition)
            _face_rec.enroll_from_directory(cfg.face_recognition.faces_dir)
    return _face_rec


def _capture_rtsp_frame(source: str | int) -> np.ndarray | None:
    """Capture a single frame from an RTSP stream or local webcam."""
    if isinstance(source, int):
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return None
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    cmd = [
        "ffmpeg", "-y",
        "-rtsp_transport", "tcp",
        "-i", source,
        "-frames:v", "1",
        tmp_path,
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=15)
    if result.returncode != 0:
        Path(tmp_path).unlink(missing_ok=True)
        return None
    frame = cv2.imread(tmp_path)
    Path(tmp_path).unlink(missing_ok=True)
    return frame


@app.on_event("startup")
def startup() -> None:
    CAPTURES_DIR.mkdir(exist_ok=True)
    SNAPSHOTS_DIR.mkdir(exist_ok=True)
    _get_detector()
    _get_face_rec()
    print("Detector y FaceRecognizer cargados")


def _save_capture(frame: np.ndarray, labels: list[str]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    tag = ".".join(labels)
    filename = f"{tag}_{ts}.jpg"
    path = CAPTURES_DIR / filename
    cv2.imwrite(str(path), frame)
    print(f"se captura [{tag.upper()}] {path}")
    return str(path)


@app.post("/detect")
async def detect(file: UploadFile) -> list[dict]:
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Could not decode image"}

    detections = _get_detector().detect(frame)

    face_rec = _get_face_rec()
    if face_rec:
        detections = face_rec.recognize(frame, detections)

    save_labels = [d.label for d in detections if d.label in SAVE_LABELS]
    if save_labels:
        _save_capture(frame, save_labels)
    else:
        _save_capture(frame, ["no_detectada"])

    return [
        {
            "label": d.label,
            "confidence": round(d.confidence, 4),
            "bbox": list(d.bbox),
            "identity": d.identity,
            "face_confidence": d.face_confidence,
            "face_bbox": list(d.face_bbox) if d.face_bbox else None,
            "occlusion": d.occlusion,
        }
        for d in detections
    ]


@app.get("/snapshot")
async def snapshot(camera: str = "balcon", detect: bool = False):
    """Capture a snapshot from an RTSP camera. Optionally run detection."""
    if camera not in RTSP_CAMERAS:
        return {"error": f"Camera '{camera}' not found", "available": list(RTSP_CAMERAS.keys())}

    frame = _capture_rtsp_frame(RTSP_CAMERAS[camera])
    if frame is None:
        return {"error": f"Could not capture frame from '{camera}'"}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{camera}_{ts}.jpg"
    path = SNAPSHOTS_DIR / filename

    result = {"camera": camera, "file": str(path), "timestamp": ts}

    if detect:
        detections = _get_detector().detect(frame)
        face_rec = _get_face_rec()
        if face_rec:
            detections = face_rec.recognize(frame, detections)
        result["detections"] = [
            {
                "label": d.label,
                "confidence": round(d.confidence, 4),
                "bbox": list(d.bbox),
                "identity": d.identity,
            }
            for d in detections
        ]
        # Draw bounding boxes on detected objects
        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            color = {"person": (0, 0, 255), "cat": (255, 0, 0), "dog": (0, 165, 255)}.get(
                d.label, (0, 255, 0)
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{d.label} {d.confidence:.0%}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Solo guardar si hay detecciones relevantes
        save_targets = {"person", "cat", "dog", "face"}
        if any(d.label in save_targets for d in detections):
            cv2.imwrite(str(path), frame)
            logger.info("Snapshot saved: %s", path)
            result["saved"] = True
        else:
            result["saved"] = False

    return result


@app.get("/snapshot/image")
async def snapshot_image(camera: str = "balcon"):
    """Capture and return the image directly."""
    if camera not in RTSP_CAMERAS:
        return {"error": f"Camera '{camera}' not found"}

    frame = _capture_rtsp_frame(RTSP_CAMERAS[camera])
    if frame is None:
        return {"error": f"Could not capture frame from '{camera}'"}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{camera}_{ts}.jpg"
    path = SNAPSHOTS_DIR / filename
    cv2.imwrite(str(path), frame)

    return FileResponse(str(path), media_type="image/jpeg", filename=filename)


@app.post("/enroll")
async def enroll(name: str, camera: str = "sala"):
    """Capture a frame and enroll the face with the given name."""
    if camera not in RTSP_CAMERAS:
        return {"error": f"Camera '{camera}' not found"}

    face_rec = _get_face_rec()
    if not face_rec:
        return {"error": "Face recognition is not enabled"}

    frame = _capture_rtsp_frame(RTSP_CAMERAS[camera])
    if frame is None:
        return {"error": f"Could not capture frame from '{camera}'"}

    success = face_rec.enroll_from_frame(name, frame)
    if not success:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_path = SNAPSHOTS_DIR / f"enroll_fail_{name}_{ts}.jpg"
        cv2.imwrite(str(debug_path), frame)
        return {"error": "No face detected in frame", "debug_image": str(debug_path)}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    enrolled_path = SNAPSHOTS_DIR / f"enrolled_{name}_{ts}.jpg"
    cv2.imwrite(str(enrolled_path), frame)
    return {"ok": True, "name": name, "image": str(enrolled_path)}


@app.post("/enroll/upload")
async def enroll_upload(name: str, files: list[UploadFile] = File(...)):
    """Enroll faces from one or multiple uploaded images."""
    face_rec = _get_face_rec()
    if not face_rec:
        return {"error": "Face recognition is not enabled"}

    results = []
    enrolled = 0
    for f in files:
        data = await f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            results.append({"file": f.filename, "ok": False, "reason": "could not decode"})
            continue
        if face_rec.enroll_from_frame(name, frame):
            enrolled += 1
            results.append({"file": f.filename, "ok": True})
        else:
            results.append({"file": f.filename, "ok": False, "reason": "no face detected"})

    return {"name": name, "enrolled": enrolled, "total": len(files), "details": results}


@app.get("/faces")
async def list_faces():
    """List all enrolled faces."""
    face_rec = _get_face_rec()
    if not face_rec:
        return {"error": "Face recognition is not enabled"}
    persons = face_rec.db.list_persons()
    return {"faces": [{"name": name, "embeddings": count} for name, count in persons]}


@app.delete("/faces/{name}")
async def delete_face(name: str):
    """Remove an enrolled face."""
    face_rec = _get_face_rec()
    if not face_rec:
        return {"error": "Face recognition is not enabled"}
    removed = face_rec.db.remove_person(name)
    if not removed:
        return {"error": f"Person '{name}' not found"}
    return {"ok": True, "removed": name}


@app.get("/cameras")
async def list_cameras():
    """List available cameras."""
    return {"cameras": list(RTSP_CAMERAS.keys())}


def run() -> None:
    uvicorn.run("thinking_buildings.server:app", host="0.0.0.0", port=8700)
