"""Thinking Buildings — FastAPI server exposing the detection pipeline."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response

from thinking_buildings.capture import ThreadedCapture
from thinking_buildings.config import AppConfig, load_config
from thinking_buildings.detector import Detector
from thinking_buildings.events import Detection
from thinking_buildings.logger_setup import setup_logging

logger = logging.getLogger("thinking_buildings")


class ServerState:
    """Holds shared pipeline objects for the server lifetime."""

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.captures: dict[str, ThreadedCapture] = {}
        self.detector: Optional[Detector] = None
        self.face_recognizer = None

    def startup(self) -> None:
        setup_logging(self.cfg.logging)
        logger.info("Starting server with %d camera(s)", len(self.cfg.cameras))

        self.detector = Detector(self.cfg.detector)

        if self.cfg.face_recognition.enabled:
            from thinking_buildings.face_recognizer import FaceRecognizer

            self.face_recognizer = FaceRecognizer(self.cfg.face_recognition)
            self.face_recognizer.enroll_from_directory(self.cfg.face_recognition.faces_dir)

        for cam_cfg in self.cfg.cameras:
            logger.info("Starting camera '%s' (source: %s)", cam_cfg.id, cam_cfg.source)
            self.captures[cam_cfg.id] = ThreadedCapture(cam_cfg).start()

        logger.info("Server ready")

    def shutdown(self) -> None:
        for capture in self.captures.values():
            capture.release()
        if self.face_recognizer:
            self.face_recognizer.close()
        logger.info("Server shutdown complete")

    def get_frame(self, camera_id: str) -> np.ndarray:
        if camera_id not in self.captures:
            raise KeyError(camera_id)
        frame = self.captures[camera_id].read()
        if frame is None:
            raise RuntimeError(f"No frame available from camera '{camera_id}'")
        return frame

    def detect_frame(self, frame: np.ndarray, camera_id: str = "") -> list[dict]:
        detections = self.detector.detect(frame)
        if self.face_recognizer:
            detections = self.face_recognizer.recognize(frame, detections)
        for det in detections:
            det.camera_id = camera_id
        return [_detection_to_dict(d) for d in detections]


def _detection_to_dict(det: Detection) -> dict:
    d = asdict(det)
    d["bbox"] = list(d["bbox"])
    if d["face_bbox"]:
        d["face_bbox"] = list(d["face_bbox"])
    return d


def _frame_to_jpeg(frame: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("Failed to encode frame as JPEG")
    return buf.tobytes()


def _read_upload_as_frame(contents: bytes) -> np.ndarray:
    arr = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Could not decode image")
    return frame


def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Factory that creates the FastAPI app with proper lifespan."""
    cfg = load_config(config_path)
    state = ServerState(cfg)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.startup()
        yield
        state.shutdown()

    app = FastAPI(
        title="Thinking Buildings",
        description="Building security detection API",
        lifespan=lifespan,
    )
    app.state.pipeline = state

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "cameras": len(state.captures),
            "face_recognition": state.face_recognizer is not None,
        }

    @app.get("/cameras")
    async def list_cameras():
        return [
            {
                "id": cam.id,
                "location": cam.location,
                "source_type": "rtsp" if isinstance(cam.source, str) else "local",
                "width": cam.width,
                "height": cam.height,
            }
            for cam in cfg.cameras
        ]

    @app.get("/cameras/{camera_id}/snapshot")
    async def snapshot(camera_id: str):
        try:
            frame = state.get_frame(camera_id)
        except KeyError:
            raise HTTPException(404, f"Camera '{camera_id}' not found")
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        return Response(content=_frame_to_jpeg(frame), media_type="image/jpeg")

    @app.post("/cameras/{camera_id}/detect")
    async def detect_camera(camera_id: str):
        try:
            frame = state.get_frame(camera_id)
        except KeyError:
            raise HTTPException(404, f"Camera '{camera_id}' not found")
        except RuntimeError as e:
            raise HTTPException(503, str(e))
        return {"camera_id": camera_id, "detections": state.detect_frame(frame, camera_id)}

    @app.post("/detect")
    async def detect_upload(file: UploadFile):
        contents = await file.read()
        try:
            frame = _read_upload_as_frame(contents)
        except ValueError:
            raise HTTPException(400, "Could not decode uploaded image")
        return {"detections": state.detect_frame(frame)}

    @app.post("/enroll")
    async def enroll(name: str, file: UploadFile):
        if not state.face_recognizer:
            raise HTTPException(400, "Face recognition is not enabled")
        contents = await file.read()
        try:
            frame = _read_upload_as_frame(contents)
        except ValueError:
            raise HTTPException(400, "Could not decode uploaded image")
        success = state.face_recognizer.enroll_from_frame(name, frame)
        if not success:
            raise HTTPException(422, "No face detected in the uploaded image")
        return {"enrolled": name}

    return app


def main() -> None:
    """Entry point for thinking-buildings-server."""
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(
        prog="thinking-buildings-server",
        description="Thinking Buildings detection API server",
    )
    parser.add_argument(
        "--config", "-c", default="config.yaml", help="Path to config file"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", "-p", type=int, default=8080, help="Port")
    args = parser.parse_args()

    app = create_app(args.config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
