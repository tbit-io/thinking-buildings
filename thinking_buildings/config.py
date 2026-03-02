from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class CameraConfig:
    source: int = 0
    width: int = 1280
    height: int = 720


@dataclass
class DetectorConfig:
    model: str = "yolo11n.pt"
    confidence: float = 0.65
    classes: List[str] = field(default_factory=lambda: ["person", "dog", "cat", "bird"])
    backend: str = "auto"


@dataclass
class AlerterConfig:
    cooldown_seconds: int = 30
    desktop_notifications: bool = True


@dataclass
class DisplayConfig:
    show_window: bool = True
    show_fps: bool = True
    box_color: Tuple[int, int, int] = (0, 255, 0)
    box_thickness: int = 2


@dataclass
class FaceRecConfig:
    enabled: bool = True
    db_path: str = "data/faces.db"
    faces_dir: str = "faces/"
    recognition_threshold: float = 0.5
    occlusion_det_threshold: float = 0.7
    occlusion_grace_frames: int = 5
    model_name: str = "buffalo_l"
    det_size: Tuple[int, int] = (640, 640)


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_file: str = "logs/detections.log"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    alerter: AlerterConfig = field(default_factory=AlerterConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    face_recognition: FaceRecConfig = field(default_factory=FaceRecConfig)


def load_config(path: str = "config.yaml") -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    camera = CameraConfig(**raw.get("camera", {}))
    detector = DetectorConfig(**raw.get("detector", {}))
    alerter = AlerterConfig(**raw.get("alerter", {}))

    display_raw = raw.get("display", {})
    if "box_color" in display_raw:
        display_raw["box_color"] = tuple(display_raw["box_color"])
    display = DisplayConfig(**display_raw)

    logging_cfg = LoggingConfig(**raw.get("logging", {}))

    face_rec_raw = raw.get("face_recognition", {})
    if "det_size" in face_rec_raw:
        face_rec_raw["det_size"] = tuple(face_rec_raw["det_size"])
    face_rec = FaceRecConfig(**face_rec_raw)

    return AppConfig(
        camera=camera,
        detector=detector,
        alerter=alerter,
        display=display,
        logging=logging_cfg,
        face_recognition=face_rec,
    )
