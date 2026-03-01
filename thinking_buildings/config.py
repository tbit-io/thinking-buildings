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
    confidence: float = 0.5
    classes: List[str] = field(default_factory=lambda: ["person", "dog", "cat", "bird"])


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

    return AppConfig(
        camera=camera,
        detector=detector,
        alerter=alerter,
        display=display,
        logging=logging_cfg,
    )
