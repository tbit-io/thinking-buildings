#!/usr/bin/env python3
"""Thinking Buildings — building security with ML detection."""

from __future__ import annotations

import argparse
import time

from thinking_buildings._version import __version__
from thinking_buildings.alerter import Alerter
from thinking_buildings.capture import ThreadedCapture
from thinking_buildings.config import load_config
from thinking_buildings.detector import Detector
from thinking_buildings.display import Display
from thinking_buildings.events import EventBus
from thinking_buildings.logger_setup import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="thinking-buildings",
        description="Building security with ML detection",
    )
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    parser.add_argument(
        "--version", "-V",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger = setup_logging(cfg.logging)
    logger.info("Starting Thinking Buildings")

    event_bus = EventBus()

    alerter = Alerter(cfg.alerter)
    event_bus.subscribe(alerter.handle)

    # Set up captures and displays for each camera
    captures: dict[str, ThreadedCapture] = {}
    displays: dict[str, Display] = {}
    for cam_cfg in cfg.cameras:
        cam_id = cam_cfg.id
        logger.info("Starting camera '%s' (source: %s)", cam_id, cam_cfg.source)
        captures[cam_id] = ThreadedCapture(cam_cfg).start()
        if cfg.display.show_window:
            displays[cam_id] = Display(cfg.display)

    detector = Detector(cfg.detector)

    face_recognizer = None
    if cfg.face_recognition.enabled:
        from thinking_buildings.face_recognizer import FaceRecognizer
        face_recognizer = FaceRecognizer(cfg.face_recognition)
        face_recognizer.enroll_from_directory(cfg.face_recognition.faces_dir)

    logger.info(
        "Pipeline ready — %d camera(s) — press 'q' to quit", len(captures)
    )

    frame_count = 0
    fps_start = time.monotonic()
    FPS_LOG_INTERVAL = 10.0
    quit_requested = False

    try:
        while not quit_requested:
            any_frame = False
            for cam_id, capture in captures.items():
                frame = capture.read()
                if frame is None:
                    continue

                any_frame = True
                detections = detector.detect(frame)
                if face_recognizer:
                    detections = face_recognizer.recognize(frame, detections)
                for det in detections:
                    det.camera_id = cam_id
                event_bus.publish(detections)

                if cam_id in displays:
                    display = displays[cam_id]
                    display.handle(detections)
                    frame = display.render(frame)
                    key = display.show(frame, window_name=cam_id)
                    if key == ord("q"):
                        quit_requested = True
                        break

            if not any_frame:
                time.sleep(0.001)

            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= FPS_LOG_INTERVAL:
                logger.info("Pipeline throughput: %.1f FPS", frame_count / elapsed)
                frame_count = 0
                fps_start = time.monotonic()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        for capture in captures.values():
            capture.release()
        if face_recognizer:
            face_recognizer.close()
        if displays:
            Display.cleanup()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
