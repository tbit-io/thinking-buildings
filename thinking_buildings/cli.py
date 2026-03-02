#!/usr/bin/env python3
"""Thinking Buildings — building security with ML detection."""

from __future__ import annotations

import time

from thinking_buildings.config import load_config
from thinking_buildings.logger_setup import setup_logging
from thinking_buildings.capture import ThreadedCapture
from thinking_buildings.detector import Detector
from thinking_buildings.events import EventBus
from thinking_buildings.alerter import Alerter
from thinking_buildings.display import Display


def main() -> None:
    cfg = load_config()
    logger = setup_logging(cfg.logging)
    logger.info("Starting Thinking Buildings")

    event_bus = EventBus()

    alerter = Alerter(cfg.alerter)
    event_bus.subscribe(alerter.handle)

    display = Display(cfg.display) if cfg.display.show_window else None
    if display:
        event_bus.subscribe(display.handle)

    capture = ThreadedCapture(cfg.camera).start()
    detector = Detector(cfg.detector)

    face_recognizer = None
    if cfg.face_recognition.enabled:
        from thinking_buildings.face_recognizer import FaceRecognizer
        face_recognizer = FaceRecognizer(cfg.face_recognition)
        face_recognizer.enroll_from_directory(cfg.face_recognition.faces_dir)

    logger.info("Pipeline ready — press 'q' to quit")

    frame_count = 0
    fps_start = time.monotonic()
    FPS_LOG_INTERVAL = 10.0

    try:
        while True:
            frame = capture.read()
            if frame is None:
                time.sleep(0.001)
                continue

            detections = detector.detect(frame)
            if face_recognizer:
                detections = face_recognizer.recognize(frame, detections)
            event_bus.publish(detections)

            frame_count += 1
            elapsed = time.monotonic() - fps_start
            if elapsed >= FPS_LOG_INTERVAL:
                logger.info("Pipeline throughput: %.1f FPS", frame_count / elapsed)
                frame_count = 0
                fps_start = time.monotonic()

            if display:
                frame = display.render(frame)
                key = display.show(frame)
                if key == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        capture.release()
        if face_recognizer:
            face_recognizer.close()
        if display:
            display.cleanup()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
