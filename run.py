#!/usr/bin/env python3
"""Thinking Buildings — building security with ML detection."""

import sys

from thinking_buildings.config import load_config
from thinking_buildings.logger_setup import setup_logging
from thinking_buildings.capture import VideoCapture
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

    capture = VideoCapture(cfg.camera)
    detector = Detector(cfg.detector)

    logger.info("Pipeline ready — press 'q' to quit")

    try:
        while True:
            frame = capture.read()
            if frame is None:
                logger.warning("Failed to read frame, retrying...")
                continue

            detections = detector.detect(frame)
            event_bus.publish(detections)

            if display:
                frame = display.render(frame)
                key = display.show(frame)
                if key == ord("q"):
                    break
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        capture.release()
        if display:
            display.cleanup()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
