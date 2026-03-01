# Thinking Buildings

Building security system using cameras and ML to detect people/animals, log events, and alert on detections.

## Architecture

```
Webcam → VideoCapture → Detector (YOLO) → EventBus → Alerter (console/log)
                                              ↓
                                         Display (OpenCV window with bounding boxes)
```

**Event bus pattern**: Components subscribe to detections via callbacks. New features plug in as subscribers without changing existing code.

## Project Structure

- `run.py` — Entry point, wires the pipeline
- `thinking_buildings/config.py` — YAML config loader with dataclasses
- `thinking_buildings/events.py` — `Detection` dataclass + `EventBus`
- `thinking_buildings/capture.py` — OpenCV video capture (V4L2 + MJPG)
- `thinking_buildings/detector.py` — YOLO inference wrapper
- `thinking_buildings/alerter.py` — Console/desktop alerts with cooldown
- `thinking_buildings/display.py` — Bounding box rendering + FPS
- `thinking_buildings/logger_setup.py` — Logging configuration
- `config.yaml` — Runtime configuration

## Running

```bash
source .venv/bin/activate
python run.py    # Press 'q' to quit
```

## Environment

- Python 3.12, venv at `.venv/`
- YOLO model `yolo11n.pt` auto-downloads on first run
- WSL2 with usbipd for webcam passthrough (V4L2 + MJPG codec)
- Camera tested: EasyCam 502 on bus 1-7

## Key Conventions

- All config changes go in `config.yaml`, not in code
- New pipeline components subscribe to `EventBus` — don't modify existing subscribers
- Detection targets are configured via `config.yaml` `detector.classes`
- Alert cooldown is per-class to avoid spam
- Use `logging` module via `thinking_buildings` logger, not print statements
