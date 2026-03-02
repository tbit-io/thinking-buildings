# Thinking Buildings

Building security system using cameras and ML to detect people/animals, recognize faces, log events, and alert on detections.

## Architecture

```
Camera → ThreadedCapture → Detector (backend abstraction) → EventBus → Subscribers
                                                                ├── Alerter (console/desktop/log)
                                                                ├── Display (bounding boxes + FPS)
                                                                └── FaceRecognizer (known/unknown/occluded)
```

**Event bus pattern**: Components subscribe to detections via callbacks. New features plug in as subscribers without changing existing code.

**Backend abstraction**: `get_backend("auto")` probes available inference engines and picks the best one. Currently supports ultralytics. ONNX backend planned.

**Threaded capture**: `ThreadedCapture` runs `cap.read()` in a daemon thread, overwrites a single frame under a lock (latest-frame-wins). Main thread never blocks on camera I/O.

## Project Structure

- `pyproject.toml` — Package metadata, deps, entry points (hatchling + hatch-vcs)
- `config.yaml` — Runtime configuration
- `run.py` — Backward-compatible entry point (shim to `cli.main()`)
- `thinking_buildings/cli.py` — Main entry point with `ThreadedCapture`
- `thinking_buildings/config.py` — YAML config loader with dataclasses
- `thinking_buildings/events.py` — `Detection` dataclass + thread-safe `EventBus`
- `thinking_buildings/capture.py` — `VideoCapture` + `ThreadedCapture`
- `thinking_buildings/detector.py` — Detection with backend abstraction
- `thinking_buildings/alerter.py` — Console/desktop alerts with cooldown
- `thinking_buildings/display.py` — Bounding box rendering + FPS
- `thinking_buildings/face_recognizer.py` — Face recognition pipeline
- `thinking_buildings/face_db.py` — SQLite face embedding database
- `thinking_buildings/camera_probe.py` — Camera enumeration + resolution negotiation
- `thinking_buildings/model_manager.py` — Model download + cache (`~/.thinking-buildings/models/`)
- `thinking_buildings/logger_setup.py` — Logging configuration
- `thinking_buildings/backends/` — Inference backend abstraction (base, ultralytics, factory)
- `tests/` — 92 tests

## Running

```bash
source .venv/bin/activate
pip install -e ".[dev]"
thinking-buildings       # installed entry point
# or
python run.py            # backward compatible
# Press 'q' to quit
```

## Environment

- Python 3.10+, venv at `.venv/`
- YOLO model auto-downloads to `~/.thinking-buildings/models/` on first run
- Works on Linux (V4L2), Windows (DirectShow), macOS (default backend)
- Camera auto-detection with `source: -1` in config

## Key Conventions

- All config changes go in `config.yaml`, not in code
- New pipeline components subscribe to `EventBus` — don't modify existing subscribers
- Detection targets are configured via `config.yaml` `detector.classes`
- Alert cooldown is per-class to avoid spam
- Occlusion requires grace period (N consecutive frames) before alerting
- Use `logging` module via `thinking_buildings` logger, not print statements
- License: AGPL-3.0

## Detection Thresholds

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `detector.confidence` | 0.65 | YOLO minimum confidence |
| `face_recognition.recognition_threshold` | 0.5 | Cosine similarity for identity match |
| `face_recognition.occlusion_det_threshold` | 0.7 | Face det score below this = possible occlusion |
| `face_recognition.occlusion_grace_frames` | 5 | Consecutive frames without face before flagging |
| `alerter.cooldown_seconds` | 30 | Default alert cooldown |
| Occlusion cooldown | min(15, cooldown) | Hardcoded in alerter |
| Unknown person cooldown | cooldown / 2 | Hardcoded in alerter |
