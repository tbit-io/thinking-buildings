# Thinking Buildings

Building security system that uses an economy-of-attention model: lightweight frame sampling, YOLO triage, and LLM-driven reasoning to detect threats and decide actions.

Full architecture: `docs/architecture.md`

## Two-Layer Architecture

```
On-site Client                              TB Server (on-prem or cloud)
(DVR/NVR/PC)
                                            Perception Layer
Cameras → Client App ── internet ──>          YOLO (heavy model) triage
           captures at s rate                 face recognition
           buffers locally                    object classification
                ^                                     |
                |                             Thought Layer
           acceleration <────────               LLM analysis
           commands                             spatial reasoning
                                                action decisions
                                                      |
                                              Action Protocol
                                                alerts (tiered)
                                                acceleration
                                                narrative logs
```

### Current Codebase (Phase 4, multi-camera pipeline)

```
Cameras (webcam + RTSP) → ThreadedCapture (per camera) → Detector → EventBus → Subscribers
                                                                        ├── Alerter (per-camera cooldowns)
                                                                        ├── Display (per-camera windows)
                                                                        └── FaceRecognizer (known/unknown/occluded)
```

## Project Structure

- `pyproject.toml` — Package metadata, deps, entry points (hatchling + hatch-vcs)
- `config.yaml` — Runtime configuration (multi-camera `cameras:` list, all config here, not in code)
- `docs/architecture.md` — Full system architecture and roadmap
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
- `tests/` — 99 tests

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

## Contributing

- **All changes via PR** — branch protection enforced on master, no direct push
- CI must pass (lint + tests) before merge
- All config changes go in `config.yaml`, not in code
- **Never hardcode credentials** — use config or environment variables
- New pipeline components subscribe to `EventBus` — don't modify existing subscribers
- Detection targets are configured via `config.yaml` `detector.classes`
- Use `logging` module via `thinking_buildings` logger, not print statements
- Write tests for new features
- License: AGPL-3.0

## Key Conventions

- Alert cooldown is per-class to avoid spam
- Occlusion requires grace period (N consecutive frames) before alerting
- Server-side can use heavier YOLO models (no real-time constraint)
- Sampling rate `s` (seconds between frames) is per-camera, LLM-adjustable
- Tiered alerts: notify (resident) → security → emergency (police)

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

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| 1-3 | Done | Local pipeline, face rec, distribution, CI/CD |
| 4 | **In progress** | Server foundation — RTSP + multi-camera done, FastAPI + API next |
| 5 | Planned | Client application (capture daemon, frame buffer, transport) |
| 6 | Planned | Perception layer (heavy YOLO, GPU batching, triage) |
| 7 | Planned | Thought layer (LLM integration, building knowledge, decisions) |
| 8 | Planned | Action & alerting (tiered routing, narrative engine) |
| 9 | Planned | Cloud platform (TBit multi-tenant, dashboard, billing) |
