# Thinking Buildings

**Open-source building security powered by ML.** Turn any camera into an intelligent security system that detects, understands, and alerts — no expensive hardware required.

> *Born in Bogota. Designed for practical security, not surveillance theater.*

---

## Why Thinking Buildings?

Millions of security cameras around the world record 24/7 but nobody watches them. Buildings spend thousands on physical security — guards, barriers, double airlocks, construction — while the cameras they already own sit there collecting footage no one reviews until after something goes wrong.

Thinking Buildings flips this model. Instead of more hardware, **add intelligence to what you already have.** A standard webcam or IP camera becomes a detection system that identifies people, animals, and objects in real time — alerting you in seconds, not the next morning.

### The problem we're solving

- **Footage without response.** A camera that only records is useful after an incident, not during one.
- **Monitoring is expensive.** Constant human monitoring doesn't scale for residential buildings, small businesses, farms, and remote sites.
- **Commercial platforms are closed and costly.** Most systems require proprietary hardware, recurring subscriptions, or cloud lock-in.

Thinking Buildings is **free, open-source, and runs on hardware you already own.**

---

## How it works

```
Camera -> VideoCapture -> Detector (YOLO) -> EventBus -> Subscribers
                                                ├── Alerter (console/desktop/log)
                                                ├── Display (bounding boxes + FPS)
                                                ├── [Your custom subscriber]
                                                └── [IoT sensors, webhooks, etc.]
```

The **Event Bus architecture** is the core design principle. Every component is a subscriber that reacts to detections. Want to add Telegram alerts? Write a subscriber. Want to log to a database? Write a subscriber. The existing code never changes.

---

## Quick start

### Requirements

- Python 3.10+
- A webcam or IP camera
- A machine with at least 4GB RAM (Raspberry Pi 4+ works)

### Installation

```bash
git clone https://github.com/tbit-io/thinking-buildings.git
cd thinking-buildings
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Run

```bash
thinking-buildings       # installed entry point
# or
python run.py            # direct script (backward compatible)
# Press 'q' to quit
```

The YOLO model downloads automatically on first run to `~/.thinking-buildings/models/`.

### Configuration

All settings live in `config.yaml` — never in code:

```yaml
camera:
  source: 0            # 0 for USB webcam, -1 for auto-detect
  width: 640
  height: 480

detector:
  model: yolo11n.pt    # Auto-downloads on first run
  confidence: 0.5
  backend: auto        # auto, ultralytics, or onnx
  classes:
    - person
    - dog
    - cat
    - bird

alerter:
  cooldown_seconds: 30
  desktop_notifications: true
```

---

## Use cases

### Residential buildings and gated communities
Replace or augment overnight security shifts with intelligent detection. Get alerts when someone is in the parking garage at 3am — instead of reviewing footage the next day.

### Small businesses and shops
Monitor after-hours activity with a basic camera and receive alerts without paying for commercial monitoring services.

### Farms and rural properties
Detect people or animals in areas where physical security is impractical. Know when someone approaches a remote entrance or when livestock is in the wrong area.

### Schools and public spaces
Monitor common areas for unusual activity during off-hours without expensive commercial platforms.

### Self-hosted home security
For people who want smart security without sending their video to the cloud. Thinking Buildings runs 100% locally — your data stays yours.

---

## Current capabilities

- Real-time object detection (YOLO) with configurable classes
- Face recognition — known vs unknown persons, occlusion detection
- Threaded video capture (latest-frame-wins, no stale frames)
- Backend abstraction — auto-selects best inference engine for your hardware
- Camera auto-detection and resolution negotiation
- Model auto-download and caching (`~/.thinking-buildings/models/`)
- Event bus architecture for decoupled, extensible components
- Alert cooldowns (per-class, with special handling for unknown/occluded faces)
- Console, log file, and desktop notifications
- Live display with bounding boxes, identity labels, and FPS counter
- YAML-based runtime configuration
- Installable Python package with CLI entry point

---

## Project structure

```
thinking-buildings/
├── pyproject.toml                        # Package metadata and dependencies
├── config.yaml                           # Runtime configuration
├── run.py                                # Backward-compatible entry point
├── thinking_buildings/
│   ├── cli.py                            # Main entry point
│   ├── config.py                         # YAML config loader with dataclasses
│   ├── events.py                         # Detection dataclass + thread-safe EventBus
│   ├── capture.py                        # VideoCapture + ThreadedCapture
│   ├── detector.py                       # Detection with backend abstraction
│   ├── alerter.py                        # Console/desktop alerts with cooldown
│   ├── display.py                        # Bounding box rendering + FPS
│   ├── face_recognizer.py                # Face recognition pipeline
│   ├── face_db.py                        # SQLite face embedding database
│   ├── camera_probe.py                   # Camera enumeration + resolution
│   ├── model_manager.py                  # Model download + cache
│   ├── logger_setup.py                   # Logging configuration
│   └── backends/
│       ├── base.py                       # InferenceBackend ABC
│       ├── ultralytics_backend.py        # YOLO via ultralytics
│       └── __init__.py                   # Backend factory (auto-detection)
└── tests/                                # 92 tests
```

---

## Roadmap

### Phase 1: MVP — Done
Webcam capture, YOLO detection, event bus, alerting with cooldowns, live display, YAML config.

### Phase 2: Face recognition — Done
Known vs unknown person identification, occlusion detection, SQLite face embedding database, per-identity alerting.

### Phase 3: Distribution and performance — In progress
- **3A: Foundation — Done.** Package setup (pyproject.toml, CLI entry point), threaded capture, backend abstraction, camera auto-detection, model manager.
- **3B: Installers — Next.** PyInstaller spec, Windows MSI installer, Linux AppImage, GitHub Actions CI/CD.

### Phase 4: Alerting and persistence
Event persistence (SQLite/PostgreSQL), snapshot capture on detection, alert channels (email, Telegram, webhook), alert scheduling, web dashboard.

### Phase 5: IoT sensor integration
PIR motion sensors, magnetic door/window contacts, vibration sensors as EventBus publishers. ESP32/Raspberry Pi GPIO. MQTT broker support.

### Phase 6: Community network (vision)
Opt-in network where nearby installations share anonymized detection metadata. Neighborhood-level security awareness without sharing video. Cooperative alerting between buildings. All data anonymized and privacy-preserving.

### Phase 7: Production hardening
Systemd service / Docker deployment, health monitoring, GPU acceleration (CUDA / TensorRT), multi-node coordination, authentication and access control.

---

## Community network: the bigger vision

A single Thinking Buildings installation makes one building smarter. But when buildings **talk to each other**, you get something fundamentally different: a community security network.

Imagine a neighborhood where 20 buildings run Thinking Buildings. Each one independently detects and alerts. But together, they can:

- **Identify patterns** that no single building can see
- **Share behavioral baselines** — the network learns that foot traffic at 7am is normal but 3am is unusual
- **Reduce false positives collectively** — if every building alerts on stray cats at dawn, the network deprioritizes that pattern
- **Create a real-time awareness layer** — "the neighborhood is quiet tonight" or "elevated activity in the north zone"

**Privacy is non-negotiable.** The network shares only anonymized detection metadata (timestamps, detection types, confidence levels, zone identifiers). No video, no images, no personal data ever leaves a building. Each installation is sovereign. Opting in is voluntary.

This isn't surveillance. It's **community awareness.**

---

## TBit services (optional, commercial)

[TBit](https://tbit.io) is the company behind Thinking Buildings. The core platform is free and open-source forever under AGPL-3.0. TBit offers optional commercial services for organizations that need more:

- **LLM-powered event reasoning** — contextual analysis of detection patterns
- **WhatsApp / SMS / Telegram alerts** with intelligent context
- **Building security scoring** — quantified vulnerability assessments
- **Dashboard and reports** for property management
- **Multi-sensor correlation engine** — advanced cross-signal analysis
- **Implementation and support** — setup, tuning, and ongoing maintenance
- **Commercial license** — for organizations that need to embed Thinking Buildings in proprietary products without AGPL obligations

You don't need TBit to use Thinking Buildings. But if you want the intelligence layer on top, we're here.

---

## Contributing

We welcome contributions from anywhere in the world. Whether you're a developer, a student, or a homelab enthusiast — if you want to help make buildings safer, you're welcome here.

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Follow the conventions in `CLAUDE.md`
4. Submit a PR with a clear description of what and why

### Conventions

- All config changes go in `config.yaml`, not in code
- New pipeline components subscribe to `EventBus` — don't modify existing subscribers
- Detection targets are configured via `config.yaml` `detector.classes`
- Alert cooldown is per-class to avoid spam
- Use `logging` module via `thinking_buildings` logger, not print statements

### Ideas for contributions

- RTSP support for IP camera streams
- Telegram/Discord bot subscriber
- SQLite event logger for detection history
- MQTT sensor subscriber for IoT integration
- Web dashboard for browsing events
- Detection analytics and time-based charts
- Translations for docs in more languages
- Bug reports and testing on different hardware/OS

---

## Environment notes

- Python 3.10+ with venv
- YOLO model `yolo11n.pt` auto-downloads on first run (~6MB)
- Works on Linux, macOS, and Windows
- WSL2 with usbipd supported for webcam passthrough
- Raspberry Pi 4+ with USB camera is a valid low-cost deployment target

---

## License

Thinking Buildings is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

You are free to use, modify, and distribute this software. If you run a modified version as a network service, you must make your modifications available under the same license.

For commercial licensing options (embedding in proprietary products without AGPL obligations), contact [TBit](https://tbit.io).

---

## About

Thinking Buildings is an open-source project by [TBit](https://tbit.io), a Colombian technology company. We believe security should be accessible to everyone — not just those who can afford expensive commercial platforms.

If Thinking Buildings helps make your building, your neighborhood, or your city a little safer, that's the whole point.
