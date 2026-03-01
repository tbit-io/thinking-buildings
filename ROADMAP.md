# Thinking Buildings - Roadmap

## Phase 1: MVP (Complete)

- [x] Webcam capture with OpenCV
- [x] YOLO object detection (person, dog, cat, bird)
- [x] Event bus architecture for decoupled components
- [x] Console + log file alerting with per-class cooldown
- [x] Live display with bounding boxes and FPS counter
- [x] YAML-based configuration
- [x] Optional desktop notifications

## Phase 2: Smarter Detection

- [ ] Face recognition (known vs unknown persons)
- [ ] Multi-camera support
- [ ] Detection zones (define regions of interest per camera)
- [ ] Object tracking across frames (reduce duplicate alerts)
- [ ] Confidence auto-tuning based on lighting conditions

## Phase 3: Alerting and Logging

- [ ] Event persistence (SQLite or PostgreSQL)
- [ ] Snapshot capture on detection (save frames with bounding boxes)
- [ ] Alert channels: email, Telegram, webhook
- [ ] Alert scheduling (e.g., only alert after hours)
- [ ] Event dashboard (web UI for browsing detection history)

## Phase 4: Intelligence

- [ ] LLM-powered event reasoning ("person at back door at 2am — unusual")
- [ ] Behavioral pattern detection (loitering, repeated visits)
- [ ] Spatial mapping (associate detections with building zones)
- [ ] Anomaly scoring based on time, location, and frequency

## Phase 5: Production Hardening

- [ ] Systemd service / Docker container deployment
- [ ] Health monitoring and auto-restart
- [ ] GPU acceleration (CUDA / TensorRT)
- [ ] RTSP stream input support
- [ ] Multi-node coordination for large buildings
- [ ] Authentication and access control for dashboard
