# Thinking Buildings — Architecture

## Vision

A building security system that **thinks about what to pay attention to**. Instead of constant high-FPS recording with dumb motion alerts, Thinking Buildings uses an economy-of-attention model: lightweight sampling, intelligent triage, and escalation only when it matters.

The system operates in two layers:
- **Perception Layer** — YOLO models filter the visual stream, surfacing only what's worth analyzing
- **Thought Layer** — An LLM reasons about detections using spatial context, time, and building knowledge to decide actions

## Design Principles

1. **Economy of attention** — Build narratives from keyframes, not brute-force FPS. A single well-chosen frame is worth more than 30 frames of empty hallway.
2. **Progressive escalation** — Start cheap, get expensive only when justified. Default idle sampling costs nearly nothing; full analysis activates only on demand.
3. **Spatial reasoning** — The system understands *where* cameras are, what they're looking at, and what's normal for that location and time of day.
4. **Tiered response** — Not every event is an emergency. Open door at 3pm is a notification; unknown person at 3am is security; forced entry is police.

## System Overview

```
                        On-site                                    TB Server
                    (DVR/NVR/PC)                              (on-prem or cloud)

  Cameras ──> Client App ──── internet (low bandwidth) ────> Perception Layer
              - captures frames                               - YOLO triage
              - buffers locally                               - face recognition
              - sends keyframes                               - object classification
              at configured s*t rate                                  |
                     ^                                               v
                     |                                        Thought Layer
                     |                                         - LLM analysis
              acceleration <──────────────────────────────     - context reasoning
              commands                                         - action decisions
              (increase fps,                                         |
               switch cameras)                                       v
                                                              Action Protocol
                                                               - alerts (tiered)
                                                               - acceleration
                                                               - narrative logs
```

## Sampling Rate: s*t Model

Each camera has a configurable sampling interval defined by `s` (seconds between frames).

| Camera context           | Default `s` | Rationale                                   |
|--------------------------|-------------|---------------------------------------------|
| Exterior / street-facing | 10          | Low baseline, wide field, most activity is transient |
| Entry points (doors)     | 5           | Higher value targets, shorter event windows  |
| Interior common areas    | 15          | Controlled access, lower threat              |
| Perimeter / parking      | 10          | Vehicle and person detection at distance     |

These defaults are starting points. The Thought Layer adjusts `s` dynamically:
- Detection of a person → reduce `s` to 2 on that camera
- Sustained no-activity → increase `s` to save bandwidth
- Time-of-day: night hours may halve `s` for exterior cameras
- Correlated events: activity on one camera reduces `s` on adjacent cameras

## Perception Layer

### Why heavier models work here

Unlike the current edge-only pipeline (yolo11n on webcam), the server-side architecture removes the real-time constraint. This unlocks:

- **Larger YOLO models** (yolov8x, yolo11l) for higher accuracy on low-quality inputs — low resolution RTSP streams, bad lighting, distant subjects
- **Longer inference time is acceptable** since we're processing 1 frame every `s` seconds, not 30 FPS
- **GPU batching** across multiple cameras — process keyframes from N cameras in a single batch
- **Specialized models** activated on demand: OCR for license plates, face recognition for identity, pose estimation for behavior

### Triage logic

The Perception Layer's job is **heavy filtering**. The LLM should only see frames that contain something worth reasoning about.

```
Frame arrives
    |
    v
YOLO detection (heavy model, high accuracy)
    |
    ├── Nothing detected → log, discard, maintain s
    ├── Known benign (resident face, regular vehicle) → log, no escalation
    └── Detection worth reasoning about → forward to Thought Layer
        - Unknown person
        - Person in unexpected area/time
        - Object left behind
        - Vehicle not in known list
        - Anomaly (door open, window broken)
        - Animal in restricted area
        - Multiple people where usually none
```

### What gets forwarded to the Thought Layer

Not raw frames — structured detection reports:

```json
{
  "camera_id": "entrance_main",
  "timestamp": "2026-03-04T23:15:00Z",
  "frame_ref": "s3://bucket/frames/entrance_main_20260304_231500.jpg",
  "sampling_rate_s": 5,
  "detections": [
    {
      "label": "person",
      "confidence": 0.92,
      "bbox": [120, 50, 340, 480],
      "identity": "unknown_person",
      "face_confidence": 0.3,
      "attributes": {
        "carrying_object": true,
        "moving_direction": "toward_door"
      }
    }
  ],
  "scene_context": {
    "time_of_day": "night",
    "lighting": "low",
    "recent_activity": "none_last_30min"
  }
}
```

## Thought Layer

The LLM operates with persistent context about the building:

### Building Knowledge Base

```yaml
building:
  name: "Edificio Central"
  type: residential
  floors: 5
  units: 20

cameras:
  entrance_main:
    location: "Main entrance, ground floor"
    faces: "street, pedestrian sidewalk"
    field_of_view: "door and 3m of sidewalk"
    normal_activity: "residents entering/exiting, deliveries 9am-6pm"
    default_s: 5

  parking_north:
    location: "North parking lot, ground floor"
    faces: "parking area, 12 spaces"
    field_of_view: "full lot, entry ramp"
    normal_activity: "vehicle movement 7-9am and 5-8pm"
    default_s: 10

  hallway_3rd:
    location: "3rd floor hallway"
    faces: "corridor between units 301-305"
    field_of_view: "full hallway, elevator door"
    normal_activity: "residents passing, low traffic"
    default_s: 15

  perimeter_east:
    location: "East wall exterior"
    faces: "back alley, service entrance"
    field_of_view: "wall, dumpsters, alley"
    normal_activity: "garbage collection Tue/Fri 6am"
    default_s: 10

known_persons:
  - name: "Javier"
    role: resident
    unit: 401
  - name: "Maria"
    role: doorman
    schedule: "6am-2pm weekdays"

known_vehicles:
  - plate: "ABC123"
    owner: "Javier"
    type: "sedan"
```

### LLM Decision Framework

When the Thought Layer receives a detection report, it reasons about:

1. **What** — What was detected and how confident are we?
2. **Where** — Which camera, what does it look at, what's normal there?
3. **When** — Time of day, day of week, is this expected?
4. **Who** — Known person, unknown, occluded face?
5. **Pattern** — Is this part of a sequence? Related to other recent events?

### Action Decisions

The LLM outputs structured actions:

```json
{
  "threat_level": "medium",
  "reasoning": "Unknown person at main entrance at 2:30am. No scheduled deliveries. Person appears to be examining the door lock.",
  "actions": [
    {
      "type": "accelerate",
      "camera_id": "entrance_main",
      "new_s": 1,
      "duration_s": 120
    },
    {
      "type": "attention",
      "camera_ids": ["parking_north", "perimeter_east"],
      "new_s": 3,
      "duration_s": 60
    },
    {
      "type": "request_history",
      "camera_id": "entrance_main",
      "lookback_s": 120,
      "max_frames": 12
    },
    {
      "type": "alert",
      "target": "security",
      "message": "Unknown person examining main entrance lock at 2:30am",
      "attachments": ["frame_ref"]
    }
  ]
}
```

### Action Types

| Action | Description | Example trigger |
|--------|-------------|-----------------|
| `accelerate` | Increase frame rate on a specific camera | Person detected, need more detail |
| `attention` | Increase frame rate on adjacent/related cameras | Activity at entrance → watch parking too |
| `request_history` | Pull recent keyframes from client buffer | Build narrative of how person arrived |
| `detect_specific` | Run specialized model (OCR, face, pose) | Read license plate, identify person |
| `alert:notify` | Push notification to resident/owner | Door left open, package delivered |
| `alert:security` | Escalate to security company | Unknown person lingering, suspicious behavior |
| `alert:emergency` | Escalate to police/fire | Forced entry, fire/smoke detected |
| `decelerate` | Reduce frame rate back to default | Threat resolved, person left |
| `log_narrative` | Store event summary with keyframes | Searchable incident history |

## Client Application

The on-site client runs on the DVR/NVR or a dedicated PC. Its responsibilities:

1. **Capture** — Pull frames from RTSP cameras at the configured `s` rate
2. **Buffer** — Keep a rolling buffer of recent frames (last N minutes) for history requests
3. **Transmit** — Send keyframes to the TB Server over internet
4. **Receive** — Accept commands from server (change `s`, send history, switch models)
5. **Local fallback** — If internet is down, run lightweight YOLO locally and queue alerts

### Bandwidth budget

At 1 frame per 10s per camera, with JPEG compression (~50KB/frame):

| Cameras | Frames/min | Bandwidth    | Monthly     |
|---------|-----------|--------------|-------------|
| 4       | 24        | ~10 KB/s     | ~25 GB      |
| 8       | 48        | ~20 KB/s     | ~50 GB      |
| 16      | 96        | ~40 KB/s     | ~100 GB     |
| 16 (accelerated, s=2) | 480 | ~200 KB/s | ~500 GB  |

Acceleration is temporary — sustained high rates only during active incidents.

## Deployment Models

### On-premise

```
Cameras → DVR/NVR → TB Client (same machine or separate PC)
                         ↕ localhost
                     TB Server (GPU machine on LAN)
                         ↕ API
                     LLM (local Ollama or API key)
```

Best for: commercial buildings, government, data-sensitive installations.

### Cloud (TBit hosted)

```
Cameras → DVR/NVR → TB Client (on-site)
                         ↕ internet
                     TB Server (TBit cloud, GPU)
                         ↕
                     LLM (managed)
```

Best for: residential buildings, small businesses, multi-site management.

## Narrative Engine

Instead of traditional DVR scrubbing, the system produces **event narratives**:

```
Incident #2847 — 2026-03-04 02:30-02:35
Camera: entrance_main

02:30:00 — Unknown person appeared on sidewalk, approaching entrance
02:30:10 — Person stopped at door, examining lock area [frame]
02:30:12 — [ACCELERATED to 1fps] Security notified
02:30:15 — Adjacent cameras activated (parking_north, perimeter_east)
02:30:30 — Person moved to window, looked inside [frame]
02:31:00 — Person walked away toward east [frame]
02:33:00 — No further activity detected
02:35:00 — [DECELERATED to default] Incident closed

Threat level: Medium
Resolution: Subject left without entry
Keyframes: 6 attached
```

This is searchable, reviewable, and far more useful than 5 minutes of continuous footage.

## Project Phases

### Phase 1-3: Done (current)
- Local webcam pipeline with YOLO detection
- Face recognition with enrollment and occlusion detection
- EventBus architecture, threaded capture
- Package distribution (pip, installers, CI/CD)

### Phase 4: Server Foundation (in progress)
- [ ] FastAPI server with proper config-driven architecture
- [x] RTSP camera integration via config (no hardcoded credentials)
- [x] Camera registry in config.yaml (multi-camera `cameras:` list with id, source, location)
- [x] Multi-camera pipeline (per-camera capture threads, display windows, alert cooldowns)
- [ ] Detection API endpoints (detect, snapshot, enroll)
- [ ] Server optional dependencies and entry point
- [ ] Tests for all server endpoints

### Phase 5: Client Application
- [ ] Client daemon that captures from local cameras
- [ ] Configurable sampling rate per camera (`s` parameter)
- [ ] Rolling frame buffer for history requests
- [ ] Secure transport to TB Server (TLS, auth tokens)
- [ ] Command channel (receive acceleration/deceleration orders)
- [ ] Local fallback mode (offline YOLO)

### Phase 6: Perception Layer (Server)
- [ ] Support for heavier YOLO models (yolov8x, yolo11l)
- [ ] GPU batch inference across cameras
- [ ] Triage logic — classify detections as benign/worth-reasoning
- [ ] Specialized model dispatch (OCR, face rec, pose)
- [ ] Detection context enrichment (time, location, history)

### Phase 7: Thought Layer
- [ ] LLM integration (API-based and local Ollama)
- [ ] Building knowledge base (camera locations, known persons, schedules)
- [ ] Decision framework (threat assessment, action selection)
- [ ] Dynamic s adjustment per camera
- [ ] Acceleration protocol
- [ ] Correlated camera attention

### Phase 8: Action & Alerting
- [ ] Tiered alert routing (notify → security → emergency)
- [ ] Narrative engine (event timelines from keyframes)
- [ ] Incident management (open, escalate, resolve)
- [ ] Integration points (WhatsApp, SMS, security company APIs)

### Phase 9: Cloud Platform (TBit)
- [ ] Multi-tenant server infrastructure
- [ ] Building onboarding and camera registration
- [ ] Dashboard for monitoring and incident review
- [ ] Usage-based billing (cameras, LLM calls, storage)
