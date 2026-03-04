#!/usr/bin/env python3
"""Cron job cada 10s — detecta person/cat/dog en cámara sala."""

from datetime import datetime

import requests
import schedule
import time

API_URL = "http://localhost:8700"
CAMERAS = ["sala"]
TARGETS = {"person", "cat", "dog"}


def check_camera(camera: str):
    try:
        resp = requests.get(
            f"{API_URL}/snapshot",
            params={"camera": camera, "detect": "true"},
            timeout=20,
        )
        data = resp.json()
    except Exception as e:
        print(f"[!] {camera}: {e}")
        return

    if "error" in data:
        print(f"[!] {camera}: {data['error']}")
        return

    detections = data.get("detections", [])
    found = [d for d in detections if d["label"] in TARGETS]

    if found:
        ts = datetime.now().strftime("%H:%M:%S")
        labels = [d["label"] for d in found]
        print(f"\n[{ts}] [{camera}] {labels} {data['file']}")
    else:
        print(".", end="", flush=True)


def check_all_cameras():
    for cam in CAMERAS:
        check_camera(cam)


# Cron: cada 10 minutos revisa todas las cámaras
schedule.every(10).seconds.do(check_all_cameras)

print(f"Cron activo — monitoreando {CAMERAS} cada 10 min")
print(f"Targets: {TARGETS}")
print("Ctrl+C para parar\n")

# Primera ejecución inmediata
check_all_cameras()

while True:
    schedule.run_pending()
    time.sleep(1)
