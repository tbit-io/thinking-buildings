#!/usr/bin/env python3
"""Capture a snapshot from an RTSP camera every N seconds."""

import subprocess
import time
from datetime import datetime
from pathlib import Path

RTSP_URL = "rtsp://admin:Ijavier.27@192.168.1.19:554/cam/realmonitor?channel=1&subtype=0"
INTERVAL = 10  # seconds
OUTPUT_DIR = Path("snapshots")


def capture_frame(url: str, output_path: Path) -> bool:
    """Capture a single frame from RTSP stream using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-rtsp_transport", "tcp",
        "-i", url,
        "-frames:v", "1",
        "-y",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=15)
    return result.returncode == 0


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Capturing from: {RTSP_URL}")
    print(f"Saving to: {OUTPUT_DIR}/")
    print(f"Interval: {INTERVAL}s")
    print("Press Ctrl+C to stop\n")

    count = 0
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"snap_{timestamp}.jpg"

        if capture_frame(RTSP_URL, output_path):
            count += 1
            print(f"[{count}] {output_path}")
        else:
            print(f"[!] Failed to capture at {timestamp}")

        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
