from __future__ import annotations

import logging
import time
from typing import Dict, List

from thinking_buildings.config import AlerterConfig
from thinking_buildings.events import Detection

logger = logging.getLogger("thinking_buildings")


class Alerter:
    def __init__(self, cfg: AlerterConfig) -> None:
        self.cooldown = cfg.cooldown_seconds
        self.desktop_notifications = cfg.desktop_notifications
        self._last_alert: Dict[str, float] = {}
        self._notifier = None
        if self.desktop_notifications:
            try:
                from plyer import notification
                self._notifier = notification
            except ImportError:
                logger.warning("plyer not available — desktop notifications disabled")

    def handle(self, detections: List[Detection]) -> None:
        now = time.time()
        for det in detections:
            cooldown = self._cooldown_for(det)
            alert_key = self._alert_key(det)
            last = self._last_alert.get(alert_key, 0)
            if now - last < cooldown:
                continue

            self._last_alert[alert_key] = now
            msg = self._format_alert(det)
            logger.info(msg)

            if self._notifier:
                try:
                    self._notifier.notify(
                        title="Thinking Buildings",
                        message=msg,
                        timeout=5,
                    )
                except Exception:
                    pass

    def _alert_key(self, det: Detection) -> str:
        prefix = f"{det.camera_id}:" if det.camera_id else ""
        if det.identity:
            return f"{prefix}{det.label}:{det.identity}"
        return f"{prefix}{det.label}"

    def _cooldown_for(self, det: Detection) -> float:
        if det.occlusion:
            return min(15.0, self.cooldown)
        if det.identity == "unknown_person":
            return self.cooldown / 2
        return self.cooldown

    def _format_alert(self, det: Detection) -> str:
        cam = f"[{det.camera_id}] " if det.camera_id else ""
        if det.occlusion:
            return (
                f"{cam}HIGH ALERT: Occluded/hidden face detected "
                f"(person confidence: {det.confidence:.0%})"
            )
        if det.identity == "unknown_person":
            conf = f", similarity: {det.face_confidence:.2f}" if det.face_confidence is not None else ""
            return f"{cam}ALERT: Unknown person detected (confidence: {det.confidence:.0%}{conf})"
        if det.identity and det.identity != "occluded_face":
            conf = f", similarity: {det.face_confidence:.2f}" if det.face_confidence is not None else ""
            return f"{cam}ALERT: {det.identity} detected (confidence: {det.confidence:.0%}{conf})"
        return f"{cam}ALERT: {det.label} detected (confidence: {det.confidence:.0%})"
