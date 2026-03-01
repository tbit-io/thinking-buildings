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
            last = self._last_alert.get(det.label, 0)
            if now - last < self.cooldown:
                continue

            self._last_alert[det.label] = now
            msg = f"ALERT: {det.label} detected (confidence: {det.confidence:.0%})"
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
