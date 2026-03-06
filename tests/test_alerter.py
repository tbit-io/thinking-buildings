import time

from thinking_buildings.alerter import Alerter
from thinking_buildings.config import AlerterConfig
from thinking_buildings.events import Detection


def _det(**kw) -> Detection:
    defaults = dict(label="person", confidence=0.85, bbox=(0, 0, 100, 200), timestamp=time.time())
    defaults.update(kw)
    return Detection(**defaults)


class TestAlertKey:
    def test_person_with_identity(self):
        alerter = Alerter(AlerterConfig(desktop_notifications=False))
        det = _det(identity="john")
        assert alerter._alert_key(det) == "person:john"

    def test_no_identity(self):
        alerter = Alerter(AlerterConfig(desktop_notifications=False))
        det = _det(label="dog", identity=None)
        assert alerter._alert_key(det) == "dog"

    def test_unknown_person(self):
        alerter = Alerter(AlerterConfig(desktop_notifications=False))
        det = _det(identity="unknown_person")
        assert alerter._alert_key(det) == "person:unknown_person"


class TestCooldownFor:
    def test_occluded_caps_at_15(self):
        alerter = Alerter(AlerterConfig(cooldown_seconds=30, desktop_notifications=False))
        det = _det(occlusion=True)
        assert alerter._cooldown_for(det) == 15.0

    def test_occluded_short_cooldown(self):
        alerter = Alerter(AlerterConfig(cooldown_seconds=3, desktop_notifications=False))
        det = _det(occlusion=True)
        assert alerter._cooldown_for(det) == 3.0  # min(15, 3)

    def test_unknown_half_cooldown(self):
        alerter = Alerter(AlerterConfig(cooldown_seconds=30, desktop_notifications=False))
        det = _det(identity="unknown_person")
        assert alerter._cooldown_for(det) == 15.0

    def test_known_full_cooldown(self):
        alerter = Alerter(AlerterConfig(cooldown_seconds=30, desktop_notifications=False))
        det = _det(identity="john")
        assert alerter._cooldown_for(det) == 30

    def test_non_person_full_cooldown(self):
        alerter = Alerter(AlerterConfig(cooldown_seconds=20, desktop_notifications=False))
        det = _det(label="dog", identity=None)
        assert alerter._cooldown_for(det) == 20


class TestFormatAlert:
    def test_occluded(self):
        alerter = Alerter(AlerterConfig(desktop_notifications=False))
        det = _det(occlusion=True, confidence=0.9)
        msg = alerter._format_alert(det)
        assert "HIGH ALERT" in msg
        assert "Occluded" in msg

    def test_unknown_person(self):
        alerter = Alerter(AlerterConfig(desktop_notifications=False))
        det = _det(identity="unknown_person", face_confidence=0.35)
        msg = alerter._format_alert(det)
        assert "Unknown person" in msg
        assert "0.35" in msg

    def test_known_person(self):
        alerter = Alerter(AlerterConfig(desktop_notifications=False))
        det = _det(identity="alice", face_confidence=0.88)
        msg = alerter._format_alert(det)
        assert "alice" in msg
        assert "0.88" in msg

    def test_non_person(self):
        alerter = Alerter(AlerterConfig(desktop_notifications=False))
        det = _det(label="cat", identity=None)
        msg = alerter._format_alert(det)
        assert "cat" in msg


class TestCameraId:
    def test_alert_key_includes_camera_id(self):
        alerter = Alerter(AlerterConfig(desktop_notifications=False))
        det = _det(camera_id="entrance", identity="john")
        assert alerter._alert_key(det) == "entrance:person:john"

    def test_alert_key_no_camera_id(self):
        alerter = Alerter(AlerterConfig(desktop_notifications=False))
        det = _det(identity="john")
        assert alerter._alert_key(det) == "person:john"

    def test_format_alert_includes_camera_id(self):
        alerter = Alerter(AlerterConfig(desktop_notifications=False))
        det = _det(camera_id="entrance", label="cat")
        msg = alerter._format_alert(det)
        assert "[entrance]" in msg

    def test_same_detection_different_cameras_both_alert(self, mocker):
        alerter = Alerter(AlerterConfig(cooldown_seconds=60, desktop_notifications=False))
        mocker.patch("thinking_buildings.alerter.logger")
        det1 = _det(label="dog", camera_id="cam1")
        det2 = _det(label="dog", camera_id="cam2")
        alerter.handle([det1])
        alerter.handle([det2])
        # Both should alert since they're different cameras
        assert len(alerter._last_alert) == 2


class TestHandle:
    def test_respects_cooldown(self, mocker):
        alerter = Alerter(AlerterConfig(cooldown_seconds=60, desktop_notifications=False))
        mock_log = mocker.patch("thinking_buildings.alerter.logger")

        det = _det(label="dog")
        alerter.handle([det])
        assert mock_log.info.call_count == 1

        # Second call within cooldown — should not alert again
        alerter.handle([det])
        assert mock_log.info.call_count == 1

    def test_alerts_again_after_cooldown(self, mocker):
        alerter = Alerter(AlerterConfig(cooldown_seconds=0, desktop_notifications=False))
        mock_log = mocker.patch("thinking_buildings.alerter.logger")

        det = _det(label="dog")
        alerter.handle([det])
        alerter.handle([det])  # cooldown=0, should alert again
        assert mock_log.info.call_count == 2
