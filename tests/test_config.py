import textwrap
from pathlib import Path

import pytest

from thinking_buildings.config import AppConfig, FaceRecConfig, load_config


class TestLoadConfig:
    def test_no_file_returns_defaults(self, tmp_path):
        cfg = load_config(str(tmp_path / "nonexistent.yaml"))
        assert isinstance(cfg, AppConfig)
        assert cfg.camera.source == 0
        assert cfg.detector.confidence == 0.65
        assert cfg.alerter.cooldown_seconds == 30

    def test_partial_yaml_fills_defaults(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            camera:
              source: 2
        """))
        cfg = load_config(str(config_file))
        assert cfg.camera.source == 2
        assert cfg.camera.width == 1280  # default
        assert cfg.detector.model == "yolo11n.pt"  # default

    def test_box_color_list_to_tuple(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            display:
              box_color: [255, 0, 0]
        """))
        cfg = load_config(str(config_file))
        assert cfg.display.box_color == (255, 0, 0)
        assert isinstance(cfg.display.box_color, tuple)

    def test_det_size_list_to_tuple(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(textwrap.dedent("""\
            face_recognition:
              det_size: [320, 320]
        """))
        cfg = load_config(str(config_file))
        assert cfg.face_recognition.det_size == (320, 320)
        assert isinstance(cfg.face_recognition.det_size, tuple)

    def test_empty_yaml_returns_defaults(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("")
        cfg = load_config(str(config_file))
        assert isinstance(cfg, AppConfig)


class TestFaceRecConfigDefaults:
    def test_defaults(self):
        cfg = FaceRecConfig()
        assert cfg.enabled is True
        assert cfg.recognition_threshold == 0.5
        assert cfg.occlusion_det_threshold == 0.7
        assert cfg.occlusion_grace_frames == 5
        assert cfg.model_name == "buffalo_l"
        assert cfg.det_size == (640, 640)
        assert cfg.db_path == "data/faces.db"
