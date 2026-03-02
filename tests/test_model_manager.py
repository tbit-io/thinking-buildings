from pathlib import Path
from unittest.mock import patch

import pytest

from thinking_buildings.model_manager import ensure_model, MODELS_DIR, MODEL_REGISTRY


class TestEnsureModel:
    def test_local_path_exists(self, tmp_path):
        model_file = tmp_path / "custom.pt"
        model_file.write_bytes(b"fake model")
        result = ensure_model(str(model_file))
        assert result == model_file

    def test_cached_model_returned(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        monkeypatch.setattr("thinking_buildings.model_manager.MODELS_DIR", cache_dir)
        monkeypatch.chdir(tmp_path)  # avoid finding yolo11n.pt in project root
        cached = cache_dir / "yolo11n.pt"
        cached.write_bytes(b"cached model")
        result = ensure_model("yolo11n.pt")
        assert result == cached

    def test_unknown_model_raises(self, tmp_path, monkeypatch):
        monkeypatch.setattr("thinking_buildings.model_manager.MODELS_DIR", tmp_path)
        with pytest.raises(FileNotFoundError, match="not found locally"):
            ensure_model("nonexistent_model.pt")

    @patch("thinking_buildings.model_manager._download")
    def test_downloads_from_registry(self, mock_download, tmp_path, monkeypatch):
        monkeypatch.setattr("thinking_buildings.model_manager.MODELS_DIR", tmp_path)
        monkeypatch.chdir(tmp_path)  # avoid finding yolo11n.pt in project root

        def fake_download(url, dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"downloaded")

        mock_download.side_effect = fake_download

        result = ensure_model("yolo11n.pt")
        assert result == tmp_path / "yolo11n.pt"
        mock_download.assert_called_once()


class TestModelRegistry:
    def test_registry_has_expected_models(self):
        assert "yolo11n.pt" in MODEL_REGISTRY
        assert "yolo11n.onnx" in MODEL_REGISTRY

    def test_registry_entries_have_url(self):
        for name, entry in MODEL_REGISTRY.items():
            assert "url" in entry, f"{name} missing url"
