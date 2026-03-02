from __future__ import annotations

import hashlib
import logging
import urllib.request
from pathlib import Path
from typing import Dict

logger = logging.getLogger("thinking_buildings")

MODELS_DIR = Path.home() / ".thinking-buildings" / "models"

MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "yolo11n.pt": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
        "sha256": "",
    },
    "yolo11n.onnx": {
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx",
        "sha256": "",
    },
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    logger.info("Downloading model: %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, str(tmp))
        tmp.rename(dest)
        logger.info("Download complete: %s", dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def ensure_model(model_name: str) -> Path:
    """Return local path to the model, downloading if necessary.

    If model_name is an absolute or relative path that exists, return it as-is.
    Otherwise look up the registry and download to ~/.thinking-buildings/models/.
    """
    # Check if it's a direct path that exists
    direct = Path(model_name)
    if direct.exists():
        return direct

    # Check in models cache
    cached = MODELS_DIR / model_name
    if cached.exists():
        logger.debug("Using cached model: %s", cached)
        return cached

    # Look up registry
    entry = MODEL_REGISTRY.get(model_name)
    if entry is None:
        raise FileNotFoundError(
            f"Model {model_name!r} not found locally and not in registry. "
            f"Provide a valid path or one of: {list(MODEL_REGISTRY.keys())}"
        )

    _download(entry["url"], cached)

    # Verify hash if provided
    expected = entry.get("sha256", "")
    if expected:
        actual = _sha256(cached)
        if actual != expected:
            cached.unlink()
            raise RuntimeError(
                f"Hash mismatch for {model_name}: "
                f"expected {expected}, got {actual}"
            )

    return cached
