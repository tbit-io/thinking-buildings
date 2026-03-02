from __future__ import annotations

import logging
from typing import Dict, List, Optional, Type

from thinking_buildings.backends.base import InferenceBackend, RawDetection

logger = logging.getLogger("thinking_buildings")

_REGISTRY: List[Type[InferenceBackend]] = []


def _register_backends() -> None:
    """Register all known backends. Import errors are caught silently."""
    global _REGISTRY
    if _REGISTRY:
        return

    from thinking_buildings.backends.ultralytics_backend import UltralyticsBackend
    _REGISTRY.append(UltralyticsBackend)


def get_backend(preferred: Optional[str] = None) -> InferenceBackend:
    """Return the best available backend, or the one specified by name.

    Args:
        preferred: "auto" or None for auto-detect, or a backend name
                   like "ultralytics".
    """
    _register_backends()

    name_map: Dict[str, Type[InferenceBackend]] = {
        "ultralytics": _REGISTRY[0] if _REGISTRY else None,  # type: ignore[dict-item]
    }

    if preferred and preferred != "auto":
        cls = name_map.get(preferred)
        if cls is None:
            raise ValueError(f"Unknown backend: {preferred!r}")
        if not cls.is_available():
            raise RuntimeError(f"Backend {preferred!r} is not available")
        logger.info("Using backend: %s (forced)", preferred)
        return cls()

    # Auto-detect: sort by priority, pick first available
    available = [cls for cls in _REGISTRY if cls.is_available()]
    if not available:
        raise RuntimeError(
            "No inference backends available. "
            "Install ultralytics (`pip install ultralytics`) or "
            "onnxruntime (`pip install onnxruntime`)."
        )
    available.sort(key=lambda c: c.priority(), reverse=True)
    chosen = available[0]
    logger.info("Auto-selected backend: %s (priority=%d)",
                chosen.__name__, chosen.priority())
    return chosen()


__all__ = ["get_backend", "InferenceBackend", "RawDetection"]
