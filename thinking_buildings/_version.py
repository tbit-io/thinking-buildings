from __future__ import annotations

try:
    from importlib.metadata import version

    __version__ = version("thinking-buildings")
except Exception:
    __version__ = "0.0.0-dev"
