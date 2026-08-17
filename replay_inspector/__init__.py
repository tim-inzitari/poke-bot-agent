"""Standalone, read-only replay and model inspection service.

The package is deliberately independent from ``dashboard``.  It never owns a
training selector, a managed-service action, or a writable replay/checkpoint
path.
"""

from .config import InspectorConfig

__all__ = ["InspectorConfig"]
