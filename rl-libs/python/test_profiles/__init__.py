"""Manifest-driven timed test profiles."""

from .runner import ProfileRunner, load_manifest, run_profile

__version__ = "0.2.0"

__all__ = ["ProfileRunner", "load_manifest", "run_profile", "__version__"]
