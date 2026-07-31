"""Age/size based log and artifact trimming with optional receipts."""

from .trim import TrimResult, trim_directory

__version__ = "0.2.0"

__all__ = ["TrimResult", "trim_directory", "__version__"]
