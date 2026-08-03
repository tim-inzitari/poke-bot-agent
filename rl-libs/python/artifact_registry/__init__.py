"""Generic artifact registry + receipt-backed retention helpers."""

from .registry import ArtifactRecord, ArtifactRegistry
from .retention import retire_with_receipt, sha256_file

__version__ = "0.2.0"

__all__ = [
    "ArtifactRecord",
    "ArtifactRegistry",
    "retire_with_receipt",
    "sha256_file",
    "__version__",
]
