"""Small, dependency-free process memory and IPC cleanup helpers.

These helpers deliberately avoid importing torch or the game runtime so they
can also be exercised by lightweight lifecycle tests.
"""

from __future__ import annotations

import ctypes
import gc
import sys
from typing import Any


def release_process_heap() -> tuple[int, bool]:
    """Collect Python garbage and return free glibc arenas on Linux.

    Deleting a multi-gigabyte replay window only drops Python references.
    CPython/glibc may otherwise keep those arenas mapped in the long-lived
    trainer, making the next collection wave overlap the old high-water mark.

    Returns ``(objects_collected, malloc_trim_succeeded)``.  ``malloc_trim``
    is best-effort because it is a glibc extension and is not available on all
    Python platforms.
    """

    collected = int(gc.collect())
    if not sys.platform.startswith("linux"):
        return collected, False

    try:
        libc = ctypes.CDLL(None)
        malloc_trim = libc.malloc_trim
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        trimmed = bool(malloc_trim(0))
    except (AttributeError, OSError):
        trimmed = False
    return collected, trimmed


def close_mp_queue(q: Any) -> None:
    """Release a multiprocessing queue without waiting on a stuck feeder.

    Leaf processes may be force-terminated with unread queue data.  Cancelling
    the queue's automatic feeder join before the explicit best-effort join
    prevents interpreter shutdown from hanging while still closing every IPC
    handle owned by the trainer.
    """

    try:
        q.close()
    except (AttributeError, OSError, ValueError):
        pass
    try:
        q.cancel_join_thread()
    except (AttributeError, OSError, ValueError):
        pass
    try:
        q.join_thread()
    except (AttributeError, OSError, RuntimeError, ValueError):
        pass
