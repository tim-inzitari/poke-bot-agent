#!/usr/bin/env python3
"""Bound the stable Blackwell console log without rotating archives.

Long unattended runs (tqdm progress bars, verbose training loops) can grow log
files without bound and fill the disk. By default this tool watches only
``outputs/logs/blackwell.log`` and, whenever it grows past 256 MiB, trims it
*from the top* -- discarding the oldest bytes and keeping the newest 16 MiB.
Directory scanning remains available for explicit maintenance, but is no
longer the default because historical logs must not be altered automatically.

Why in-place truncation (and NOT ``os.replace`` of a tail copy)
--------------------------------------------------------------
The obvious "copy the tail to a temp file then ``os.replace``" approach is
subtly WRONG for the actual goal here. If a writer (e.g. a shell ``>>``
redirect from a training run) already holds the log file open, ``os.replace``
swaps the path to a brand-new inode. The writer's file descriptor still points
at the OLD, now-unlinked inode, so:

  * the writer keeps appending to the unlinked inode -> disk is NOT freed, and
  * the on-disk path (new inode) never grows again -> the watchdog never sees
    it get big, while the hidden inode fills the disk anyway.

To actually reclaim space from a file an active process is appending to, we
must rewrite it *in place on the same inode*: read the newest tail into memory,
seek to the start, write a marker + tail, then ``ftruncate`` down to the new
length. An ``O_APPEND`` writer's offset is repositioned to EOF by the kernel on
every write, so it simply continues appending after the retained tail -- it is
never corrupted. In the worst case a few of the very newest lines written in
the exact instant of the trim may be dropped; that is acceptable for a rare
multi-GB trim event and never corrupts the writer.

Usage
-----
Run as a standalone watchdog (loops forever)::

    python scripts/log_trimmer.py --file outputs/logs/blackwell.log \
        --threshold-mb 256 --keep-mb 16 --interval 30

One-shot scan (trim once and exit)::

    python scripts/log_trimmer.py --once

Import the trim function::

    from scripts.log_trimmer import trim_file_if_needed
    trim_file_if_needed("outputs/logs/train.log", threshold_bytes=10<<30,
                        keep_bytes=1<<20)
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable

MB = 1 << 20
GB = 1 << 30

DEFAULT_THRESHOLD_BYTES = 256 * MB
DEFAULT_KEEP_BYTES = 16 * MB
DEFAULT_INTERVAL_S = 30
DEFAULT_LOG_FILE = Path("outputs/logs/blackwell.log")

# The watchdog's own action log. Exempt from the normal *.log scan (it does not
# match the glob unless placed in a scanned dir) and self-capped separately.
SELF_LOG_NAME = "log_trimmer.watchdog.log"
SELF_LOG_CAP_BYTES = 2 * MB


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def acquire_trim_lock(path: str | os.PathLike) -> int | None:
    """Lock the watched inode, returning an fd held for monitor ownership.

    Writers do not participate in this advisory lock. Other trimmers do, which
    guarantees that only one truncation monitor owns a given active log inode.
    Opening through a symlink intentionally locks its current-run target.
    """
    p = Path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(p, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError:
        return None
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError):
        os.close(fd)
        return None
    return fd


def trim_lock_matches(fd: int, path: str | os.PathLike) -> bool:
    """Return whether ``path`` still names the inode locked by ``fd``."""
    try:
        held = os.fstat(fd)
        current = Path(path).stat()
    except OSError:
        return False
    return (held.st_dev, held.st_ino) == (current.st_dev, current.st_ino)


class _ActionLogger:
    """Tiny append logger that keeps its own file capped, never crashes."""

    def __init__(self, path: Path | None):
        self.path = path
        if self.path is not None:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                self.path = None

    def __call__(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        if self.path is None:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._self_cap()
        except OSError:
            # Never let logging our own actions take the watchdog down.
            pass

    def _self_cap(self) -> None:
        try:
            if self.path.stat().st_size > SELF_LOG_CAP_BYTES:
                trim_file_if_needed(
                    self.path,
                    threshold_bytes=SELF_LOG_CAP_BYTES,
                    keep_bytes=SELF_LOG_CAP_BYTES // 2,
                    log=lambda _m: None,  # avoid recursion
                )
        except OSError:
            pass


def trim_file_if_needed(
    path: str | os.PathLike,
    threshold_bytes: int = DEFAULT_THRESHOLD_BYTES,
    keep_bytes: int = DEFAULT_KEEP_BYTES,
    log=None,
) -> bool:
    """Trim ``path`` in place if it exceeds ``threshold_bytes``.

    Keeps only the newest ``keep_bytes`` (starting on a clean line boundary)
    and prepends a marker line noting the trim. The file's inode is preserved
    so a process actively appending to it keeps working.

    Returns True if a trim was performed, False otherwise. Never raises on
    missing files or permission errors.
    """
    if log is None:
        log = lambda _m: None  # noqa: E731
    p = Path(path)

    try:
        st = p.stat()
    except FileNotFoundError:
        return False
    except OSError as exc:
        log(f"stat failed for {p}: {exc}")
        return False

    if not (st.st_mode & 0o170000 == 0o100000):  # not a regular file
        return False
    size = st.st_size
    if size <= threshold_bytes:
        return False

    marker = (
        f"[log trimmed: older lines removed, kept last "
        f"{keep_bytes / MB:.3g}MB @ {_now()}]\n"
    ).encode("utf-8")

    fd = None
    try:
        fd = os.open(p, os.O_RDWR)
    except OSError as exc:
        log(f"open failed for {p}: {exc}")
        return False

    try:
        # Re-stat via the fd so we act on the real current size.
        cur_size = os.fstat(fd).st_size
        if cur_size <= threshold_bytes:
            return False

        read_from = max(0, cur_size - keep_bytes)
        starts_on_boundary = read_from == 0
        if read_from > 0:
            os.lseek(fd, read_from - 1, os.SEEK_SET)
            starts_on_boundary = os.read(fd, 1) in (b"\n", b"\r")
        os.lseek(fd, read_from, os.SEEK_SET)
        tail = _read_exact(fd, cur_size - read_from)

        # Drop a leading partial line so the retained tail starts cleanly.
        if not starts_on_boundary:
            separators = [
                pos
                for pos in (tail.find(b"\n"), tail.find(b"\r"))
                if pos >= 0
            ]
            if separators:
                split_at = min(separators)
                skip = split_at + 1
                if tail[split_at : split_at + 2] == b"\r\n":
                    skip += 1
                tail = tail[skip:]

        new_content = marker + tail

        # Rewrite in place on the SAME inode, then shrink.
        os.lseek(fd, 0, os.SEEK_SET)
        _write_all(fd, new_content)
        os.ftruncate(fd, len(new_content))
        try:
            os.fsync(fd)
        except OSError:
            pass

        log(
            f"trimmed {p}: {size / GB:.3f}GB -> {len(new_content) / MB:.3f}MB "
            f"(kept newest {len(tail) / MB:.3f}MB)"
        )
        return True
    except OSError as exc:
        log(f"trim failed for {p}: {exc}")
        return False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass


def _read_exact(fd: int, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = os.read(fd, min(remaining, 8 * MB))
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def iter_log_files(dirs: Iterable[str | os.PathLike]) -> Iterable[Path]:
    for d in dirs:
        base = Path(d)
        if not base.exists():
            continue
        try:
            for f in sorted(base.glob("*.log")):
                if f.name == SELF_LOG_NAME:
                    continue
                yield f
        except OSError:
            continue


def iter_targets(
    files: Iterable[str | os.PathLike],
    dirs: Iterable[str | os.PathLike],
) -> Iterable[Path]:
    """Yield explicit files and directory logs once per underlying inode."""
    seen: set[tuple[int, int] | tuple[str, str]] = set()
    for path in [Path(raw) for raw in files] + list(iter_log_files(dirs)):
        try:
            st = path.stat()
            key: tuple[int, int] | tuple[str, str] = (st.st_dev, st.st_ino)
        except OSError:
            key = ("path", str(path.absolute()))
        if key in seen:
            continue
        seen.add(key)
        yield path


def scan_once(
    dirs: Iterable[str | os.PathLike],
    threshold_bytes: int,
    keep_bytes: int,
    log=None,
    files: Iterable[str | os.PathLike] = (),
) -> int:
    trimmed = 0
    for f in iter_targets(files, dirs):
        if trim_file_if_needed(f, threshold_bytes, keep_bytes, log=log):
            trimmed += 1
    return trimmed


def watch(
    dirs: Iterable[str | os.PathLike],
    threshold_bytes: int,
    keep_bytes: int,
    interval_s: float,
    log=None,
    files: Iterable[str | os.PathLike] = (),
    lock_fd: int | None = None,
) -> None:
    dirs = list(dirs)
    files = list(files)
    if log is None:
        log = lambda _m: None  # noqa: E731
    log(
        f"watchdog start: files={files} dirs={dirs} "
        f"threshold={threshold_bytes / MB:.3g}MiB "
        f"keep={keep_bytes / MB:.3g}MB interval={interval_s}s pid={os.getpid()}"
    )
    while True:
        try:
            if lock_fd is not None and files and not trim_lock_matches(
                lock_fd, files[0]
            ):
                log(f"watched path changed inode; relinquishing {files[0]}")
                return
            scan_once(
                dirs,
                threshold_bytes,
                keep_bytes,
                log=log,
                files=files,
            )
        except Exception as exc:  # never die on an unexpected error
            log(f"scan error (continuing): {exc!r}")
        time.sleep(interval_s)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Top-down trim oversized log files to protect disk space."
    )
    size = ap.add_mutually_exclusive_group()
    size.add_argument(
        "--threshold-mb",
        type=float,
        default=_env_float(
            "POKEBOT_LOG_THRESHOLD_MB", DEFAULT_THRESHOLD_BYTES / MB
        ),
        help="Trim above this many MiB (default/env POKEBOT_LOG_THRESHOLD_MB: 256).",
    )
    size.add_argument(
        "--threshold-gb",
        type=float,
        default=None,
        help="Deprecated threshold in GiB; overrides --threshold-mb.",
    )
    ap.add_argument(
        "--keep-mb",
        type=float,
        default=_env_float("POKEBOT_LOG_KEEP_MB", DEFAULT_KEEP_BYTES / MB),
        help="Newest MiB to retain (default/env POKEBOT_LOG_KEEP_MB: 16).",
    )
    ap.add_argument(
        "--interval",
        type=float,
        default=_env_float("POKEBOT_LOG_TRIM_INTERVAL", DEFAULT_INTERVAL_S),
        help="Seconds between checks (default/env POKEBOT_LOG_TRIM_INTERVAL: 30).",
    )
    target = ap.add_mutually_exclusive_group()
    target.add_argument(
        "--file",
        type=Path,
        default=None,
        help=f"Single file to watch (default: {DEFAULT_LOG_FILE}).",
    )
    target.add_argument(
        "--dirs",
        nargs="+",
        default=None,
        help="Explicit directories to scan; historical logs are otherwise untouched.",
    )
    ap.add_argument("--once", action="store_true",
                    help="Scan a single time and exit.")
    ap.add_argument("--self-log", default=None,
                    help="Optional path for the watchdog's own capped action log.")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    threshold_bytes = int(
        args.threshold_gb * GB
        if args.threshold_gb is not None
        else args.threshold_mb * MB
    )
    keep_bytes = int(args.keep_mb * MB)
    if threshold_bytes <= 0 or keep_bytes <= 0 or keep_bytes >= threshold_bytes:
        print(
            "error: threshold and keep must be positive, with keep below threshold",
            file=sys.stderr,
        )
        return 2
    if args.interval <= 0:
        print("error: interval must be positive", file=sys.stderr)
        return 2

    files = [args.file or DEFAULT_LOG_FILE] if args.dirs is None else []
    dirs = args.dirs or []
    self_log_path = Path(args.self_log) if args.self_log else None
    log = _ActionLogger(self_log_path)

    lock_fd = None
    if files:
        lock_fd = acquire_trim_lock(files[0])
        if lock_fd is None:
            print(
                f"error: another truncation monitor owns {files[0]}",
                file=sys.stderr,
            )
            return 3

    if args.once:
        n = scan_once(
            dirs,
            threshold_bytes,
            keep_bytes,
            log=log,
            files=files,
        )
        log(f"one-shot scan complete: trimmed {n} file(s)")
        if lock_fd is not None:
            os.close(lock_fd)
        return 0

    try:
        watch(
            dirs,
            threshold_bytes,
            keep_bytes,
            args.interval,
            log=log,
            files=files,
            lock_fd=lock_fd,
        )
    except KeyboardInterrupt:
        log("watchdog stopped (KeyboardInterrupt)")
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
