"""Disk-backed, bounded-batch r260 OwnDeck sidecar lookup on Inzi."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset import GameSequence, attach_own_deck_sidecar
from .own_deck_rollout_store import (
    DAILY_DIRECTORY_NAME,
    iter_daily_sidecar_rows,
    read_daily_meta,
)


class R260InziSidecarIndexError(RuntimeError):
    pass


class R260InziSidecarIndex:
    """Immutable SQLite key index; query only selected host batches."""

    def __init__(self, path: Path, *, source_manifest_sha256: str, daily_meta_sha256s: Mapping[str, str]):
        self.path = Path(path).resolve()
        self.source_manifest_sha256 = str(source_manifest_sha256)
        self.daily_meta_sha256s = {
            str(key): str(value)
            for key, value in sorted(daily_meta_sha256s.items())
        }

    @classmethod
    def build(
        cls, *, sidecar_root: Path | str, output: Path | str,
        source_manifest_sha256: str, daily_meta_sha256s: Mapping[str, str],
    ) -> "R260InziSidecarIndex":
        root, target = Path(sidecar_root).expanduser(), Path(output).expanduser().resolve()
        if root.is_symlink() or not root.is_dir() or target.exists():
            raise R260InziSidecarIndexError("sidecar root must be local and index output create-only")
        expected_days = tuple(sorted(str(day) for day in daily_meta_sha256s))
        if len(expected_days) != 20 or len(set(expected_days)) != 20:
            raise R260InziSidecarIndexError("r260 index requires exactly 20 distinct committed days")
        daily_root = root / DAILY_DIRECTORY_NAME
        if daily_root.is_symlink() or not daily_root.is_dir():
            raise R260InziSidecarIndexError("r260 Inzi dataset has no committed daily layout")
        try:
            entries = tuple(daily_root.iterdir())
        except OSError as exc:
            raise R260InziSidecarIndexError("r260 Inzi daily layout is unreadable") from exc
        names = tuple(sorted(entry.name for entry in entries))
        if (
            any(name.startswith(".") for name in names)
            or names != expected_days
            or any(entry.is_symlink() or not entry.is_dir() for entry in entries)
        ):
            raise R260InziSidecarIndexError(
                "r260 index accepts only the final 20 committed non-dot daily directories"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(target)
        try:
            conn.executescript("""PRAGMA journal_mode=DELETE; PRAGMA synchronous=FULL;
                CREATE TABLE rows (episode_id TEXT NOT NULL, seat INTEGER NOT NULL,
                env_step INTEGER NOT NULL, observation_fingerprint TEXT NOT NULL,
                payload TEXT NOT NULL, PRIMARY KEY(episode_id,seat,env_step,observation_fingerprint));""")
            conn.execute(
                "CREATE TABLE metadata (key TEXT NOT NULL PRIMARY KEY, value TEXT NOT NULL)"
            )
            metadata = {
                "schema": "poke_bot.r260_inzi_sidecar_index/v1",
                "source_manifest_sha256": str(source_manifest_sha256),
                "daily_meta_sha256s": dict(sorted((str(day), str(digest)) for day, digest in daily_meta_sha256s.items())),
            }
            conn.executemany(
                "INSERT INTO metadata VALUES (?, ?)",
                [
                    (key, json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
                    for key, value in sorted(metadata.items())
                ],
            )
            for day, expected in sorted(daily_meta_sha256s.items()):
                meta = read_daily_meta(root, day)
                if meta.get("meta_sha256") != expected or (((meta.get("source") or {}).get("manifest") or {}).get("sha256")) != source_manifest_sha256:
                    raise R260InziSidecarIndexError(f"daily sidecar identity changed: {day}")
                for row in iter_daily_sidecar_rows(root, day, expected_meta_sha256=expected):
                    key = (row.get("episode_id"), row.get("seat"), row.get("env_step"), row.get("observation_fingerprint"))
                    if not isinstance(key[0], str) or key[1] not in (0, 1) or not isinstance(key[2], int) or not isinstance(key[3], str):
                        raise R260InziSidecarIndexError("sidecar row has invalid four-key identity")
                    try:
                        conn.execute("INSERT INTO rows VALUES (?,?,?,?,?)", (*key, json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)))
                    except sqlite3.IntegrityError as exc:
                        raise R260InziSidecarIndexError("duplicate sidecar four-key identity") from exc
            conn.commit()
        except Exception:
            conn.close(); target.unlink(missing_ok=True); raise
        conn.close(); target.chmod(0o444)
        return cls(target, source_manifest_sha256=source_manifest_sha256, daily_meta_sha256s=daily_meta_sha256s)

    def assert_verified(self, *, expected_source_manifest_sha256: str, daily_meta_sha256s: Mapping[str, str]) -> None:
        expected_daily = {
            str(key): str(value)
            for key, value in sorted(daily_meta_sha256s.items())
        }
        if (
            self.path.is_symlink()
            or not self.path.is_file()
            or self.path.stat().st_mode & 0o222
            or str(expected_source_manifest_sha256) != self.source_manifest_sha256
            or expected_daily != self.daily_meta_sha256s
        ):
            raise R260InziSidecarIndexError("sidecar index provenance mismatch")
        conn: sqlite3.Connection | None = None
        try:
            conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            rows = dict(conn.execute("SELECT key, value FROM metadata"))
        except (OSError, sqlite3.Error) as exc:
            raise R260InziSidecarIndexError("sidecar index metadata is unreadable") from exc
        finally:
            if conn is not None:
                conn.close()
        try:
            observed = {key: json.loads(value) for key, value in rows.items()}
        except (TypeError, json.JSONDecodeError) as exc:
            raise R260InziSidecarIndexError("sidecar index metadata is malformed") from exc
        if observed != {
            "schema": "poke_bot.r260_inzi_sidecar_index/v1",
            "source_manifest_sha256": str(expected_source_manifest_sha256),
            "daily_meta_sha256s": expected_daily,
        }:
            raise R260InziSidecarIndexError("sidecar index metadata provenance drifted")

    def rows_for_sequences(self, sequences: Sequence[GameSequence]) -> Iterable[Mapping[str, Any]]:
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            for sequence in sequences:
                for decision in sequence.decisions:
                    row = conn.execute("SELECT payload FROM rows WHERE episode_id=? AND seat=? AND env_step=? AND observation_fingerprint=?", (str(sequence.episode_id), int(sequence.seat), int(decision.env_step), str(decision.observation_fingerprint))).fetchone()
                    if row is None:
                        raise R260InziSidecarIndexError("missing selected sidecar four-key identity")
                    yield json.loads(str(row[0]))
        finally:
            conn.close()

    def attach_batch(self, sequences: Sequence[GameSequence]) -> dict[str, Any]:
        """Exact-key attach only the selected host batch; no full corpus in RAM."""
        return attach_own_deck_sidecar(
            sequences,
            expected_source_manifest_sha256=self.source_manifest_sha256,
            daily_meta_sha256s=self.daily_meta_sha256s,
            sidecar_rows=self.rows_for_sequences(sequences),
            verified_sidecar_index=self,
        )
