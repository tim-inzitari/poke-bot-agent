"""Disk-backed, bounded-batch r260 OwnDeck sidecar lookup on Inzi."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset import (
    GameSequence,
    OwnDeckSidecarJoinError,
    _validate_sidecar_row,
    attach_own_deck_sidecar,
)
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
        # Revision 274 explicitly preserves failed prefix-transfer remnants as
        # audit evidence.  They are dot-prefixed and are not part of the
        # committed daily layout, so ignore them while validating every
        # visible entry as one of the exact 20 receipted days.
        committed_entries = tuple(
            entry for entry in entries if not entry.name.startswith(".")
        )
        names = tuple(sorted(entry.name for entry in committed_entries))
        if (
            names != expected_days
            or any(
                entry.is_symlink() or not entry.is_dir()
                for entry in committed_entries
            )
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
                    if isinstance(decision.observation_fingerprint, str) and decision.observation_fingerprint.startswith("sha256:"):
                        rows = conn.execute("SELECT observation_fingerprint,payload FROM rows WHERE episode_id=? AND seat=? AND env_step=? AND observation_fingerprint=?", (str(sequence.episode_id), int(sequence.seat), int(decision.env_step), str(decision.observation_fingerprint))).fetchall()
                    else:
                        rows = conn.execute("SELECT observation_fingerprint,payload FROM rows WHERE episode_id=? AND seat=? AND env_step=?", (str(sequence.episode_id), int(sequence.seat), int(decision.env_step))).fetchall()
                    if len(rows) != 1:
                        raise R260InziSidecarIndexError("missing selected sidecar four-key identity")
                    decision.observation_fingerprint = str(rows[0][0])
                    yield json.loads(str(rows[0][1]))
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
            allow_legacy_board_abi=True,
        )

    def attach_available_batch(
        self, sequences: Sequence[GameSequence]
    ) -> dict[str, Any]:
        """Attach every exact sidecar match and factually mask absent rows.

        The archive-native OwnDeck producer emits rows only for joinable policy
        stages.  The expert feature stream also contains causal actions before
        the first joinable stage.  Those expert actions remain valid policy and
        ordinary-head supervision; they must be retained with OwnDeck targets
        absent rather than dropped or assigned invented labels.
        """

        joined = 0
        missing = 0
        incompatible = 0
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            for sequence in sequences:
                for index, decision in enumerate(sequence.decisions):
                    query = (
                        "SELECT observation_fingerprint,payload FROM rows "
                        "WHERE episode_id=? AND seat=? AND env_step=?"
                    )
                    parameters: tuple[Any, ...] = (
                        str(sequence.episode_id),
                        int(sequence.seat),
                        int(decision.env_step),
                    )
                    if (
                        isinstance(decision.observation_fingerprint, str)
                        and decision.observation_fingerprint.startswith("sha256:")
                    ):
                        query += " AND observation_fingerprint=?"
                        parameters += (str(decision.observation_fingerprint),)
                    rows = connection.execute(query, parameters).fetchall()
                    if not rows:
                        missing += 1
                        continue
                    if len(rows) != 1:
                        raise R260InziSidecarIndexError(
                            "duplicate selected sidecar four-key identity"
                        )
                    original_fingerprint = decision.observation_fingerprint
                    decision.observation_fingerprint = str(rows[0][0])
                    payload = json.loads(str(rows[0][1]))
                    try:
                        (
                            snapshot,
                            supervision,
                            option_features,
                            stage_indices,
                            selected_indices,
                            _raw_parity,
                            _projected_selected_index,
                            action_combos_fingerprints,
                        ) = _validate_sidecar_row(
                            payload,
                            sequence=sequence,
                            decision=decision,
                            expected_source_manifest_sha256=self.source_manifest_sha256,
                            allow_legacy_board_abi=True,
                        )
                    except OwnDeckSidecarJoinError:
                        decision.observation_fingerprint = original_fingerprint
                        incompatible += 1
                        continue
                    decision.ledger_snapshot = snapshot
                    decision.own_deck_supervision = supervision
                    decision.own_deck_supervision_stage_indices = stage_indices
                    decision.sidecar_action_combos_fingerprints = (
                        action_combos_fingerprints
                    )
                    for stage, option_rows, selected_index in zip(
                        decision.policy_stages,
                        option_features,
                        selected_indices,
                        strict=True,
                    ):
                        stage.target_index = int(selected_index)
                        stage.ledger_option_features = option_rows
                    joined += 1
        finally:
            connection.close()
        return {
            "joined_decision_count": int(joined),
            "masked_unjoinable_decision_count": int(missing + incompatible),
            "missing_sidecar_decision_count": int(missing),
            "incompatible_sidecar_decision_count": int(incompatible),
        }

    def attach_available_rows(
        self,
        sequences: Sequence[GameSequence],
        sidecar_rows: Iterable[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Bulk-attach one verified daily sidecar stream without SQLite seeks.

        Daily r259 files are immutable, checksum-verified inputs.  Building the
        selected decision-key map once and streaming that day's compressed rows
        changes the join from millions of SQLite point queries into one linear
        pass while preserving the same row validator and factual masking rules
        as :meth:`attach_available_batch`.
        """

        # Compact feature shards retain the pre-projection observation
        # fingerprint.  The committed sidecar/index owns the canonical public
        # fingerprint, so match by the same unique three-key identity used by
        # ``attach_available_batch`` and then validate the canonical four-key
        # row through ``_validate_sidecar_row``.
        expected: dict[tuple[str, int, int], tuple[GameSequence, Any]] = {}
        total = 0
        for sequence in sequences:
            for decision in sequence.decisions:
                total += 1
                key = (
                    str(sequence.episode_id),
                    int(sequence.seat),
                    int(decision.env_step),
                )
                if key in expected:
                    raise R260InziSidecarIndexError(
                        "daily feature shard repeats a sidecar four-key identity"
                    )
                expected[key] = (sequence, decision)

        joined = 0
        incompatible = 0
        scanned = 0
        matched_keys: set[tuple[str, int, int]] = set()
        for payload in sidecar_rows:
            scanned += 1
            if not isinstance(payload, Mapping):
                raise R260InziSidecarIndexError(
                    "daily sidecar stream emitted a non-object row"
                )
            try:
                key = (
                    str(payload["episode_id"]),
                    int(payload["seat"]),
                    int(payload["env_step"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise R260InziSidecarIndexError(
                    "daily sidecar row has an invalid four-key identity"
                ) from exc
            target = expected.get(key)
            if target is None:
                continue
            if key in matched_keys:
                raise R260InziSidecarIndexError(
                    "daily sidecar repeats a selected four-key identity"
                )
            matched_keys.add(key)
            sequence, decision = target
            original_fingerprint = decision.observation_fingerprint
            decision.observation_fingerprint = str(
                payload["observation_fingerprint"]
            )
            try:
                (
                    snapshot,
                    supervision,
                    option_features,
                    stage_indices,
                    selected_indices,
                    _raw_parity,
                    _projected_selected_index,
                    action_combos_fingerprints,
                ) = _validate_sidecar_row(
                    payload,
                    sequence=sequence,
                    decision=decision,
                    expected_source_manifest_sha256=self.source_manifest_sha256,
                    allow_legacy_board_abi=True,
                )
            except OwnDeckSidecarJoinError:
                decision.observation_fingerprint = original_fingerprint
                incompatible += 1
                continue
            decision.ledger_snapshot = snapshot
            decision.own_deck_supervision = supervision
            decision.own_deck_supervision_stage_indices = stage_indices
            decision.sidecar_action_combos_fingerprints = (
                action_combos_fingerprints
            )
            for stage, option_rows, selected_index in zip(
                decision.policy_stages,
                option_features,
                selected_indices,
                strict=True,
            ):
                stage.target_index = int(selected_index)
                stage.ledger_option_features = option_rows
            joined += 1

        missing = total - len(matched_keys)
        return {
            "joined_decision_count": int(joined),
            "masked_unjoinable_decision_count": int(missing + incompatible),
            "missing_sidecar_decision_count": int(missing),
            "incompatible_sidecar_decision_count": int(incompatible),
            "daily_sidecar_rows_scanned": int(scanned),
            "daily_bulk_join": True,
            "sqlite_point_queries": 0,
        }

    def attach_available_resident(
        self, sequences: Sequence[GameSequence]
    ) -> dict[str, Any]:
        """Bulk-attach a complete RAM-resident corpus in one SQLite scan.

        This is the full-corpus counterpart of ``attach_available_batch``.
        It trades a bounded in-memory key map for eliminating millions of
        point queries and is intended only for a host with explicit resident-
        corpus capacity.
        """

        decisions: dict[tuple[str, int, int, str], tuple[GameSequence, Any]] = {}
        total = 0
        for sequence in sequences:
            for decision in sequence.decisions:
                total += 1
                fingerprint = str(decision.observation_fingerprint or "")
                key = (
                    str(sequence.episode_id),
                    int(sequence.seat),
                    int(decision.env_step),
                    fingerprint,
                )
                if key in decisions:
                    raise R260InziSidecarIndexError(
                        "resident corpus repeats a sidecar four-key identity"
                    )
                decisions[key] = (sequence, decision)

        joined = 0
        incompatible = 0
        matched_keys: set[tuple[str, int, int, str]] = set()
        connection = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            cursor = connection.execute(
                "SELECT episode_id,seat,env_step,observation_fingerprint,payload "
                "FROM rows"
            )
            for episode_id, seat, env_step, fingerprint, raw_payload in cursor:
                key = (str(episode_id), int(seat), int(env_step), str(fingerprint))
                target = decisions.get(key)
                if target is None:
                    continue
                matched_keys.add(key)
                sequence, decision = target
                payload = json.loads(str(raw_payload))
                try:
                    (
                        snapshot,
                        supervision,
                        option_features,
                        stage_indices,
                        selected_indices,
                        _raw_parity,
                        _projected_selected_index,
                        action_combos_fingerprints,
                    ) = _validate_sidecar_row(
                        payload,
                        sequence=sequence,
                        decision=decision,
                        expected_source_manifest_sha256=self.source_manifest_sha256,
                        allow_legacy_board_abi=True,
                    )
                except OwnDeckSidecarJoinError:
                    incompatible += 1
                    continue
                decision.ledger_snapshot = snapshot
                decision.own_deck_supervision = supervision
                decision.own_deck_supervision_stage_indices = stage_indices
                decision.sidecar_action_combos_fingerprints = (
                    action_combos_fingerprints
                )
                for stage, option_rows, selected_index in zip(
                    decision.policy_stages,
                    option_features,
                    selected_indices,
                    strict=True,
                ):
                    stage.target_index = int(selected_index)
                    stage.ledger_option_features = option_rows
                joined += 1
        finally:
            connection.close()
        missing = total - len(matched_keys)
        return {
            "joined_decision_count": int(joined),
            "masked_unjoinable_decision_count": int(missing + incompatible),
            "missing_sidecar_decision_count": int(missing),
            "incompatible_sidecar_decision_count": int(incompatible),
            "resident_decision_count": int(total),
            "index_scan_count": 1,
        }
