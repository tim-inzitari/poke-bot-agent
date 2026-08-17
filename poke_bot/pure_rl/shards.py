"""Compact pure-RL shard IO: selected actions + masks, not soft behavior π."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


_MAX_WRITE_BUFFER_BYTES = 64 * 1024 * 1024


def compact_shard_write_buffer_bytes(*, default: int = 0) -> int:
    """Return the explicitly opted-in compact-shard sequential write buffer.

    The historical writer opens and closes the shard once per game.  That is
    safest for a live append-only recovery path, so the default remains exact
    legacy behavior (``0``).  A receipt-gated collection deployment may opt
    into a bounded buffer with ``PURE_RL_SHARD_WRITE_BUFFER_BYTES``; this only
    changes when already-complete JSONL lines become visible, never their
    content, order, counters, or final shard bytes.
    """

    raw = os.environ.get("PURE_RL_SHARD_WRITE_BUFFER_BYTES")
    if raw is None or not str(raw).strip():
        return max(0, int(default))
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return max(0, int(default))
    return max(0, min(_MAX_WRITE_BUFFER_BYTES, value))


@dataclass
class CompactDecision:
    """One training decision without soft policy clone targets."""

    env_step: int
    selected_index: int
    n_options: int
    action: list[int]
    observation: dict[str, Any] = field(default_factory=dict)
    # Training-only labels are never part of the policy observation.  They are
    # retained only when the simulator/replay importer can construct them
    # exactly; downstream losses mask individual missing targets.
    aux_labels: dict[str, Any] = field(default_factory=dict)
    tactical_sequence_supervision: dict[str, Any] | None = None


@dataclass
class CompactGame:
    episode_id: str
    seat: int
    archetype: str
    opp_archetype: str
    deck: list[int]
    value: float
    decisions: list[CompactDecision] = field(default_factory=list)
    source: str = "pure_rl"
    target_provenance: dict[str, Any] = field(default_factory=dict)


def compact_decision_from_step(
    step: dict[str, Any],
    *,
    selected_index: Optional[int] = None,
) -> CompactDecision:
    """Build a compact decision; drops soft ``policy`` rows if present."""
    action = [int(x) for x in (step.get("action") or [])]
    n_options = int(step.get("legal_action_count") or step.get("n_options") or 0)
    idx = selected_index
    if idx is None:
        raw = step.get("selected_index")
        if raw is not None:
            idx = int(raw)
        else:
            # Factorized stage row may carry selected_index.
            stages = step.get("factorized_policy_targets") or step.get("policy_stages")
            if isinstance(stages, list) and stages:
                first = stages[0] if not isinstance(stages[0], list) else (
                    stages[0][0] if stages[0] else {}
                )
                if isinstance(first, dict) and first.get("selected_index") is not None:
                    idx = int(first["selected_index"])
    if idx is None:
        idx = 0
    if n_options <= 0:
        n_options = max(idx + 1, 1)
    return CompactDecision(
        env_step=int(step.get("env_step") or step.get("step") or 0),
        selected_index=int(idx),
        n_options=int(n_options),
        action=action,
        observation=dict(step.get("observation") or {}),
        aux_labels=dict(step.get("aux_labels") or {}),
        tactical_sequence_supervision=(
            dict(step["tactical_sequence_supervision"])
            if isinstance(step.get("tactical_sequence_supervision"), dict)
            else None
        ),
    )


def game_to_jsonable(game: CompactGame) -> dict[str, Any]:
    return {
        "episode_id": game.episode_id,
        "seat": game.seat,
        "archetype": game.archetype,
        "opp_archetype": game.opp_archetype,
        "deck": list(game.deck),
        "value": float(game.value),
        "source": game.source,
        "target_provenance": {
            **dict(game.target_provenance),
            "pure_rl": True,
            "soft_policy_targets": False,
        },
        "decisions": [
            {
                "env_step": d.env_step,
                "selected_index": d.selected_index,
                "n_options": d.n_options,
                "action": list(d.action),
                "observation": d.observation,
                "aux_labels": d.aux_labels,
                "tactical_sequence_supervision": (
                    d.tactical_sequence_supervision
                ),
            }
            for d in game.decisions
        ],
    }


class CompactShardWriter:
    """Append compact games to a JSONL shard (worker-safe via one writer)."""

    def __init__(self, path: Path, *, buffer_bytes: Optional[int] = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._n_games = 0
        self._n_decisions = 0
        self._t0 = time.time()
        self._read_only = False
        requested_buffer = (
            compact_shard_write_buffer_bytes()
            if buffer_bytes is None
            else int(buffer_bytes)
        )
        self._buffer_bytes = max(0, min(_MAX_WRITE_BUFFER_BYTES, requested_buffer))
        self._pending_lines: list[str] = []
        self._pending_bytes = 0
        self._flush_count = 0

    @classmethod
    def from_completed_shard(
        cls,
        path: Path,
        *,
        n_games: int,
        n_decisions: int,
        elapsed_sec: float,
    ) -> "CompactShardWriter":
        """Rehydrate counters for an immutable, receipt-verified shard."""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if int(n_games) < 0 or int(n_decisions) < 0:
            raise ValueError("completed shard counters must be non-negative")
        writer = cls(path)
        writer._n_games = int(n_games)
        writer._n_decisions = int(n_decisions)
        writer._t0 = time.time() - max(float(elapsed_sec), 1e-6)
        writer._read_only = True
        return writer

    def write_game(self, game: CompactGame) -> bool:
        """Append one complete JSONL game, returning whether bytes were flushed.

        A buffered writer retains only complete newline-terminated records in
        RAM.  This is important for prefix readers: they either see the prior
        durable prefix or a whole collection of new lines, never a torn JSON
        record.  Callers that stream-featurize the shard use the return value
        to avoid rescanning after an in-memory-only append.
        """

        if self._read_only:
            raise RuntimeError(f"completed shard is immutable: {self.path}")
        line = json.dumps(game_to_jsonable(game), separators=(",", ":")) + "\n"
        self._n_games += 1
        self._n_decisions += len(game.decisions)
        if self._buffer_bytes <= 0:
            self._append_text(line)
            return True
        self._pending_lines.append(line)
        self._pending_bytes += len(line.encode("utf-8"))
        if self._pending_bytes >= self._buffer_bytes:
            return self.flush()
        return False

    def write_games(self, games: Iterable[CompactGame]) -> None:
        for game in games:
            self.write_game(game)
        # ``write_games`` has historically completed a fully visible append
        # before it returns. Preserve that useful batch boundary even when the
        # caller explicitly enables the throughput buffer.
        self.flush()

    def _append_text(self, data: str) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(data)
        self._flush_count += 1

    def flush(self) -> bool:
        """Flush complete buffered records in one sequential append.

        No per-flush ``fsync`` is introduced: the legacy path did not fsync
        every game either, and final receipt/sealing remains the durability
        boundary.  On an I/O exception the pending records remain intact for
        the caller to surface or retry; they are never silently discarded.
        """

        if self._read_only:
            return False
        if not self._pending_lines:
            return False
        data = "".join(self._pending_lines)
        self._append_text(data)
        self._pending_lines.clear()
        self._pending_bytes = 0
        return True

    @property
    def n_games(self) -> int:
        return self._n_games

    @property
    def n_decisions(self) -> int:
        return self._n_decisions

    @property
    def buffer_bytes(self) -> int:
        """Configured bounded buffer capacity (zero is legacy immediate IO)."""

        return self._buffer_bytes

    @property
    def pending_bytes(self) -> int:
        """Current not-yet-visible complete JSONL payload size."""

        return self._pending_bytes

    @property
    def flush_count(self) -> int:
        """Number of sequential append syscalls performed by this writer."""

        return self._flush_count

    def throughput(self) -> dict[str, float]:
        elapsed = max(time.time() - self._t0, 1e-6)
        return {
            "games_per_sec": self._n_games / elapsed,
            "decisions_per_sec": self._n_decisions / elapsed,
            "elapsed_sec": elapsed,
        }


def iter_shard_games(path: Path) -> Iterator[CompactGame]:
    path = Path(path)
    if not path.is_file():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            decisions = [
                CompactDecision(
                    env_step=int(d.get("env_step") or 0),
                    selected_index=int(d.get("selected_index") or 0),
                    n_options=int(d.get("n_options") or 1),
                    action=[int(x) for x in (d.get("action") or [])],
                    observation=dict(d.get("observation") or {}),
                    aux_labels=dict(d.get("aux_labels") or {}),
                    tactical_sequence_supervision=(
                        dict(d["tactical_sequence_supervision"])
                        if isinstance(d.get("tactical_sequence_supervision"), dict)
                        else None
                    ),
                )
                for d in (raw.get("decisions") or [])
            ]
            yield CompactGame(
                episode_id=str(raw.get("episode_id") or ""),
                seat=int(raw.get("seat") or 0),
                archetype=str(raw.get("archetype") or ""),
                opp_archetype=str(raw.get("opp_archetype") or ""),
                deck=[int(x) for x in (raw.get("deck") or [])],
                value=float(raw.get("value") or 0.0),
                decisions=decisions,
                source=str(raw.get("source") or "pure_rl"),
                target_provenance=dict(raw.get("target_provenance") or {}),
            )
