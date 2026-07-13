"""Bootstrap dataset: JSONL → whole-game / windowed sequences for the transformer.

Loads records produced by :mod:`poke_bot.replay_import` and featurizes each
decision step with :func:`poke_bot.features.build_board_tokens` /
:func:`poke_bot.features.build_option_tokens`. Sequences are capped at
``MODEL.max_context`` (default 320) — the whole game lives in context; if a
seat exceeds the cap, oldest timesteps are dropped.

Optional RAM-resident cache under ``config.HARDWARE.cache_dir`` (default
``data/cache/``).
"""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional, Union

from . import config, features
from .replay_import import assert_info_set, InfoSetViolation


@dataclass
class DecisionSample:
    """One decision timestep inside a game sequence."""

    board: features.SparseVector
    options: features.SparseVector
    action: list[int]
    #: Index of the chosen combo in ``action_combos`` (-1 if not found).
    action_combo_index: int
    action_combos: list[list[int]]
    env_step: int
    aux_labels: dict[str, Any] = field(default_factory=dict)


@dataclass
class GameSequence:
    """Whole-game (per-seat) training sequence."""

    episode_id: str
    seat: int
    archetype: str
    opp_archetype: str
    deck: list[int]
    value: float
    decisions: list[DecisionSample]
    info_set_ok: bool = True
    source: str = ""
    policy_targets: Optional[list[Optional[list[float]]]] = None

    def __len__(self) -> int:
        return len(self.decisions)


def _combo_index(action: list[int], combos: list[list[int]]) -> int:
    target = list(action)
    for i, c in enumerate(combos):
        if c == target:
            return i
    # Fallback: single-select match on first index.
    if len(target) == 1:
        for i, c in enumerate(combos):
            if c == target:
                return i
    return -1


def featurize_step(
    step: dict[str, Any],
    deck: list[int],
    *,
    verify_info_set: bool = True,
) -> DecisionSample:
    """Build board/option tokens for one JSONL step dict."""
    obs = step.get("observation") or {}
    if verify_info_set:
        report = assert_info_set(obs, strict=False)
        if not report.ok:
            raise InfoSetViolation("; ".join(report.violations))
        # Prefer remasked observation if assert remasked a copy — re-run strip.
        from .replay_import import _strip_opp_private

        obs, _aux, _ = _strip_opp_private(obs)

    action = [int(x) for x in (step.get("action") or [])]
    combos = features.enumerate_action_combos(obs)
    board = features.build_board_tokens(obs, deck)
    option_sv = features.build_option_tokens(obs, combos)
    return DecisionSample(
        board=board,
        options=option_sv,
        action=action,
        action_combo_index=_combo_index(action, combos),
        action_combos=combos,
        env_step=int(step.get("env_step", -1)),
        aux_labels=dict(step.get("aux_labels") or {}),
    )


def record_to_sequence(
    record: dict[str, Any],
    *,
    max_context: Optional[int] = None,
    verify_info_set: bool = True,
) -> Optional[GameSequence]:
    """Convert one JSONL record into a :class:`GameSequence`.

    Returns None if the record fails info-set checks or has no decisions.
    """
    if not record.get("info_set_ok", True):
        return None
    deck = list(record.get("deck") or [])
    if len(deck) != 60:
        return None
    steps = record.get("steps") or []
    if not steps:
        return None

    max_ctx = int(max_context if max_context is not None else config.MODEL.max_context)
    # Keep the most recent max_ctx decisions (drop oldest if over cap).
    if len(steps) > max_ctx:
        steps = steps[-max_ctx:]

    decisions: list[DecisionSample] = []
    for step in steps:
        try:
            decisions.append(
                featurize_step(step, deck, verify_info_set=verify_info_set)
            )
        except InfoSetViolation:
            return None
        except Exception:
            # Malformed select / option — skip this step but keep the sequence
            # only if we still get something; for safety drop the game.
            return None

    if not decisions:
        return None

    return GameSequence(
        episode_id=str(record.get("episode_id", "")),
        seat=int(record.get("seat", 0)),
        archetype=str(record.get("archetype", "")),
        opp_archetype=str(record.get("opp_archetype", "")),
        deck=deck,
        value=float(record.get("value", 0.0)),
        decisions=decisions,
        info_set_ok=True,
        source=str(record.get("source", "")),
        policy_targets=record.get("policy_targets"),
    )


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _cache_key(jsonl_path: Path, max_context: int, verify: bool) -> str:
    st = jsonl_path.stat()
    raw = f"{jsonl_path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{max_context}|{int(verify)}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


class BootstrapDataset:
    """In-memory (optionally disk-cached) list of :class:`GameSequence`.

    Compatible with a future torch Dataset wrapper — exposes ``__len__`` /
    ``__getitem__`` over game sequences. Training loops can flatten decisions
    or pack whole games up to ``MAX_CONTEXT``.
    """

    def __init__(
        self,
        sequences: list[GameSequence],
        *,
        jsonl_path: Optional[Path] = None,
    ) -> None:
        self.sequences = sequences
        self.jsonl_path = jsonl_path

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> GameSequence:
        return self.sequences[idx]

    @property
    def n_decisions(self) -> int:
        return sum(len(s) for s in self.sequences)

    @property
    def info_set_ok_all(self) -> bool:
        return all(s.info_set_ok for s in self.sequences)

    def summary(self) -> dict[str, Any]:
        arches: dict[str, int] = {}
        opp: dict[str, int] = {}
        for s in self.sequences:
            arches[s.archetype] = arches.get(s.archetype, 0) + 1
            opp[s.opp_archetype] = opp.get(s.opp_archetype, 0) + 1
        lengths = [len(s) for s in self.sequences]
        return {
            "n_games": len(self.sequences),
            "n_decisions": self.n_decisions,
            "info_set_ok_all": self.info_set_ok_all,
            "archetypes": arches,
            "opp_archetypes": opp,
            "max_decisions": max(lengths) if lengths else 0,
            "mean_decisions": (sum(lengths) / len(lengths)) if lengths else 0.0,
        }

    @classmethod
    def from_jsonl(
        cls,
        path: Union[str, Path],
        *,
        max_context: Optional[int] = None,
        verify_info_set: bool = True,
        max_games: int = 0,
        use_cache: bool = True,
        cache_dir: Optional[Path] = None,
    ) -> "BootstrapDataset":
        path = Path(path)
        max_ctx = int(max_context if max_context is not None else config.MODEL.max_context)
        cache_dir = Path(cache_dir) if cache_dir else Path(config.HARDWARE.cache_dir)
        cache_path: Optional[Path] = None

        if use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            key = _cache_key(path, max_ctx, verify_info_set)
            cache_path = cache_dir / f"bootstrap_{path.stem}_{key}.pkl"
            if cache_path.is_file():
                with cache_path.open("rb") as fh:
                    sequences = pickle.load(fh)
                if max_games > 0:
                    sequences = sequences[:max_games]
                return cls(sequences, jsonl_path=path)

        sequences: list[GameSequence] = []
        from tqdm.auto import tqdm

        records = list(iter_jsonl(path))
        if max_games > 0:
            records = records[:max_games]
        for record in tqdm(records, desc=f"featurize {path.name}", unit="seq"):
            seq = record_to_sequence(
                record, max_context=max_ctx, verify_info_set=verify_info_set
            )
            if seq is None:
                continue
            sequences.append(seq)

        if use_cache and cache_path is not None:
            tmp = cache_path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(sequences, fh, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache_path)

        return cls(sequences, jsonl_path=path)


def load_bootstrap_dataset(
    path: Union[str, Path],
    **kwargs: Any,
) -> BootstrapDataset:
    """Convenience wrapper around :meth:`BootstrapDataset.from_jsonl`."""
    return BootstrapDataset.from_jsonl(path, **kwargs)


def sparse_to_dict(sv: features.SparseVector) -> dict[str, Any]:
    """Serialize a SparseVector (for debugging / light caches)."""
    return {
        "index": list(sv.index),
        "value": list(sv.value),
        "offset": list(sv.offset),
        "num_words": sv.num_words,
    }
