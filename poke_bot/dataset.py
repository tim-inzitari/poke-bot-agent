"""Bootstrap dataset conversion with explicit accounting.

Loads records produced by :mod:`poke_bot.replay_import` and featurizes each
decision step with :func:`poke_bot.features.build_board_tokens` /
:func:`poke_bot.features.build_option_tokens`. Decisions remain grouped as each
acting seat's causal, deployment-visible history. ``max_context`` truncation is
applied identically to steps and policy targets and reported.

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


DATASET_CACHE_SCHEMA_VERSION = 4


@dataclass
class PolicyStage:
    """One teacher-forced autoregressive option/STOP decision."""

    options: features.SparseVector
    action_combos: list[list[int]]
    target_index: int


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
    #: One-word decoder feature for the action actually taken. Shifted by one
    #: timestep when building causal history so no current-target leakage occurs.
    action_token: Optional[features.SparseVector] = None
    policy_stages: list[PolicyStage] = field(default_factory=list)


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
    factorized_policy_targets: Optional[list[Optional[list[dict[str, Any]]]]] = None
    target_provenance: dict[str, Any] = field(default_factory=dict)

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
    stage_defs = features.factorized_teacher_forcing_stages(obs, action)
    policy_stages = [
        PolicyStage(
            options=features.build_option_tokens(obs, combos),
            action_combos=[list(combo) for combo in combos],
            target_index=target_index,
        )
        for combos, target_index in stage_defs
    ]
    combos = policy_stages[0].action_combos
    board = features.build_board_tokens(obs, deck)
    option_sv = policy_stages[0].options
    action_token = features.build_option_tokens(obs, [action])
    return DecisionSample(
        board=board,
        options=option_sv,
        action=action,
        action_combo_index=policy_stages[0].target_index,
        action_combos=combos,
        env_step=int(step.get("env_step", -1)),
        aux_labels=dict(step.get("aux_labels") or {}),
        action_token=action_token,
        policy_stages=policy_stages,
    )


def record_to_sequence(
    record: dict[str, Any],
    *,
    max_context: Optional[int] = None,
    verify_info_set: bool = True,
) -> Optional[GameSequence]:
    """Convert one record; use :func:`convert_record` for drop diagnostics."""
    seq, _reason, _details = convert_record(
        record,
        max_context=max_context,
        verify_info_set=verify_info_set,
    )
    return seq


def convert_record(
    record: dict[str, Any],
    *,
    max_context: Optional[int] = None,
    verify_info_set: bool = True,
) -> tuple[Optional[GameSequence], Optional[str], dict[str, int]]:
    """Convert a record and return ``(sequence, drop_reason, accounting)``."""
    details = {
        "decisions_truncated": 0,
        "policy_targets_padded": 0,
        "policy_targets_truncated": 0,
    }
    if not record.get("info_set_ok", True):
        return None, "info_set_marked_invalid", details
    deck = list(record.get("deck") or [])
    if len(deck) != 60:
        return None, "invalid_deck_length", details
    original_steps = list(record.get("steps") or [])
    if not original_steps:
        return None, "no_steps", details

    max_ctx = int(max_context if max_context is not None else config.MODEL.max_context)
    if max_ctx <= 0:
        return None, "invalid_max_context", details

    raw_targets = record.get("policy_targets")
    policy_targets: Optional[list] = None
    if isinstance(raw_targets, list):
        policy_targets = list(raw_targets)
        if len(policy_targets) < len(original_steps):
            missing = len(original_steps) - len(policy_targets)
            policy_targets.extend([None] * missing)
            details["policy_targets_padded"] = missing
        elif len(policy_targets) > len(original_steps):
            details["policy_targets_truncated"] = len(policy_targets) - len(original_steps)
            policy_targets = policy_targets[: len(original_steps)]

    raw_factorized = record.get("factorized_policy_targets")
    factorized_policy_targets: Optional[list] = None
    if isinstance(raw_factorized, list):
        factorized_policy_targets = list(raw_factorized)
        if len(factorized_policy_targets) < len(original_steps):
            factorized_policy_targets.extend(
                [None] * (len(original_steps) - len(factorized_policy_targets))
            )
        elif len(factorized_policy_targets) > len(original_steps):
            factorized_policy_targets = factorized_policy_targets[
                : len(original_steps)
            ]

    start = max(0, len(original_steps) - max_ctx)
    steps = original_steps[start:]
    details["decisions_truncated"] = start
    if policy_targets is not None:
        policy_targets = policy_targets[start:]
        if len(policy_targets) != len(steps):
            raise AssertionError("step/policy target truncation lost alignment")
    if factorized_policy_targets is not None:
        factorized_policy_targets = factorized_policy_targets[start:]
        if len(factorized_policy_targets) != len(steps):
            raise AssertionError("step/factorized target truncation lost alignment")

    # Ensure Scope-B labels exist for older JSONL that predate replay_import
    # attachment. Public prize_race + post-hoc lethal; losses mask if absent.
    from .blackwell_heads import attach_blackwell_strategy_labels

    attach_blackwell_strategy_labels(steps)

    decisions: list[DecisionSample] = []
    for step in steps:
        try:
            decisions.append(
                featurize_step(step, deck, verify_info_set=verify_info_set)
            )
        except InfoSetViolation:
            return None, "info_set_violation", details
        except features.ActionSpaceTooLarge:
            return None, "action_space_too_large", details
        except Exception:
            return None, "malformed_step", details

    if not decisions:
        return None, "no_decisions", details

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
        policy_targets=policy_targets,
        factorized_policy_targets=factorized_policy_targets,
        target_provenance=dict(record.get("target_provenance") or {}),
    ), None, details


def iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _cache_key(
    jsonl_path: Path,
    max_context: int,
    verify: bool,
    max_games: int = 0,
) -> str:
    st = jsonl_path.stat()
    raw = (
        f"{jsonl_path.resolve()}|{st.st_mtime_ns}|{st.st_size}|{max_context}|"
        f"{int(verify)}|{int(max_games)}|dataset={DATASET_CACHE_SCHEMA_VERSION}|"
        f"features={features.FEATURE_SCHEMA_VERSION}|contract=history"
    )
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
        conversion_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        self.sequences = sequences
        self.jsonl_path = jsonl_path
        self.conversion_stats = dict(conversion_stats or {})

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
            "conversion": dict(self.conversion_stats),
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
            key = _cache_key(path, max_ctx, verify_info_set, max_games)
            cache_path = cache_dir / f"bootstrap_{path.stem}_{key}.pkl"
            if cache_path.is_file():
                try:
                    with cache_path.open("rb") as fh:
                        payload = pickle.load(fh)
                    if (
                        isinstance(payload, dict)
                        and payload.get("dataset_schema") == DATASET_CACHE_SCHEMA_VERSION
                        and payload.get("feature_schema") == features.FEATURE_SCHEMA_VERSION
                    ):
                        return cls(
                            list(payload["sequences"]),
                            jsonl_path=path,
                            conversion_stats=dict(payload.get("conversion_stats") or {}),
                        )
                except Exception:
                    # Rebuild below. A corrupt/incompatible cache is never
                    # accepted as training data.
                    pass

        sequences: list[GameSequence] = []
        from tqdm.auto import tqdm

        stats: dict[str, Any] = {
            "records_total": 0,
            "records_kept": 0,
            "records_dropped": 0,
            "drop_reasons": {},
            "decisions_truncated": 0,
            "policy_targets_padded": 0,
            "policy_targets_truncated": 0,
            "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
            "feature_schema": features.FEATURE_SCHEMA_VERSION,
        }
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                if max_games > 0 and stats["records_total"] >= max_games:
                    break
                stats["records_total"] += 1
                try:
                    obj = json.loads(raw)
                    if not isinstance(obj, dict):
                        raise TypeError("record is not an object")
                    records.append(obj)
                except Exception:
                    reasons = stats["drop_reasons"]
                    reasons["invalid_json"] = reasons.get("invalid_json", 0) + 1
                    stats["records_dropped"] += 1
        for record in tqdm(records, desc=f"featurize {path.name}", unit="seq"):
            seq, reason, details = convert_record(
                record, max_context=max_ctx, verify_info_set=verify_info_set
            )
            for key in (
                "decisions_truncated",
                "policy_targets_padded",
                "policy_targets_truncated",
            ):
                stats[key] += int(details.get(key, 0))
            if seq is None:
                reason = reason or "unknown"
                reasons = stats["drop_reasons"]
                reasons[reason] = reasons.get(reason, 0) + 1
                stats["records_dropped"] += 1
                continue
            sequences.append(seq)
            stats["records_kept"] += 1

        if use_cache and cache_path is not None:
            tmp = cache_path.with_suffix(".tmp")
            with tmp.open("wb") as fh:
                pickle.dump(
                    {
                        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
                        "feature_schema": features.FEATURE_SCHEMA_VERSION,
                        "sequences": sequences,
                        "conversion_stats": stats,
                    },
                    fh,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            tmp.replace(cache_path)

        return cls(sequences, jsonl_path=path, conversion_stats=stats)


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
