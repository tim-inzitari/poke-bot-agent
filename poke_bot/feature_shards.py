"""Bounded, portable bootstrap feature shards.

Large ladder corpora must not be materialized as Python objects while they are
being featurized.  This module converts JSONL records with a bounded process
pool and writes each :class:`~poke_bot.dataset.GameSequence` to an append-only
pickle stream immediately.  Shards can therefore be built on multiple hosts,
validated by digest, and loaded on the trainer only when training starts.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
from array import array
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Optional

from . import features
from .dataset import (
    DATASET_CACHE_SCHEMA_VERSION,
    BootstrapDataset,
    GameSequence,
    convert_record,
)
from .strategic_heads import (
    expanded_strategic_sequence_coverage,
    masked_expanded_strategic_coverage,
    merge_expanded_strategic_coverages,
)


SHARD_FORMAT = "pokebot-bootstrap-feature-shard"
SHARD_FORMAT_VERSION = 1
MANIFEST_FORMAT = "pokebot-bootstrap-feature-manifest"
MANIFEST_FORMAT_VERSION = 1
COMPACT_MODE = "stateless-core-v1"
COMPACT_MODE_TEMPORAL_EXPERT = "temporal-expert-v1"
SUPPORTED_COMPACT_MODES = frozenset({COMPACT_MODE, COMPACT_MODE_TEMPORAL_EXPERT})
MATCHUP_ADAPTER_LEGACY_DATASET_SCHEMA = 4
SETUP_METADATA_LEGACY_DATASET_SCHEMA = 6
PREVIOUS_DATASET_SCHEMA = 7


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _compact_sparse(vector: features.SparseVector) -> None:
    """Replace Python number lists with equivalent fixed-width buffers."""
    vector.index = array("I", vector.index)
    vector.value = array("f", vector.value)
    vector.offset = array("I", vector.offset)


def compact_stateless_sequence(sequence: GameSequence) -> GameSequence:
    """Losslessly compact tensors and drop fields unused by stateless BC.

    The current pure-RL model consumes board/option sparse vectors and hard
    factorized target indices.  Realized previous-action tokens, source text,
    deck copies, and hard-target candidate lists are redundant for that model.
    Soft factorized targets retain their candidate lists for ordering checks.
    """
    sequence.deck = array("I", sequence.deck)  # type: ignore[assignment]
    sequence.source = ""
    sequence.target_provenance = {}
    hard_targets_only = sequence.factorized_policy_targets is None
    for decision in sequence.decisions:
        _compact_sparse(decision.board)
        _compact_sparse(decision.options)
        decision.action = []
        decision.action_token = None
        decision.aux_labels = {}
        if hard_targets_only:
            decision.action_combos = []
        for stage in decision.policy_stages:
            _compact_sparse(stage.options)
            if hard_targets_only:
                stage.action_combos = []
                # Guide labels are aligned to the candidate list and cannot be
                # consumed after stateless expert-shard compaction drops it.
                stage.guide_target_index = -1
                stage.guide_confidence = 0.0
    return sequence


def compact_temporal_expert_sequence(sequence: GameSequence) -> GameSequence:
    """Compact an expert sequence without destroying temporal/aux targets.

    Unlike :func:`compact_stateless_sequence`, this representation retains the
    shifted previous-action token, privileged target-only labels, and collapsed
    Alakazam guide targets.  Opponent-private values remain exclusively under
    ``DecisionSample.aux_labels``; they are never copied into board features.
    """
    sequence.deck = array("I", sequence.deck)  # type: ignore[assignment]
    sequence.source = ""
    sequence.target_provenance = {}
    hard_targets_only = sequence.factorized_policy_targets is None
    for decision in sequence.decisions:
        _compact_sparse(decision.board)
        _compact_sparse(decision.options)
        if decision.action_token is None:
            raise ValueError("temporal expert decision is missing its action token")
        _compact_sparse(decision.action_token)
        decision.action = []
        if hard_targets_only:
            decision.action_combos = []
        for stage in decision.policy_stages:
            _compact_sparse(stage.options)
            if hard_targets_only:
                stage.action_combos = []
    return sequence


def _target_coverage(sequence: GameSequence) -> dict[str, int]:
    coverage = {
        "temporal_action_rows": 0,
        "opponent_hand_rows": 0,
        "opponent_remainder_rows": 0,
        "opponent_private_prize_rows": 0,
        "lethal_threat_rows": 0,
        "prize_race_rows": 0,
        "guide_rows": 0,
        "combo_state_rows": 0,
    }
    for decision in sequence.decisions:
        # Legacy compact shards (and callers inspecting them before full
        # hydration) may contain decision-like rows without the newer temporal
        # or auxiliary attributes.  Missing labels mean zero coverage; they do
        # not make an otherwise readable legacy manifest unfilterable.
        if getattr(decision, "action_token", None) is not None:
            coverage["temporal_action_rows"] += 1
        aux = dict(getattr(decision, "aux_labels", None) or {})
        if aux.get("opp_hand") is not None:
            coverage["opponent_hand_rows"] += 1
        if aux.get("opp_hidden_remainder") is not None or any(
            aux.get(key) is not None
            for key in ("opp_hand", "opp_deck_order", "opp_prizes")
        ):
            coverage["opponent_remainder_rows"] += 1
        if aux.get("opp_prizes") is not None:
            coverage["opponent_private_prize_rows"] += 1
        if aux.get("lethal_threat") is not None:
            coverage["lethal_threat_rows"] += 1
        if aux.get("prize_race") is not None:
            coverage["prize_race_rows"] += 1
        coverage["guide_rows"] += sum(
            int(getattr(stage, "guide_target_index", -1) >= 0)
            for stage in (getattr(decision, "policy_stages", None) or ())
        )
        if aux.get("combo_state") is not None:
            coverage["combo_state_rows"] += 1
    return coverage


def _convert_raw(
    raw: str,
    max_context: int,
    verify_info_set: bool,
    allowed_sources: tuple[str, ...],
    compact_mode: str,
    required_archetype: str,
) -> tuple[Optional[GameSequence], Optional[str], dict[str, int]]:
    try:
        record = json.loads(raw)
        if not isinstance(record, dict):
            raise TypeError("record is not an object")
    except Exception:
        return None, "invalid_json", {
            "decisions_truncated": 0,
            "policy_targets_padded": 0,
            "policy_targets_truncated": 0,
        }
    if str(record.get("source") or "") not in allowed_sources:
        return None, "source_date_mismatch", {
            "decisions_truncated": 0,
            "policy_targets_padded": 0,
            "policy_targets_truncated": 0,
        }
    if required_archetype and str(record.get("archetype") or "").casefold() != required_archetype:
        return None, "archetype_mismatch", {
            "decisions_truncated": 0,
            "policy_targets_padded": 0,
            "policy_targets_truncated": 0,
        }
    sequence, reason, details = convert_record(
        record,
        max_context=max_context,
        verify_info_set=verify_info_set,
    )
    if sequence is not None:
        if compact_mode == COMPACT_MODE:
            compact_stateless_sequence(sequence)
        elif compact_mode == COMPACT_MODE_TEMPORAL_EXPERT:
            compact_temporal_expert_sequence(sequence)
        else:  # guarded by write_feature_shard; defensive for worker callers
            raise ValueError(f"unsupported compact mode: {compact_mode}")
    return sequence, reason, details


def _iter_nonempty_lines(path: Path, max_records: int = 0) -> Iterator[str]:
    emitted = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            yield raw
            emitted += 1
            if max_records > 0 and emitted >= max_records:
                return


def _new_stats(compact_mode: str = COMPACT_MODE) -> dict[str, Any]:
    return {
        "records_total": 0,
        "records_kept": 0,
        "records_dropped": 0,
        "decisions_kept": 0,
        "drop_reasons": {},
        "decisions_truncated": 0,
        "policy_targets_padded": 0,
        "policy_targets_truncated": 0,
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": features.FEATURE_SCHEMA_VERSION,
        "compact_mode": compact_mode,
        "target_coverage": {
            "temporal_action_rows": 0,
            "opponent_hand_rows": 0,
            "opponent_remainder_rows": 0,
            "opponent_private_prize_rows": 0,
            "lethal_threat_rows": 0,
            "prize_race_rows": 0,
            "guide_rows": 0,
        },
        "expanded_strategic_targets": masked_expanded_strategic_coverage(0),
    }


def _account(
    result: tuple[Optional[GameSequence], Optional[str], dict[str, int]],
    stats: dict[str, Any],
) -> Optional[GameSequence]:
    sequence, reason, details = result
    stats["records_total"] += 1
    for key in (
        "decisions_truncated",
        "policy_targets_padded",
        "policy_targets_truncated",
    ):
        stats[key] += int(details.get(key, 0))
    if sequence is None:
        reason = reason or "unknown"
        drops = stats["drop_reasons"]
        drops[reason] = int(drops.get(reason, 0)) + 1
        stats["records_dropped"] += 1
        return None
    stats["records_kept"] += 1
    stats["decisions_kept"] += len(sequence)
    coverage = stats["target_coverage"]
    for key, count in _target_coverage(sequence).items():
        coverage[key] = int(coverage.get(key, 0)) + int(count)
    stats["expanded_strategic_targets"] = (
        merge_expanded_strategic_coverages(
            (
                stats["expanded_strategic_targets"],
                expanded_strategic_sequence_coverage(sequence.decisions),
            )
        )
    )
    return sequence


def write_feature_shard(
    jsonl_path: Path,
    output_path: Path,
    *,
    source_dates: list[str],
    max_context: int,
    workers: int,
    max_in_flight: int = 0,
    max_records: int = 0,
    verify_info_set: bool = True,
    compact_mode: str = COMPACT_MODE,
    required_archetype: str = "",
) -> dict[str, Any]:
    """Build one atomic feature stream with bounded parent/worker memory."""
    jsonl_path = Path(jsonl_path).resolve()
    output_path = Path(output_path).resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(jsonl_path)
    if workers <= 0:
        raise ValueError("workers must be positive")
    if output_path.exists():
        raise FileExistsError(output_path)
    compact_mode = str(compact_mode).strip()
    if compact_mode not in SUPPORTED_COMPACT_MODES:
        raise ValueError(f"unsupported compact mode: {compact_mode}")
    required_archetype = str(required_archetype).strip().casefold()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.name}.partial.{os.getpid()}")
    sidecar = output_path.with_suffix(output_path.suffix + ".json")
    sidecar_tmp = sidecar.with_name(f".{sidecar.name}.partial.{os.getpid()}")
    in_flight = max(workers, max_in_flight or workers * 2)
    stats = _new_stats(compact_mode)
    header = {
        "format": SHARD_FORMAT,
        "format_version": SHARD_FORMAT_VERSION,
        "dataset_schema": DATASET_CACHE_SCHEMA_VERSION,
        "feature_schema": features.FEATURE_SCHEMA_VERSION,
        "compact_mode": compact_mode,
        "required_archetype": required_archetype or None,
        "source_jsonl": jsonl_path.name,
        "source_jsonl_bytes": jsonl_path.stat().st_size,
        "source_dates": list(source_dates),
        "max_context": int(max_context),
    }
    allowed_sources = tuple(
        f"pokemon-tcg-ai-battle-episodes-{day}" for day in source_dates
    )
    started = time.time()
    from tqdm.auto import tqdm

    lines = iter(_iter_nonempty_lines(jsonl_path, max_records=max_records))
    try:
        with partial.open("xb") as output, ProcessPoolExecutor(
            max_workers=workers
        ) as pool:
            pickle.dump(header, output, protocol=pickle.HIGHEST_PROTOCOL)
            pending: deque[Future] = deque()

            def submit_one() -> bool:
                try:
                    raw = next(lines)
                except StopIteration:
                    return False
                pending.append(
                    pool.submit(
                        _convert_raw,
                        raw,
                        max_context,
                        verify_info_set,
                        allowed_sources,
                        compact_mode,
                        required_archetype,
                    )
                )
                return True

            while len(pending) < in_flight and submit_one():
                pass
            with tqdm(desc=f"feature {jsonl_path.name}", unit="seq") as progress:
                while pending:
                    result = pending.popleft().result()
                    sequence = _account(result, stats)
                    if sequence is not None:
                        pickle.dump(sequence, output, protocol=pickle.HIGHEST_PROTOCOL)
                    progress.update(1)
                    progress.set_postfix(
                        kept=stats["records_kept"],
                        drop=stats["records_dropped"],
                    )
                    submit_one()
            footer = {
                "format": SHARD_FORMAT + "-footer",
                "format_version": SHARD_FORMAT_VERSION,
                "stats": stats,
            }
            pickle.dump(footer, output, protocol=pickle.HIGHEST_PROTOCOL)
            output.flush()
            os.fsync(output.fileno())
        partial.replace(output_path)
        digest = _sha256(output_path)
        metadata = {
            **header,
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": digest,
            "stats": stats,
            "workers": workers,
            "max_in_flight": in_flight,
            "elapsed_seconds": time.time() - started,
        }
        sidecar_tmp.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        sidecar_tmp.replace(sidecar)
        return metadata
    except BaseException:
        partial.unlink(missing_ok=True)
        sidecar_tmp.unlink(missing_ok=True)
        raise


def iter_feature_shard(path: Path) -> Iterator[GameSequence]:
    """Yield a fully validated shard stream, rejecting truncation/trailing data."""
    with Path(path).open("rb") as handle:
        header = pickle.load(handle)
        if not isinstance(header, dict) or header.get("format") != SHARD_FORMAT:
            raise ValueError(f"invalid feature shard header: {path}")
        if int(header.get("format_version", -1)) != SHARD_FORMAT_VERSION:
            raise ValueError(f"unsupported feature shard version: {path}")
        dataset_schema = int(header.get("dataset_schema", -1))
        if dataset_schema not in {
            DATASET_CACHE_SCHEMA_VERSION,
            PREVIOUS_DATASET_SCHEMA,
            SETUP_METADATA_LEGACY_DATASET_SCHEMA,
            MATCHUP_ADAPTER_LEGACY_DATASET_SCHEMA,
        }:
            raise ValueError(f"dataset schema mismatch: {path}")
        if int(header.get("feature_schema", -1)) != features.FEATURE_SCHEMA_VERSION:
            raise ValueError(f"feature schema mismatch: {path}")
        count = 0
        while True:
            try:
                item = pickle.load(handle)
            except EOFError as exc:
                raise ValueError(f"feature shard is missing its footer: {path}") from exc
            if isinstance(item, dict) and item.get("format") == SHARD_FORMAT + "-footer":
                expected = int((item.get("stats") or {}).get("records_kept", -1))
                if expected != count:
                    raise ValueError(
                        f"feature shard count mismatch: expected {expected}, loaded {count}"
                    )
                if handle.read(1):
                    raise ValueError(f"trailing bytes after feature shard footer: {path}")
                return
            if not isinstance(item, GameSequence):
                raise ValueError(f"unexpected feature shard item: {type(item)!r}")
            if dataset_schema == MATCHUP_ADAPTER_LEGACY_DATASET_SCHEMA:
                # Schema 5 added only dormant matchup-adapter routing fields.
                # A schema-4 expert shard contains no audited routing ticket,
                # so migrate it fail-closed to UNKNOWN/no-ticket.  All policy,
                # temporal, belief, lethal, and prize targets are unchanged.
                if not hasattr(item, "matchup_adapter_training_ticket"):
                    item.matchup_adapter_training_ticket = {}
                for decision in item.decisions:
                    if not hasattr(decision, "matchup_adapter_oracle_route"):
                        decision.matchup_adapter_oracle_route = -1
                    if not hasattr(decision, "matchup_adapter_public_route"):
                        decision.matchup_adapter_public_route = -1
            if dataset_schema == SETUP_METADATA_LEGACY_DATASET_SCHEMA:
                # Schema 7 added exact setup SelectContext/STOP metadata only.
                # Schema-6 protected expert shards remain valid for every
                # pre-existing causal target. Migrate the new setup objective
                # fail-closed: UNKNOWN context masks those rows, while the
                # unrelated expert objectives retain their original labels.
                for decision in item.decisions:
                    for stage in decision.policy_stages:
                        if not hasattr(stage, "select_context"):
                            stage.select_context = -1
                        if not hasattr(stage, "selected_is_stop"):
                            stage.selected_is_stop = False
            count += 1
            yield item


def load_feature_manifest(
    manifest_path: Path,
    *,
    verify_hashes: bool = True,
) -> BootstrapDataset:
    """Load ordered compact shards described by a portable JSON manifest."""
    manifest_path = Path(manifest_path).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != MANIFEST_FORMAT:
        raise ValueError("invalid feature manifest format")
    if int(payload.get("format_version", -1)) != MANIFEST_FORMAT_VERSION:
        raise ValueError("unsupported feature manifest version")
    shards = list(payload.get("shards") or [])
    if not shards:
        raise ValueError("feature manifest contains no shards")
    sequences: list[GameSequence] = []
    manifest_mode = str(payload.get("compact_mode") or COMPACT_MODE)
    if manifest_mode not in SUPPORTED_COMPACT_MODES:
        raise ValueError(f"unsupported manifest compact mode: {manifest_mode}")
    combined = _new_stats(manifest_mode)
    combined["records_total"] = 0
    from tqdm.auto import tqdm

    for row in shards:
        path = (manifest_path.parent / str(row["path"])).resolve()
        expected_hash = str(row.get("sha256") or "")
        if verify_hashes and _sha256(path) != expected_hash:
            raise ValueError(f"feature shard digest mismatch: {path}")
        stats = dict(row.get("stats") or {})
        for key in (
            "records_total",
            "records_kept",
            "records_dropped",
            "decisions_kept",
            "decisions_truncated",
            "policy_targets_padded",
            "policy_targets_truncated",
        ):
            combined[key] += int(stats.get(key, 0))
        for reason, count in dict(stats.get("drop_reasons") or {}).items():
            combined["drop_reasons"][reason] = (
                int(combined["drop_reasons"].get(reason, 0)) + int(count)
            )
        combined_coverage = combined["target_coverage"]
        for key, count in dict(stats.get("target_coverage") or {}).items():
            combined_coverage[key] = int(combined_coverage.get(key, 0)) + int(count)
        decisions = int(stats.get("decisions_kept", 0))
        expanded = stats.get("expanded_strategic_targets")
        if expanded is None:
            expanded = masked_expanded_strategic_coverage(decisions)
        combined["expanded_strategic_targets"] = (
            merge_expanded_strategic_coverages(
                (combined["expanded_strategic_targets"], expanded)
            )
        )
        expected = int(stats.get("records_kept", 0))
        before = len(sequences)
        for sequence in tqdm(
            iter_feature_shard(path),
            total=expected or None,
            desc=f"load {path.name}",
            unit="seq",
        ):
            sequences.append(sequence)
        if expected and len(sequences) - before != expected:
            raise ValueError(f"manifest count mismatch for {path}")
    return BootstrapDataset(sequences=sequences, conversion_stats=combined)
