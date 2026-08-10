"""Archetype-generic Recursive Turn Planner training pipeline.

One recipe for every specialist (Alakazam, Marnie, Crustle, future decks):

1. Bind a frozen parent CABT checkpoint plus a legacy shard or sealed complete-action corpus to a ``specialist_id``
2. Encode features with the frozen parent
3. Train an RTP sidecar (optional PokeRLM sidecar)
4. Emit a per-archetype receipt + env load hints

Does not rewrite GOAL.md, parent tensors, selectors, or managed training.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from poke_bot.poke_rlm.training.shadow_train import (
    PokeRLMTrainConfig,
    train_poke_rlm_shadow,
)
from poke_bot.recursive_turn_planner.training.shadow_train import (
    RTPDecisionBatch,
    RTPTrainConfig,
    encode_sequences_to_batches,
    make_synthetic_batches,
    required_recursive_passes,
    split_batches_by_game,
    train_rtp_shadow,
)
from poke_bot.recursive_turn_planner.config import RTP_MAX_AUTHORIZED_NEURAL_PASSES


PIPELINE_SCHEMA = "poke_bot.recursive_turn_planner.archetype_pipeline/v1"
REGISTRY_SCHEMA = "poke_bot.recursive_turn_planner.archetype_registry/v1"
R197_COMPLETE_ACTION_ROW_SCHEMA = "poke_bot.rtp_complete_ordered_action_row/v1"
R197_COMPLETE_ACTION_CORPUS_SCHEMA = "poke_bot.rtp_complete_action_shadow_corpus/v1"
R197_COMPLETE_ACTION_CAP = 1024
# This is the canonical global ceiling in ``RTPConfig`` as well.  Keep the
# explicit r197 contract tied to it so a job cannot advertise more work than
# the planner can ever be authorized to execute.
R197_MAX_NEURAL_PASSES = RTP_MAX_AUTHORIZED_NEURAL_PASSES
R197_SPLIT_SEED = 5_000_000


@dataclass
class ArchetypeRTPJob:
    """One specialist's RTP train job. Paths may be empty for synthetic smoke."""

    specialist_id: str
    display_name: str = ""
    parent_checkpoint: str = ""
    training_shard: str = ""
    #: Optional expected hash.  When supplied it must match checkpoint bytes.
    parent_digest: str = ""
    #: Optional expected hash.  The observed shard digest is always bound.
    training_shard_digest: str = ""
    #: Immutable output directory from ``materialize_r197_complete_action_corpus``.
    #: This is a separate input shape from a legacy compact training shard.
    complete_action_corpus: str = ""
    #: Required for r197 so the exact derivative manifest is known before
    #: training begins, not merely copied into a post-hoc receipt.
    complete_action_corpus_manifest_digest: str = ""
    #: Optional expected digest for the immutable corpus RECEIPT.json.
    complete_action_corpus_receipt_digest: str = ""
    #: Optional source-pointer digest expected in the corpus manifest.
    complete_action_corpus_source_pointer_digest: str = ""
    #: Pre-training whole-episode selection receipt digests.  The stage binds
    #: these to the candidate ID before any sidecar gradients are taken.
    complete_action_corpus_selection_plan_digest: str = ""
    complete_action_corpus_train_selection_digest: str = ""
    complete_action_corpus_heldout_selection_digest: str = ""
    profile: str = "pure_rl"
    d_model: int = 96
    max_games: int = 256
    epochs: int = 2
    lr: float = 1e-3
    seed: int = 0
    device: str = "cpu"
    also_poke_rlm: bool = False
    complexity_option_threshold: int = 8
    complexity_entropy_threshold: float = 1.5
    num_plan_candidates: int = 4
    max_recursion_depth: int = 2
    max_neural_passes: int = 4
    heldout_fraction: float = 0.20
    #: Host training requires exact whole-action support; old factorized-only
    #: shards may be inspected but cannot claim runtime alignment.
    require_complete_ordered_actions: bool = True
    max_runtime_action_combos: int = 256
    #: r197 materialization uses a stable whole-episode split namespace.
    split_seed: int = 0
    #: r197 uses bounded whole-episode selection within its immutable split.
    #: Zero means uncapped for legacy jobs only.
    max_train_games: int = 0
    max_heldout_games: int = 0
    max_train_batches: int = 0
    max_heldout_batches: int = 0
    enabled: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        sid = str(self.specialist_id or "").strip().lower().replace("_", "-")
        if not sid:
            raise ValueError("specialist_id is required")
        self.specialist_id = sid
        if not self.display_name:
            self.display_name = sid
        if int(self.d_model) < 1:
            raise ValueError("d_model must be positive")
        if int(self.max_games) < 1:
            raise ValueError("max_games must be positive")
        if int(self.max_neural_passes) < 1:
            raise ValueError("max_neural_passes must be positive")
        if int(self.max_neural_passes) > RTP_MAX_AUTHORIZED_NEURAL_PASSES:
            raise ValueError(
                "max_neural_passes exceeds the global authorized ceiling "
                f"({RTP_MAX_AUTHORIZED_NEURAL_PASSES})"
            )
        if int(self.num_plan_candidates) < 1:
            raise ValueError("num_plan_candidates must be positive")
        if int(self.max_recursion_depth) < 0:
            raise ValueError("max_recursion_depth must be non-negative")
        if str(self.profile).strip().lower() == "pure_rl_r197":
            if not str(self.parent_checkpoint or "").strip():
                raise ValueError("pure_rl_r197 requires parent_checkpoint")
            if int(self.d_model) != 96:
                raise ValueError("pure_rl_r197 requires d_model=96")
            if int(self.num_plan_candidates) != 4:
                raise ValueError("pure_rl_r197 requires exactly four plan candidates")
            if int(self.max_recursion_depth) != 2:
                raise ValueError("pure_rl_r197 requires max_recursion_depth=2")
            if int(self.max_neural_passes) != R197_MAX_NEURAL_PASSES:
                raise ValueError("pure_rl_r197 requires max_neural_passes=256")
            if bool(self.also_poke_rlm):
                raise ValueError("pure_rl_r197 trains only the RTP shadow sidecar")
            if abs(float(self.heldout_fraction) - 0.20) > 1e-12:
                raise ValueError("pure_rl_r197 requires heldout_fraction=0.20")
            if not str(self.parent_digest or "").strip():
                raise ValueError("pure_rl_r197 requires a supplied parent_digest")
            if not str(self.complete_action_corpus or "").strip():
                raise ValueError("pure_rl_r197 requires complete_action_corpus")
            if not str(self.complete_action_corpus_manifest_digest or "").strip():
                raise ValueError(
                    "pure_rl_r197 requires complete_action_corpus_manifest_digest"
                )
            if not str(self.complete_action_corpus_receipt_digest or "").strip():
                raise ValueError(
                    "pure_rl_r197 requires complete_action_corpus_receipt_digest"
                )
            if not str(self.complete_action_corpus_source_pointer_digest or "").strip():
                raise ValueError(
                    "pure_rl_r197 requires complete_action_corpus_source_pointer_digest"
                )
            for field_name in (
                "complete_action_corpus_selection_plan_digest",
                "complete_action_corpus_train_selection_digest",
                "complete_action_corpus_heldout_selection_digest",
            ):
                if not str(getattr(self, field_name) or "").strip():
                    raise ValueError(f"pure_rl_r197 requires {field_name}")
            if str(self.training_shard or "").strip():
                raise ValueError(
                    "pure_rl_r197 consumes only the canonical complete-action corpus, "
                    "not a compact training_shard"
                )
            if not bool(self.require_complete_ordered_actions):
                raise ValueError("pure_rl_r197 forbids factorized-action fallback")
            if int(self.max_runtime_action_combos) != R197_COMPLETE_ACTION_CAP:
                raise ValueError(
                    "pure_rl_r197 requires max_runtime_action_combos=1024"
                )
            if int(self.split_seed) != R197_SPLIT_SEED:
                raise ValueError("pure_rl_r197 requires split_seed=5_000_000")
            if int(self.max_train_games) != 512:
                raise ValueError("pure_rl_r197 requires max_train_games=512")
            if int(self.max_heldout_games) != 128:
                raise ValueError("pure_rl_r197 requires max_heldout_games=128")
            if int(self.max_train_batches) != 32_000:
                raise ValueError("pure_rl_r197 requires max_train_batches=32000")
            if int(self.max_heldout_batches) != 8_000:
                raise ValueError("pure_rl_r197 requires max_heldout_batches=8000")
        if int(self.max_runtime_action_combos) < 1:
            raise ValueError("max_runtime_action_combos must be positive")
        for field_name in (
            "max_train_games",
            "max_heldout_games",
            "max_train_batches",
            "max_heldout_batches",
        ):
            if int(getattr(self, field_name)) < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if not 0.0 <= float(self.heldout_fraction) < 1.0:
            raise ValueError("heldout_fraction must be in [0, 1)")

    @property
    def ready_for_host_train(self) -> bool:
        return bool(self.parent_checkpoint) and bool(
            self.complete_action_corpus or self.training_shard
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ArchetypeRTPResult:
    specialist_id: str
    source: str
    out_dir: str
    n_batches: int
    rtp_checkpoint: str
    rtp_receipt: str
    poke_rlm_checkpoint: str = ""
    poke_rlm_receipt: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)
    serving_eligible: bool = False

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _r197_training_code_digests() -> dict[str, str]:
    """Bind the exact loader/objective/checkpoint bytes to an r197 plan."""
    root = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        root / "poke_bot/recursive_turn_planner/training/shadow_train.py",
        root / "poke_bot/recursive_turn_planner/training/losses.py",
        root / "poke_bot/recursive_turn_planner/training/checkpoint.py",
        root / "poke_bot/recursive_turn_planner/profiles.py",
        root / "poke_bot/recursive_turn_planner/planner.py",
        root / "poke_bot/recursive_turn_planner/r197_corpus.py",
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("r197 training source is missing: " + ", ".join(missing))
    return {
        str(path.relative_to(root)): _sha256_file(path)
        for path in paths
    }


def _normalize_sha256(value: str, *, field: str) -> str:
    """Return canonical ``sha256:<lowercase hex>`` or fail closed."""
    raw = str(value or "").strip().lower()
    if raw.startswith("sha256:"):
        raw = raw[len("sha256:") :]
    if len(raw) != 64 or any(char not in "0123456789abcdef" for char in raw):
        raise ValueError(f"{field} must be a SHA-256 digest")
    return "sha256:" + raw


def _verify_expected_digest(
    *,
    expected: str,
    actual: str,
    field: str,
) -> bool:
    """Verify an optionally supplied digest without accepting a nearby file."""
    if not str(expected or "").strip():
        return False
    normalized = _normalize_sha256(expected, field=field)
    if normalized != actual:
        raise ValueError(
            f"{field} mismatch: expected {normalized}, observed {actual}"
        )
    return True


def _canonical_json_sha256(value: Any) -> str:
    """Digest the corpus's canonical JSON representation exactly."""
    try:
        encoded = (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not readable canonical JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _declared_split_value(manifest: Mapping[str, Any], field: str) -> Any:
    """Read a materializer split field without silently inventing defaults."""
    if field in manifest:
        return manifest.get(field)
    split = manifest.get("split")
    if isinstance(split, Mapping) and field in split:
        return split.get(field)
    return None


def _declared_digest_fields(value: Any, *, prefix: str = "") -> dict[str, str]:
    """Collect checksum declarations already bound by an immutable manifest."""
    found: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            if str(key).endswith("sha256") and isinstance(item, str):
                try:
                    found[child] = _normalize_sha256(item, field=child)
                except ValueError:
                    # The materializer's own verifier will reject malformed
                    # declarations.  Preserve no fake digest in provenance.
                    continue
            found.update(_declared_digest_fields(item, prefix=child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_declared_digest_fields(item, prefix=f"{prefix}[{index}]"))
    return found


def _manifest_pointer_digest(manifest: Mapping[str, Any]) -> str:
    """Return the corpus source-pointer digest if its manifest declares one."""
    candidates: list[Any] = [
        manifest.get("source_pointer_sha256"),
        manifest.get("pointer_sha256"),
    ]
    for nested_key in ("source_pointer", "source", "input"):
        nested = manifest.get(nested_key)
        if isinstance(nested, Mapping):
            candidates.extend(
                [nested.get("sha256"), nested.get("pointer_sha256")]
            )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return _normalize_sha256(candidate, field="corpus source pointer digest")
    return ""


def _r197_selection_priority(
    episode_id: str,
    *,
    split: str,
    selection_seed: int,
) -> str:
    material = (
        f"r197-training-selection/v1\0{int(selection_seed)}\0"
        f"{str(split)}\0{str(episode_id)}"
    ).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _select_r197_episodes(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    limit: int,
    selection_seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Choose a bounded deterministic set of whole episodes from one split.

    The selection scans rows but retains only unique episode identities and a
    fixed-size priority heap.  It never samples individual decisions, so both
    seats and every retained decision of a selected game stay in the corpus's
    original train/heldout partition.
    """
    if int(limit) < 1:
        raise ValueError("r197 whole-episode selection limit must be positive")
    seen: set[str] = set()
    # Negative integer priority makes heap[0] the *worst* (largest hash)
    # selected episode, which can be replaced in O(log(limit)).
    heap: list[tuple[int, str, str]] = []
    for row in rows:
        episode_id = str(row.get("episode_id") or "").strip()
        if not episode_id:
            raise ValueError("r197 complete-action row has no episode_id")
        if episode_id in seen:
            continue
        seen.add(episode_id)
        priority = _r197_selection_priority(
            episode_id,
            split=split,
            selection_seed=selection_seed,
        )
        entry = (-int(priority, 16), episode_id, priority)
        if len(heap) < int(limit):
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            # A smaller real priority has a less-negative heap key.
            heapq.heapreplace(heap, entry)
    ordered = [
        episode_id
        for _negative_priority, episode_id, _priority in sorted(
            heap, key=lambda item: (item[2], item[1])
        )
    ]
    if not ordered:
        raise RuntimeError(f"r197 {split} corpus has no selectable episodes")
    return ordered, {
        "schema": "poke_bot.recursive_turn_planner.r197_whole_episode_selection/v1",
        "split": str(split),
        "selection_seed": int(selection_seed),
        "episode_limit": int(limit),
        "available_episodes": len(seen),
        "selected_episodes": len(ordered),
        "selection_sha256": _canonical_json_sha256(
            {
                "schema": "poke_bot.recursive_turn_planner.r197_whole_episode_selection/v1",
                "split": str(split),
                "selection_seed": int(selection_seed),
                "episode_ids": ordered,
            }
        ),
        "selected_episode_ids_sha256": _canonical_json_sha256(ordered),
        "row_level_sampling": False,
    }


def _iter_r197_split_identities(path: Path, *, split: str) -> Iterable[dict[str, Any]]:
    """Stream only the identities needed for bounded whole-game selection.

    Call this only after ``verify_r197_complete_action_manifest`` has checked
    the full stream's byte hash.  Selected rows are later passed through the
    materializer's complete row/action-space verifier; this lightweight pass
    avoids recomputing every candidate action fingerprint merely to select a
    bounded set of episode ids.
    """
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(
                    f"blank r197 {split} corpus row during selection at {line_number}"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid r197 {split} corpus row during selection at {line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"non-object r197 {split} corpus row during selection at {line_number}"
                )
            if (
                row.get("schema") != R197_COMPLETE_ACTION_ROW_SCHEMA
                or row.get("corpus_schema") != R197_COMPLETE_ACTION_CORPUS_SCHEMA
                or row.get("split") != split
            ):
                raise ValueError(
                    f"r197 {split} corpus row has an unexpected schema/split"
                )
            if not str(row.get("episode_id") or "").strip():
                raise ValueError("r197 complete-action row has no episode_id")
            yield row


def _plan_r197_batch_cap(
    counts: Mapping[str, int],
    *,
    episode_order: Sequence[str],
    batch_cap: int,
    split: str,
) -> tuple[list[str], dict[str, Any]]:
    """Apply a cap only by accepting or rejecting complete episodes.

    Every materialized complete-action row maps to one RTP batch after the
    corpus verifier has passed.  Counting rows therefore gives an exact,
    pre-encoding bound without row-level truncation or a cross-window target.
    """
    if int(batch_cap) < 1:
        raise ValueError("r197 batch cap must be positive")
    kept: list[str] = []
    skipped_due_cap: list[str] = []
    used = 0
    for episode_id in episode_order:
        count = int(counts.get(episode_id, 0))
        if count <= 0:
            raise ValueError(
                f"r197 selected {split} episode has no complete-action rows: {episode_id!r}"
            )
        if used + count <= int(batch_cap):
            kept.append(episode_id)
            used += count
        else:
            skipped_due_cap.append(episode_id)
    if not kept:
        raise RuntimeError(
            f"r197 {split} batch cap cannot contain one complete selected episode"
        )
    return kept, {
        "batch_cap": int(batch_cap),
        "candidate_episode_count": len(episode_order),
        "retained_episode_count": len(kept),
        "skipped_episode_count_due_to_batch_cap": len(skipped_due_cap),
        "retained_batch_count_pre_encoding": used,
        "retained_episode_ids_sha256": _canonical_json_sha256(kept),
        "skipped_episode_ids_sha256": _canonical_json_sha256(skipped_due_cap),
        "row_level_sampling": False,
        "cross_window_dynamics_target": False,
    }


def _apply_r197_batch_cap(
    sequences: Sequence[Any],
    records: Mapping[tuple[str, int, int], Mapping[str, Any]],
    *,
    episode_order: Sequence[str],
    batch_cap: int,
    split: str,
) -> tuple[list[Any], dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    """Apply the precomputed whole-episode cap to encoded source sequences."""
    counts: dict[str, int] = {}
    for episode_id, _seat, _step in records:
        counts[episode_id] = counts.get(episode_id, 0) + 1
    kept, cap_provenance = _plan_r197_batch_cap(
        counts,
        episode_order=episode_order,
        batch_cap=batch_cap,
        split=split,
    )
    kept_set = set(kept)
    capped_sequences = [
        sequence for sequence in sequences if str(sequence.episode_id) in kept_set
    ]
    capped_records = {
        key: dict(value)
        for key, value in records.items()
        if key[0] in kept_set
    }
    return capped_sequences, capped_records, cap_provenance


def plan_r197_complete_action_selection(
    complete_action_corpus: Path | str,
    *,
    expected_manifest_digest: str,
    expected_receipt_digest: str,
    expected_source_pointer_digest: str,
    selection_seed: int,
    max_train_games: int,
    max_heldout_games: int,
    max_train_batches: int,
    max_heldout_batches: int,
    heldout_fraction: float = 0.20,
) -> dict[str, Any]:
    """Precompute the immutable r197 training selection before gradients.

    Staging calls this read-only function immediately after corpus
    materialization, binds ``selection_plan_sha256`` and the two retained
    split digests into its candidate contract/ID, then passes those same
    expected values to :class:`ArchetypeRTPJob`.  Training recomputes and
    checks them again before loading parent features.
    """
    if int(max_train_games) != 512 or int(max_heldout_games) != 128:
        raise ValueError("r197 selection requires exactly 512 train and 128 heldout games")
    if int(max_train_batches) != 32_000 or int(max_heldout_batches) != 8_000:
        raise ValueError("r197 selection requires exactly 32000/8000 batch caps")
    if abs(float(heldout_fraction) - 0.20) > 1e-12:
        raise ValueError("r197 selection requires heldout_fraction=0.20")
    raw_path = Path(complete_action_corpus).expanduser().resolve()
    manifest_path = (
        raw_path if raw_path.name == "MANIFEST.json" else raw_path / "MANIFEST.json"
    )
    corpus_dir = manifest_path.parent
    if not manifest_path.is_file():
        raise FileNotFoundError(f"complete-action corpus manifest missing: {manifest_path}")
    manifest_digest = _sha256_file(manifest_path)
    _verify_expected_digest(
        expected=expected_manifest_digest,
        actual=manifest_digest,
        field="complete_action_corpus_manifest_digest",
    )
    manifest = _read_json_object(manifest_path, label="complete-action corpus MANIFEST")
    expected_files = {
        "train": corpus_dir / "train.complete-actions.jsonl",
        "heldout": corpus_dir / "heldout.complete-actions.jsonl",
        "overflow": corpus_dir / "action-space-too-large.jsonl",
        "verified_identities": corpus_dir / "verified-episode-seats.jsonl",
        "episode_splits": corpus_dir / "episode-splits.jsonl",
        "receipt": corpus_dir / "RECEIPT.json",
    }
    missing = [name for name, path in expected_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "complete-action corpus is incomplete; missing " + ", ".join(missing)
        )
    file_digests = {name: _sha256_file(path) for name, path in expected_files.items()}
    _verify_expected_digest(
        expected=expected_receipt_digest,
        actual=file_digests["receipt"],
        field="complete_action_corpus_receipt_digest",
    )
    pointer_digest = _manifest_pointer_digest(manifest)
    _verify_expected_digest(
        expected=expected_source_pointer_digest,
        actual=pointer_digest,
        field="complete_action_corpus_source_pointer_digest",
    )
    split_payload = manifest.get("split")
    action_space = manifest.get("action_space")
    if (
        (manifest.get("eligibility") or {}).get("kaggle_replay_eligible")
        is not False
        or not isinstance(split_payload, Mapping)
        or split_payload.get("unit") != "episode_id"
        or split_payload.get("source_disjoint") is not True
        or int(split_payload.get("seed", -1)) != R197_SPLIT_SEED
        or abs(float(split_payload.get("heldout_fraction")) - 0.20) > 1e-12
        or not isinstance(action_space, Mapping)
        or int(action_space.get("max_action_combos", -1))
        != R197_COMPLETE_ACTION_CAP
        or action_space.get("factorized_policy_stage_substitution_allowed")
        is not False
    ):
        raise ValueError("complete-action corpus does not satisfy the r197 contract")
    try:
        from poke_bot.recursive_turn_planner.r197_corpus import (
            verify_r197_complete_action_manifest,
        )
    except ImportError as exc:
        raise RuntimeError("r197 corpus verifier is unavailable") from exc
    receipt = verify_r197_complete_action_manifest(manifest_path)
    train_order, train_selection = _select_r197_episodes(
        _iter_r197_split_identities(expected_files["train"], split="train"),
        split="train",
        limit=int(max_train_games),
        selection_seed=int(selection_seed),
    )
    heldout_order, heldout_selection = _select_r197_episodes(
        _iter_r197_split_identities(expected_files["heldout"], split="heldout"),
        split="heldout",
        limit=int(max_heldout_games),
        selection_seed=int(selection_seed),
    )

    def selected_counts(path: Path, split: str, selected: set[str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in _iter_r197_split_identities(path, split=split):
            episode_id = str(row["episode_id"])
            if episode_id in selected:
                counts[episode_id] = counts.get(episode_id, 0) + 1
        return counts

    train_retained, train_cap = _plan_r197_batch_cap(
        selected_counts(expected_files["train"], "train", set(train_order)),
        episode_order=train_order,
        batch_cap=int(max_train_batches),
        split="train",
    )
    heldout_retained, heldout_cap = _plan_r197_batch_cap(
        selected_counts(
            expected_files["heldout"], "heldout", set(heldout_order)
        ),
        episode_order=heldout_order,
        batch_cap=int(max_heldout_batches),
        split="heldout",
    )
    plan = {
        "schema": "poke_bot.recursive_turn_planner.r197_training_selection_plan/v1",
        "corpus": {
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_digest,
            "receipt_path": str(expected_files["receipt"]),
            "receipt_sha256": file_digests["receipt"],
            "receipt_schema": str(receipt.get("schema") or ""),
            "source_pointer_sha256": pointer_digest,
            "derived_corpus_fingerprint": str(
                receipt.get("derived_corpus_fingerprint") or ""
            ),
        },
        "split": {
            "seed": R197_SPLIT_SEED,
            "heldout_fraction": 0.20,
            "unit": "episode_id",
            "source_disjoint": True,
        },
        "selection_seed": int(selection_seed),
        "action_space_max_combos": R197_COMPLETE_ACTION_CAP,
        "training_code_sha256": _r197_training_code_digests(),
        "train": {
            "candidate_selection": train_selection,
            "batch_cap_selection": train_cap,
            "retained_episode_ids": train_retained,
        },
        "heldout": {
            "candidate_selection": heldout_selection,
            "batch_cap_selection": heldout_cap,
            "retained_episode_ids": heldout_retained,
        },
        "row_level_sampling": False,
        "cross_window_dynamics_target": False,
    }
    plan["selection_plan_sha256"] = _canonical_json_sha256(plan)
    return plan


def _coerce_complete_action(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an action integer sequence")
    action: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"{field} contains a non-integer action token")
        action.append(int(item))
    return action


def _coerce_complete_legal_actions(value: Any) -> list[list[int]]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("legal_actions must be a non-empty ordered action list")
    return [
        _coerce_complete_action(action, field="legal_actions") for action in value
    ]


def _coerce_row_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _build_complete_action_sequences(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    max_context: int,
) -> tuple[list[Any], dict[tuple[str, int, int], dict[str, Any]], dict[str, Any]]:
    """Convert verified corpus rows to encoder sequences without factorization.

    ``featurize_step`` is used only for causal board/history features.  Its
    factorized policy stages are never used by the RTP encoder: every emitted
    batch later revalidates the row's full ordered legal action support.
    """
    from poke_bot.dataset import GameSequence, featurize_step

    if int(max_context) < 1:
        raise ValueError("parent model max_context must be positive")
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    records: dict[tuple[str, int, int], dict[str, Any]] = {}
    source_days: set[str] = set()
    for row_number, raw in enumerate(rows, start=1):
        if not isinstance(raw, Mapping):
            raise ValueError(f"{split} corpus row {row_number} is not an object")
        row = dict(raw)
        if (
            row.get("schema") != R197_COMPLETE_ACTION_ROW_SCHEMA
            or row.get("corpus_schema") != R197_COMPLETE_ACTION_CORPUS_SCHEMA
            or row.get("split") != split
        ):
            raise ValueError(
                f"{split} corpus row {row_number} has an unrecognized schema"
            )
        supplied_row_fingerprint = str(row.get("row_fingerprint") or "")
        row_without_fingerprint = dict(row)
        row_without_fingerprint.pop("row_fingerprint", None)
        if (
            _normalize_sha256(
                supplied_row_fingerprint, field="row_fingerprint"
            )
            != _canonical_json_sha256(row_without_fingerprint)
        ):
            raise ValueError(f"{split} corpus row {row_number} fingerprint mismatch")
        episode_id = str(row.get("episode_id") or "").strip()
        if not episode_id:
            raise ValueError(f"{split} corpus row {row_number} has no episode_id")
        seat = _coerce_row_int(row.get("seat"), field="seat")
        env_step = _coerce_row_int(row.get("env_step"), field="env_step")
        source_day = str(row.get("source_day") or "").strip()
        if not source_day:
            raise ValueError(f"{split} corpus row {row_number} has no source_day")
        source_days.add(source_day)
        observation = row.get("observation")
        if not isinstance(observation, Mapping):
            raise ValueError(f"{split} corpus row {row_number} has no observation")
        observation = dict(observation)
        observation_fingerprint = _normalize_sha256(
            str(row.get("observation_fingerprint") or ""),
            field="observation_fingerprint",
        )
        if observation_fingerprint != _canonical_json_sha256(observation):
            raise ValueError(
                f"{split} corpus row {row_number} observation_fingerprint mismatch"
            )
        action = _coerce_complete_action(row.get("action"), field="action")
        legal_actions = _coerce_complete_legal_actions(row.get("legal_actions"))
        if len(legal_actions) > R197_COMPLETE_ACTION_CAP:
            raise ValueError(
                f"{split} corpus row {row_number} exceeds complete-action cap"
            )
        selected_index = _coerce_row_int(
            row.get("selected_action_index"), field="selected_action_index"
        )
        if not 0 <= selected_index < len(legal_actions):
            raise ValueError(
                f"{split} corpus row {row_number} selected_action_index is out of range"
            )
        if legal_actions[selected_index] != action:
            raise ValueError(
                f"{split} corpus row {row_number} selected action/index mismatch"
            )
        if str(row.get("action_space_source") or "") != (
            "runtime_complete_observation"
        ):
            raise ValueError(
                f"{split} corpus row {row_number} is not runtime complete-observation"
            )
        _normalize_sha256(
            str(row.get("action_space_fingerprint") or ""),
            field="action_space_fingerprint",
        )
        action_space = row.get("action_space")
        if (
            not isinstance(action_space, Mapping)
            or action_space.get("schema")
            != "poke_bot.rtp_complete_ordered_legal_actions/v1"
            or int(action_space.get("max_action_combos", -1))
            != R197_COMPLETE_ACTION_CAP
            or action_space.get("complete_ordered_actions") != legal_actions
        ):
            raise ValueError(
                f"{split} corpus row {row_number} action_space is not canonical"
            )
        if row.get("outcome_available") is not True or row.get("terminal_complete") is not True:
            raise ValueError(
                f"{split} corpus row {row_number} lacks a terminal selected outcome"
            )
        raw_value = row.get("game_value")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{split} corpus row {row_number} game_value is invalid")
        game_value = float(raw_value)
        if game_value not in {-1.0, 0.0, 1.0}:
            raise ValueError(
                f"{split} corpus row {row_number} game_value is not terminal [-1,0,1]"
            )
        if row.get("factorized_prefix_substitution") is not False:
            raise ValueError(
                f"{split} corpus row {row_number} permits factorized substitution"
            )
        # The canonical behavior corpus deliberately omits this field.  A
        # null-looking placeholder is not equivalent evidence: accepting one
        # could blur the provenance boundary for future evaluator supervision.
        if "evaluator_targets" in row:
            raise ValueError(
                f"{split} corpus row {row_number} fabricates evaluator targets"
            )
        if row.get("unobserved_action_targets_present") is not False:
            raise ValueError(
                f"{split} corpus row {row_number} contains unobserved action targets"
            )
        target_provenance = row.get("target_provenance")
        if not isinstance(target_provenance, Mapping) or (
            target_provenance.get("unobserved_action_targets") != "absent_masked"
            or target_provenance.get("value_of_planning_target") != "absent_masked"
        ):
            raise ValueError(
                f"{split} corpus row {row_number} has untrusted target provenance"
            )
        deck_raw = row.get("deck")
        if not isinstance(deck_raw, (list, tuple)):
            raise ValueError(f"{split} corpus row {row_number} deck is malformed")
        deck = _coerce_complete_action(deck_raw, field="deck")
        if len(deck) != 60:
            raise ValueError(
                f"{split} corpus row {row_number} does not have a 60-card deck"
            )
        key = (episode_id, seat, env_step)
        if key in records:
            raise ValueError(
                f"duplicate complete-action row for episode={episode_id!r}, "
                f"seat={seat}, env_step={env_step}"
            )
        row["observation"] = observation
        row["action"] = action
        row["legal_actions"] = legal_actions
        row["selected_action_index"] = selected_index
        row["game_value"] = game_value
        row["deck"] = deck
        records[key] = row
        grouped.setdefault((episode_id, seat), []).append(row)

    sequences: list[Any] = []
    max_game_decisions = 0
    window_count = 0
    for (episode_id, seat), game_rows in sorted(grouped.items()):
        game_rows.sort(key=lambda row: int(row["env_step"]))
        max_game_decisions = max(max_game_decisions, len(game_rows))
        first = game_rows[0]
        deck = list(first["deck"])
        game_value = float(first["game_value"])
        archetype = str(first.get("archetype") or "")
        for row in game_rows[1:]:
            if list(row["deck"]) != deck:
                raise ValueError(
                    f"complete-action corpus deck changes within episode {episode_id!r}"
                )
            if float(row["game_value"]) != game_value:
                raise ValueError(
                    f"complete-action corpus game_value changes within episode {episode_id!r}"
                )
            if str(row.get("archetype") or "") != archetype:
                raise ValueError(
                    f"complete-action corpus archetype changes within episode {episode_id!r}"
                )
        # The frozen encoder has a bounded causal context.  Split a long
        # source game into deterministic, non-overlapping windows rather than
        # aborting the entire corpus or allowing a dynamics target to cross an
        # invisible window boundary.  The original episode id deliberately
        # remains on every window so the corpus's whole-game split identity is
        # preserved; ``sequence_window_id`` distinguishes batch provenance.
        for window_index, start in enumerate(
            range(0, len(game_rows), int(max_context))
        ):
            window_rows = game_rows[start : start + int(max_context)]
            window_id = (
                f"{episode_id}:seat={seat}:window={window_index:04d}:"
                f"steps={window_rows[0]['env_step']}-{window_rows[-1]['env_step']}"
            )
            for row in window_rows:
                row["sequence_window_id"] = window_id
            decisions = []
            for row in window_rows:
                try:
                    decisions.append(
                        featurize_step(
                            {
                                "observation": row["observation"],
                                "action": row["action"],
                                "env_step": row["env_step"],
                            },
                            deck,
                            verify_info_set=False,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    raise ValueError(
                        f"complete-action corpus row cannot be featurized: "
                        f"episode={episode_id!r}, seat={seat}, "
                        f"env_step={row['env_step']}"
                    ) from exc
            sequences.append(
                GameSequence(
                    episode_id=episode_id,
                    seat=seat,
                    archetype=archetype,
                    opp_archetype="",
                    deck=deck,
                    value=game_value,
                    decisions=decisions,
                    source="r197_complete_action_corpus",
                    target_provenance={
                        "complete_action_corpus": True,
                        "split": split,
                        "terminal_complete": True,
                        "outcome_available": True,
                        "original_episode_id": episode_id,
                        "sequence_window_id": window_id,
                        "window_index": window_index,
                        "window_length": len(window_rows),
                        "cross_window_dynamics_target": False,
                    },
                )
            )
            window_count += 1
    return sequences, records, {
        "n_rows": len(records),
        "n_sequences": len(sequences),
        "n_games": len(grouped),
        "n_encoder_windows": window_count,
        "max_game_decisions": max_game_decisions,
        "max_encoder_window_decisions": int(max_context),
        "cross_window_dynamics_target": False,
        "episode_ids": {str(sequence.episode_id) for sequence in sequences},
        "source_days": sorted(source_days),
    }


def load_r197_complete_action_corpus(
    job: ArchetypeRTPJob,
    model: Any,
) -> tuple[list[RTPDecisionBatch], list[RTPDecisionBatch], dict[str, Any]]:
    """Load the materialized r197 corpus and return its immutable split.

    This is intentionally distinct from compact-shard loading.  It uses the
    corpus module's verifier, rechecks all derivative bytes, validates each
    row's canonical complete action list/index/fingerprint against current
    runtime feature enumeration, and never reads overflow or factorized rows.
    """
    if str(job.profile).strip().lower() != "pure_rl_r197":
        raise ValueError("complete-action corpus loading is reserved for pure_rl_r197")
    raw_path = Path(job.complete_action_corpus).expanduser().resolve()
    manifest_path = raw_path if raw_path.name == "MANIFEST.json" else raw_path / "MANIFEST.json"
    corpus_dir = manifest_path.parent
    if not manifest_path.is_file():
        raise FileNotFoundError(f"complete-action corpus manifest missing: {manifest_path}")
    manifest_digest = _sha256_file(manifest_path)
    _verify_expected_digest(
        expected=job.complete_action_corpus_manifest_digest,
        actual=manifest_digest,
        field="complete_action_corpus_manifest_digest",
    )
    manifest = _read_json_object(manifest_path, label="complete-action corpus MANIFEST")
    if (manifest.get("eligibility") or {}).get("kaggle_replay_eligible") is not False:
        raise ValueError("complete-action corpus must declare kaggle_replay_eligible=false")
    split_seed = _declared_split_value(manifest, "seed")
    if split_seed is None:
        split_seed = _declared_split_value(manifest, "split_seed")
    if split_seed is None or int(split_seed) != R197_SPLIT_SEED:
        raise ValueError("complete-action corpus split seed is not the required 5_000_000")
    split_payload = manifest.get("split")
    if (
        not isinstance(split_payload, Mapping)
        or split_payload.get("unit") != "episode_id"
        or split_payload.get("source_disjoint") is not True
    ):
        raise ValueError("complete-action corpus split is not whole-episode source-disjoint")
    try:
        declared_fraction = float(split_payload.get("heldout_fraction"))
    except (TypeError, ValueError) as exc:
        raise ValueError("complete-action corpus heldout fraction is malformed") from exc
    if abs(declared_fraction - float(job.heldout_fraction)) > 1e-12:
        raise ValueError("complete-action corpus heldout fraction differs from job")
    declared_cap = manifest.get("max_action_combos")
    if declared_cap is None:
        action_space = manifest.get("action_space")
        if isinstance(action_space, Mapping):
            declared_cap = action_space.get("max_action_combos")
    if declared_cap is None or int(declared_cap) != R197_COMPLETE_ACTION_CAP:
        raise ValueError("complete-action corpus cap is not the required 1024")
    pointer_digest = _manifest_pointer_digest(manifest)
    if job.complete_action_corpus_source_pointer_digest:
        if not pointer_digest:
            raise ValueError("complete-action corpus manifest omits source pointer digest")
        _verify_expected_digest(
            expected=job.complete_action_corpus_source_pointer_digest,
            actual=pointer_digest,
            field="complete_action_corpus_source_pointer_digest",
        )

    expected_files = {
        "train": corpus_dir / "train.complete-actions.jsonl",
        "heldout": corpus_dir / "heldout.complete-actions.jsonl",
        "overflow": corpus_dir / "action-space-too-large.jsonl",
        "verified_identities": corpus_dir / "verified-episode-seats.jsonl",
        "episode_splits": corpus_dir / "episode-splits.jsonl",
        "receipt": corpus_dir / "RECEIPT.json",
    }
    missing = [name for name, path in expected_files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "complete-action corpus is incomplete; missing " + ", ".join(missing)
        )
    file_digests = {name: _sha256_file(path) for name, path in expected_files.items()}
    _verify_expected_digest(
        expected=job.complete_action_corpus_receipt_digest,
        actual=file_digests["receipt"],
        field="complete_action_corpus_receipt_digest",
    )
    try:
        from poke_bot.recursive_turn_planner.r197_corpus import (
            complete_action_space_fingerprint,
            iter_complete_action_rows,
        )
    except ImportError as exc:
        raise RuntimeError(
            "r197 complete-action corpus reader is unavailable; materialize the "
            "canonical corpus module before training"
        ) from exc

    def runtime_fingerprint(
        observation: Mapping[str, Any], legal_actions: Sequence[Sequence[int]]
    ) -> str:
        return str(
            complete_action_space_fingerprint(
                dict(observation),
                [list(action) for action in legal_actions],
                max_action_combos=R197_COMPLETE_ACTION_CAP,
            )
        )

    try:
        selection_plan = plan_r197_complete_action_selection(
            job.complete_action_corpus,
            expected_manifest_digest=job.complete_action_corpus_manifest_digest,
            expected_receipt_digest=job.complete_action_corpus_receipt_digest,
            expected_source_pointer_digest=(
                job.complete_action_corpus_source_pointer_digest
            ),
            selection_seed=int(job.seed),
            max_train_games=int(job.max_train_games),
            max_heldout_games=int(job.max_heldout_games),
            max_train_batches=int(job.max_train_batches),
            max_heldout_batches=int(job.max_heldout_batches),
            heldout_fraction=float(job.heldout_fraction),
        )
    except Exception as exc:  # noqa: BLE001
        raise ValueError("complete-action corpus selection preflight failed") from exc
    _verify_expected_digest(
        expected=job.complete_action_corpus_selection_plan_digest,
        actual=str(selection_plan["selection_plan_sha256"]),
        field="complete_action_corpus_selection_plan_digest",
    )
    _verify_expected_digest(
        expected=job.complete_action_corpus_train_selection_digest,
        actual=str(
            selection_plan["train"]["batch_cap_selection"][
                "retained_episode_ids_sha256"
            ]
        ),
        field="complete_action_corpus_train_selection_digest",
    )
    _verify_expected_digest(
        expected=job.complete_action_corpus_heldout_selection_digest,
        actual=str(
            selection_plan["heldout"]["batch_cap_selection"][
                "retained_episode_ids_sha256"
            ]
        ),
        field="complete_action_corpus_heldout_selection_digest",
    )
    train_episode_order = list(selection_plan["train"]["retained_episode_ids"])
    heldout_episode_order = list(
        selection_plan["heldout"]["retained_episode_ids"]
    )
    train_selection = dict(selection_plan["train"]["candidate_selection"])
    heldout_selection = dict(selection_plan["heldout"]["candidate_selection"])
    train_batch_cap = dict(selection_plan["train"]["batch_cap_selection"])
    heldout_batch_cap = dict(selection_plan["heldout"]["batch_cap_selection"])
    train_episode_set = set(train_episode_order)
    heldout_episode_set = set(heldout_episode_order)
    train_rows = iter_complete_action_rows(
        manifest_path,
        "train",
        verify=True,
        episode_ids=train_episode_set,
    )
    heldout_rows = iter_complete_action_rows(
        manifest_path,
        "heldout",
        verify=True,
        episode_ids=heldout_episode_set,
    )
    train_sequences, train_records, train_info = _build_complete_action_sequences(
        train_rows,
        split="train",
        max_context=int(model.max_context),
    )
    heldout_sequences, heldout_records, heldout_info = _build_complete_action_sequences(
        heldout_rows,
        split="heldout",
        max_context=int(model.max_context),
    )
    train_candidate_info = dict(train_info)
    heldout_candidate_info = dict(heldout_info)
    train_info = {
        **train_candidate_info,
        "candidate_n_rows": int(train_candidate_info["n_rows"]),
        "candidate_n_games": int(train_candidate_info["n_games"]),
        "candidate_n_encoder_windows": int(
            train_candidate_info["n_encoder_windows"]
        ),
        "n_rows": len(train_records),
        "n_sequences": len(train_sequences),
        "n_games": len({str(sequence.episode_id) for sequence in train_sequences}),
        "n_encoder_windows": len(train_sequences),
        "max_game_decisions": max(
            (
                sum(1 for key in train_records if key[:2] == episode_seat)
                for episode_seat in {
                    (key[0], key[1]) for key in train_records
                }
            ),
            default=0,
        ),
        "source_days": sorted(
            {str(row.get("source_day") or "") for row in train_records.values()}
        ),
        "episode_ids": {str(sequence.episode_id) for sequence in train_sequences},
        "candidate_selection": train_selection,
        "batch_cap_selection": train_batch_cap,
    }
    heldout_info = {
        **heldout_candidate_info,
        "candidate_n_rows": int(heldout_candidate_info["n_rows"]),
        "candidate_n_games": int(heldout_candidate_info["n_games"]),
        "candidate_n_encoder_windows": int(
            heldout_candidate_info["n_encoder_windows"]
        ),
        "n_rows": len(heldout_records),
        "n_sequences": len(heldout_sequences),
        "n_games": len(
            {str(sequence.episode_id) for sequence in heldout_sequences}
        ),
        "n_encoder_windows": len(heldout_sequences),
        "max_game_decisions": max(
            (
                sum(1 for key in heldout_records if key[:2] == episode_seat)
                for episode_seat in {
                    (key[0], key[1]) for key in heldout_records
                }
            ),
            default=0,
        ),
        "source_days": sorted(
            {
                str(row.get("source_day") or "")
                for row in heldout_records.values()
            }
        ),
        "episode_ids": {
            str(sequence.episode_id) for sequence in heldout_sequences
        },
        "candidate_selection": heldout_selection,
        "batch_cap_selection": heldout_batch_cap,
    }
    overlap = set(train_info["episode_ids"]).intersection(heldout_info["episode_ids"])
    if overlap:
        raise ValueError(
            "complete-action corpus leaks whole episodes across train/heldout split"
        )
    if not train_sequences or not heldout_sequences:
        raise RuntimeError("complete-action corpus requires non-empty train and heldout games")
    common_kwargs = {
        "option_threshold": int(job.complexity_option_threshold),
        "entropy_threshold": float(job.complexity_entropy_threshold),
        "num_plan_candidates": int(job.num_plan_candidates),
        "runtime_action_fingerprint": runtime_fingerprint,
        "require_complete_ordered_actions": True,
        "max_runtime_action_combos": R197_COMPLETE_ACTION_CAP,
        "return_provenance": True,
    }
    train_encoded = encode_sequences_to_batches(
        model,
        train_sequences,
        runtime_records=train_records,
        **common_kwargs,
    )
    heldout_encoded = encode_sequences_to_batches(
        model,
        heldout_sequences,
        runtime_records=heldout_records,
        **common_kwargs,
    )
    train_batches, train_encoding = train_encoded
    heldout_batches, heldout_encoding = heldout_encoded
    if not train_batches or not heldout_batches:
        raise RuntimeError(
            "complete-action corpus produced no complete ordered train/heldout batches"
        )
    if len(train_batches) != int(train_batch_cap["retained_batch_count_pre_encoding"]):
        raise RuntimeError(
            "r197 train encoding changed the selected complete-action batch count"
        )
    if len(heldout_batches) != int(
        heldout_batch_cap["retained_batch_count_pre_encoding"]
    ):
        raise RuntimeError(
            "r197 heldout encoding changed the selected complete-action batch count"
        )
    if any(
        batch.action_space_source != "runtime_complete_observation"
        for batch in [*train_batches, *heldout_batches]
    ):
        raise RuntimeError("r197 complete-action reader emitted a non-runtime action source")
    evaluator_statuses = [
        str(batch.candidate_target_provenance.get("status") or "")
        for batch in [*train_batches, *heldout_batches]
    ]
    provenance = {
        "schema": "poke_bot.recursive_turn_planner.r197_complete_action_training_input/v1",
        "corpus_directory": str(corpus_dir),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_digest,
        "receipt_path": str(expected_files["receipt"]),
        "receipt_sha256": file_digests["receipt"],
        "receipt_schema": str(
            selection_plan["corpus"].get("receipt_schema") or ""
        ),
        "derived_corpus_fingerprint": str(
            selection_plan["corpus"].get("derived_corpus_fingerprint") or ""
        ),
        "selection_plan_sha256": str(selection_plan["selection_plan_sha256"]),
        "train_selection_sha256": str(
            selection_plan["train"]["batch_cap_selection"][
                "retained_episode_ids_sha256"
            ]
        ),
        "heldout_selection_sha256": str(
            selection_plan["heldout"]["batch_cap_selection"][
                "retained_episode_ids_sha256"
            ]
        ),
        "training_code_sha256": dict(selection_plan["training_code_sha256"]),
        "files": {
            name: {"path": str(path), "sha256": file_digests[name]}
            for name, path in expected_files.items()
            if name != "receipt"
        },
        "manifest_declared_digest_fields": _declared_digest_fields(manifest),
        "source_pointer_sha256": pointer_digest,
        "split": {
            "seed": R197_SPLIT_SEED,
            "unit": "episode_id",
            "heldout_fraction": float(job.heldout_fraction),
            "train_episode_ids_digest": _canonical_json_sha256(
                sorted(train_info["episode_ids"])
            ),
            "heldout_episode_ids_digest": _canonical_json_sha256(
                sorted(heldout_info["episode_ids"])
            ),
            "overlap_episode_count": 0,
        },
        "action_space": {
            "schema": "poke_bot.rtp_complete_ordered_legal_actions/v1",
            "max_action_combos": R197_COMPLETE_ACTION_CAP,
            "canonical_order_required": True,
            "factorized_prefix_substitution": False,
        },
        "train": {
            **{
                key: value
                for key, value in train_info.items()
                if key != "episode_ids"
            },
            "batch_encoding": train_encoding,
        },
        "heldout": {
            **{
                key: value
                for key, value in heldout_info.items()
                if key != "episode_ids"
            },
            "batch_encoding": heldout_encoding,
        },
        "evaluator_targets": {
            "trusted_bound_batch_count": sum(
                status == "trusted_action_space_bound" for status in evaluator_statuses
            ),
            "masked_or_absent_batch_count": sum(
                status != "trusted_action_space_bound" for status in evaluator_statuses
            ),
            "parent_latent_lookahead_targets": "not_wired_future_input",
        },
        "kaggle_replay_eligible": False,
    }
    return list(train_batches), list(heldout_batches), provenance


def load_archetype_registry(path: Path | str) -> list[ArchetypeRTPJob]:
    """Load a YAML/JSON registry of archetype RTP jobs."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("PyYAML required to load archetype registry") from exc
        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("registry root must be an object")
    rows = payload.get("archetypes") or payload.get("jobs") or []
    if not isinstance(rows, list) or not rows:
        raise ValueError("registry requires non-empty archetypes list")
    jobs = [ArchetypeRTPJob(**{k: v for k, v in row.items() if k in ArchetypeRTPJob.__dataclass_fields__}) for row in rows]
    return jobs


def select_jobs(
    jobs: Sequence[ArchetypeRTPJob],
    *,
    specialist_ids: Optional[Sequence[str]] = None,
    only_ready: bool = False,
    include_disabled: bool = False,
) -> list[ArchetypeRTPJob]:
    wanted = None
    if specialist_ids:
        wanted = {str(s).strip().lower().replace("_", "-") for s in specialist_ids}
    out: list[ArchetypeRTPJob] = []
    for job in jobs:
        if wanted is not None and job.specialist_id not in wanted:
            continue
        if not include_disabled and not job.enabled:
            continue
        if only_ready and not job.ready_for_host_train:
            continue
        out.append(job)
    return out


def load_batches_for_job(
    job: ArchetypeRTPJob,
    *,
    synthetic: bool = False,
    n_synthetic: int = 64,
    return_presplit_heldout: bool = False,
) -> Any:
    """Return batches/source/provenance, plus r197's supplied heldout split.

    The legacy return shape remains ``(batches, source, provenance)``.  The
    pipeline requests ``return_presplit_heldout=True`` so a materialized r197
    corpus can retain its already source-disjoint whole-game partition rather
    than being split a second time by decisions or rows.
    """
    def finish(
        batches: list[RTPDecisionBatch],
        source: str,
        provenance: dict[str, Any],
        heldout: Optional[list[RTPDecisionBatch]] = None,
    ) -> Any:
        if return_presplit_heldout:
            return batches, source, provenance, heldout
        return batches, source, provenance

    provenance: dict[str, Any] = {
        "specialist_id": job.specialist_id,
        "shadow_only": True,
    }
    if str(job.profile).strip().lower() == "pure_rl_r197" and synthetic:
        raise ValueError(
            "pure_rl_r197 never permits synthetic training; use the sealed "
            "complete-action corpus"
        )
    if synthetic or not job.ready_for_host_train:
        batches = make_synthetic_batches(
            n_decisions=int(n_synthetic),
            d_model=int(job.d_model),
            seed=int(job.seed),
            option_threshold=int(job.complexity_option_threshold),
            entropy_threshold=float(job.complexity_entropy_threshold),
        )
        provenance.update(
            {
                "mode": "synthetic",
                "parent_digest_verified": False,
                "training_shard_digest": "",
                "runtime_action_alignment": "synthetic_complete_ordered",
            }
        )
        return finish(batches, "synthetic", provenance)

    from poke_bot.train import load_model_from_checkpoint

    ckpt = Path(job.parent_checkpoint).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"parent checkpoint missing: {ckpt}")

    # Compute before loading so a supplied digest is a byte-level contract,
    # not merely metadata copied into a receipt.
    parent_digest = _sha256_file(ckpt)
    parent_digest_verified = _verify_expected_digest(
        expected=job.parent_digest,
        actual=parent_digest,
        field="parent_digest",
    )
    model = load_model_from_checkpoint(ckpt, device=job.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    # Do not silently reinterpret a declared parent as a different model
    # profile; a sidecar trained at the wrong width is not attachable.
    live_d = int(getattr(model, "d_model", job.d_model))
    if live_d != int(job.d_model):
        raise ValueError(
            f"declared d_model={job.d_model} does not match parent d_model={live_d}"
        )

    if str(job.profile).strip().lower() == "pure_rl_r197":
        train_batches, heldout_batches, corpus_provenance = (
            load_r197_complete_action_corpus(job, model)
        )
        provenance.update(
            {
                "mode": "host_complete_action_corpus",
                "parent_checkpoint": str(ckpt),
                "parent_digest": parent_digest,
                "parent_digest_supplied": True,
                "parent_digest_verified": parent_digest_verified,
                "training_shard": "",
                "training_shard_digest": "",
                "profile": "pure_rl_r197",
                "d_model": int(job.d_model),
                "complete_action_corpus": corpus_provenance,
                "runtime_action_alignment": "canonical_complete_ordered_actions",
                "factorized_policy_stage_substitution": False,
                "shadow_only": True,
            }
        )
        return finish(
            train_batches,
            "complete_action_corpus",
            provenance,
            heldout_batches,
        )

    from poke_bot.pure_rl.dataset_bridge import compact_game_to_sequence
    from poke_bot.pure_rl.shards import iter_shard_games

    shard = Path(job.training_shard).expanduser().resolve()
    if not shard.is_file():
        raise FileNotFoundError(f"training shard missing: {shard}")
    shard_digest = _sha256_file(shard)
    shard_digest_verified = _verify_expected_digest(
        expected=job.training_shard_digest,
        actual=shard_digest,
        field="training_shard_digest",
    )

    sequences = []
    runtime_records: dict[tuple[str, int, int], dict[str, Any]] = {}
    records_seen = 0
    records_kept = 0
    for index, game in enumerate(iter_shard_games(shard)):
        if index >= int(job.max_games):
            break
        records_seen += 1
        sequence = compact_game_to_sequence(
            game, verify_info_set=False, max_context=int(model.max_context)
        )
        if sequence is not None:
            # A blank episode identifier cannot safely be split as a game.
            # Create an in-memory stable ID without mutating the source shard.
            if not str(sequence.episode_id or "").strip():
                sequence.episode_id = f"shard-row-{index:09d}"
            sequences.append(sequence)
            records_kept += 1
            terminal_complete = not bool(
                dict(game.target_provenance or {}).get("terminal_policy_failure")
            )
            try:
                game_value = float(game.value)
            except (TypeError, ValueError):
                game_value = float("nan")
            outcome_available = bool(
                terminal_complete
                and math.isfinite(game_value)
                and -1.0 <= game_value <= 1.0
            )
            episode_id = str(sequence.episode_id)
            for raw_decision in game.decisions:
                key = (
                    episode_id,
                    int(sequence.seat),
                    int(raw_decision.env_step),
                )
                if key in runtime_records:
                    raise ValueError(
                        "duplicate runtime-action record for "
                        f"episode_id={episode_id!r}, seat={sequence.seat}, "
                        f"env_step={raw_decision.env_step}"
                    )
                runtime_records[key] = {
                    "observation": dict(raw_decision.observation or {}),
                    "action": [int(value) for value in raw_decision.action],
                    "game_value": game_value,
                    "outcome_available": outcome_available,
                    "terminal_complete": terminal_complete,
                }
    if not sequences:
        raise RuntimeError(f"no trainable sequences for {job.specialist_id}")

    encoded = encode_sequences_to_batches(
        model,
        sequences,
        option_threshold=int(job.complexity_option_threshold),
        entropy_threshold=float(job.complexity_entropy_threshold),
        num_plan_candidates=int(job.num_plan_candidates),
        runtime_records=runtime_records,
        require_complete_ordered_actions=bool(job.require_complete_ordered_actions),
        max_runtime_action_combos=int(job.max_runtime_action_combos),
        return_provenance=True,
    )
    batches, encoding_provenance = encoded
    if not batches:
        raise RuntimeError(
            "no RTP batches with complete ordered runtime action support; "
            "do not train a sidecar from factorized prefixes"
        )
    provenance.update(
        {
            "mode": "host",
            "parent_checkpoint": str(ckpt),
            "parent_digest": parent_digest,
            "parent_digest_supplied": bool(str(job.parent_digest or "").strip()),
            "parent_digest_verified": parent_digest_verified,
            "training_shard": str(shard),
            "training_shard_digest": shard_digest,
            "training_shard_digest_supplied": bool(
                str(job.training_shard_digest or "").strip()
            ),
            "training_shard_digest_verified": shard_digest_verified,
            "training_shard_max_games": int(job.max_games),
            "shard_records_seen": records_seen,
            "shard_records_kept": records_kept,
            "runtime_record_count": len(runtime_records),
            "n_sequences": len(sequences),
            "d_model": int(job.d_model),
            "profile": job.profile,
            "batch_encoding": encoding_provenance,
        }
    )
    return finish(batches, "shard", provenance)


def run_archetype_rtp_pipeline(
    job: ArchetypeRTPJob,
    *,
    out_root: Path | str,
    synthetic: bool = False,
    n_synthetic: int = 64,
) -> ArchetypeRTPResult:
    """Train RTP (+ optional PokeRLM) for one specialist_id."""
    out_dir = Path(out_root) / job.specialist_id
    out_dir.mkdir(parents=True, exist_ok=True)
    batches, source, provenance, pre_split_heldout = load_batches_for_job(
        job,
        synthetic=synthetic,
        n_synthetic=n_synthetic,
        return_presplit_heldout=True,
    )
    if pre_split_heldout is not None:
        train_batches = batches
        heldout_batches = pre_split_heldout
        corpus_split = dict(provenance.get("complete_action_corpus") or {}).get(
            "split"
        )
        if not isinstance(corpus_split, Mapping):
            raise RuntimeError("r197 complete-action corpus omitted split provenance")
        split_provenance = {
            "schema": "poke_bot.recursive_turn_planner.game_heldout_split/v1",
            "source": "materialized_r197_complete_action_corpus",
            "seed": int(corpus_split["seed"]),
            "group_unit": str(corpus_split["unit"]),
            "heldout_fraction_requested": float(corpus_split["heldout_fraction"]),
            "n_train_batches": len(train_batches),
            "n_heldout_batches": len(heldout_batches),
            "train_game_ids_digest": str(corpus_split["train_episode_ids_digest"]),
            "heldout_game_ids_digest": str(
                corpus_split["heldout_episode_ids_digest"]
            ),
            "heldout_available": bool(heldout_batches),
            "source_disjoint": True,
            "resplit_by_pipeline": False,
        }
    else:
        train_batches, heldout_batches, split_provenance = split_batches_by_game(
            batches,
            heldout_fraction=float(job.heldout_fraction),
            seed=int(job.seed),
        )
    if not train_batches:
        raise RuntimeError("RTP game-level split produced no training batches")
    if source in {"shard", "complete_action_corpus"} and not heldout_batches:
        raise RuntimeError(
            "RTP host training requires at least two distinct games for a "
            "game-level heldout evaluation"
        )
    provenance = {
        **provenance,
        "game_heldout_split": split_provenance,
        "rtp_config_contract": {
            "profile": str(job.profile),
            "d_model": int(job.d_model),
            "num_plan_candidates": int(job.num_plan_candidates),
            "max_recursion_depth": int(job.max_recursion_depth),
            "max_neural_passes": int(job.max_neural_passes),
            "max_neural_passes_global_ceiling": int(
                RTP_MAX_AUTHORIZED_NEURAL_PASSES
            ),
            "required_recursive_passes": required_recursive_passes(
                num_plan_candidates=int(job.num_plan_candidates),
                max_recursion_depth=int(job.max_recursion_depth),
            ),
            "max_runtime_action_combos": int(job.max_runtime_action_combos),
            "heldout_fraction": float(job.heldout_fraction),
            "split_seed": int(job.split_seed),
            "max_train_games": int(job.max_train_games),
            "max_heldout_games": int(job.max_heldout_games),
            "max_train_batches": int(job.max_train_batches),
            "max_heldout_batches": int(job.max_heldout_batches),
        },
    }

    rtp = train_rtp_shadow(
        train_batches,
        output_dir=out_dir / "rtp",
        config=RTPTrainConfig(
            d_model=int(job.d_model),
            profile=str(job.profile),
            epochs=int(job.epochs),
            lr=float(job.lr),
            seed=int(job.seed),
            device=str(job.device),
            complexity_option_threshold=int(job.complexity_option_threshold),
            complexity_entropy_threshold=float(job.complexity_entropy_threshold),
            num_plan_candidates=int(job.num_plan_candidates),
            max_recursion_depth=int(job.max_recursion_depth),
            max_neural_passes=int(job.max_neural_passes),
        ),
        heldout_batches=heldout_batches,
        provenance=provenance,
        parent_checkpoint_sha256=(
            str(provenance.get("parent_digest") or "") or None
        ),
    )

    poke_ckpt = ""
    poke_receipt = ""
    poke_metrics: dict[str, Any] = {}
    if job.also_poke_rlm:
        poke = train_poke_rlm_shadow(
            train_batches,
            output_dir=out_dir / "poke_rlm",
            config=PokeRLMTrainConfig(
                profile="pure_rl_96" if int(job.d_model) == 96 else "base_384",
                d_model=int(job.d_model),
                epochs=int(job.epochs),
                lr=float(job.lr),
                seed=int(job.seed),
                device=str(job.device),
            ),
        )
        poke_ckpt = poke.checkpoint_path
        poke_receipt = poke.receipt_path
        poke_metrics = poke.metrics

    env = {
        # These are offline evaluation hints, never production activation.
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
        "POKEBOT_RTP_SHADOW_CHECKPOINT": rtp.checkpoint_path,
        "POKEBOT_RTP_SPECIALIST_ID": job.specialist_id,
        "POKEBOT_RTP_SHADOW_ONLY": "1",
    }
    result = ArchetypeRTPResult(
        specialist_id=job.specialist_id,
        source=source,
        out_dir=str(out_dir.resolve()),
        n_batches=len(train_batches),
        rtp_checkpoint=rtp.checkpoint_path,
        rtp_receipt=rtp.receipt_path,
        poke_rlm_checkpoint=poke_ckpt,
        poke_rlm_receipt=poke_receipt,
        metrics={
            "rtp": rtp.metrics,
            "rtp_heldout": rtp.heldout_metrics,
            "poke_rlm": poke_metrics,
        },
        env=env,
        serving_eligible=False,
    )
    summary = {
        "schema": PIPELINE_SCHEMA,
        "generated_at_unix": time.time(),
        "job": job.to_json(),
        "provenance": provenance,
        "result": result.to_json(),
        "serving_eligible": False,
        "selector_authority": False,
        "action_authority_enabled": False,
        "shadow_only": True,
        "notes": [
            "Sidecar only; parent CABT tensors are not rewritten.",
            "Game-level heldout metrics are evidence, not serving authority.",
            "Run a paired greedy bakeoff before treating as ladder-eligible.",
            "Repeat this job for every new archetype with its own checkpoint+shard.",
        ],
    }
    (out_dir / "pipeline_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def run_registry(
    jobs: Sequence[ArchetypeRTPJob],
    *,
    out_root: Path | str,
    synthetic: bool = False,
    n_synthetic: int = 64,
) -> dict[str, Any]:
    """Run many archetype jobs; skip disabled. Failures are recorded, not hidden."""
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for job in jobs:
        try:
            result = run_archetype_rtp_pipeline(
                job,
                out_root=root,
                synthetic=synthetic or not job.ready_for_host_train,
                n_synthetic=n_synthetic,
            )
            results.append(result.to_json())
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "specialist_id": job.specialist_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    fleet = {
        "schema": REGISTRY_SCHEMA,
        "generated_at_unix": time.time(),
        "out_root": str(root.resolve()),
        "n_jobs": len(jobs),
        "n_ok": len(results),
        "n_errors": len(errors),
        "results": results,
        "errors": errors,
        "serving_eligible": False,
    }
    (root / "fleet_summary.json").write_text(
        json.dumps(fleet, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return fleet


def example_registry_jobs() -> list[ArchetypeRTPJob]:
    """Canonical starter set — fill checkpoint/shard paths on host."""
    return [
        ArchetypeRTPJob(
            specialist_id="alakazam",
            display_name="Alakazam",
            profile="pure_rl",
            d_model=96,
            also_poke_rlm=True,
            notes="Train first; mainline ~900 bakeoff target.",
        ),
        ArchetypeRTPJob(
            specialist_id="marnie-s-grimmsnarl-ex",
            display_name="Marnie's Grimmsnarl ex",
            profile="pure_rl",
            d_model=96,
            also_poke_rlm=True,
            notes="After Alakazam RTP clears greedy bakeoff.",
        ),
        ArchetypeRTPJob(
            specialist_id="crustle",
            display_name="Crustle",
            profile="pure_rl",
            d_model=96,
            also_poke_rlm=True,
            enabled=True,
            notes="Fill paths after Crustle freeze/register; do not interrupt training.",
        ),
    ]


__all__ = [
    "PIPELINE_SCHEMA",
    "REGISTRY_SCHEMA",
    "R197_COMPLETE_ACTION_CAP",
    "R197_COMPLETE_ACTION_CORPUS_SCHEMA",
    "R197_COMPLETE_ACTION_ROW_SCHEMA",
    "R197_MAX_NEURAL_PASSES",
    "R197_SPLIT_SEED",
    "ArchetypeRTPJob",
    "ArchetypeRTPResult",
    "example_registry_jobs",
    "load_archetype_registry",
    "load_batches_for_job",
    "load_r197_complete_action_corpus",
    "plan_r197_complete_action_selection",
    "run_archetype_rtp_pipeline",
    "run_registry",
    "select_jobs",
]
