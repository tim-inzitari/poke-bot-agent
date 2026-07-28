"""Durable periodic expert rehearsal for the long-lived pure-RL learner.

The champion remains the protected rollout/deployment policy.  A separate
learner may accumulate AWR candidates that have not yet passed promotion.  At
an explicit cadence the learner receives one short supervised pass over an
immutable, validated top-ladder feature manifest.

Every pass has an immutable checkpoint plus a small append-only receipt.  If a
process dies after the checkpoint write but before the receipt write, the
checkpoint metadata is sufficient to reconstruct the receipt without training
again.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


REHEARSAL_RECEIPT_SCHEMA_VERSION = 2
EXPANDED_REHEARSAL_RECEIPT_SCHEMA_VERSION = 3
REHEARSAL_LOSS_WEIGHT_KEYS = (
    "value",
    "archetype",
    "opponent_hand",
    "opponent_hidden_remainder",
    "lethal_threat",
    "prize_race",
    "alakazam_guide",
)


def canonical_rehearsal_loss_weights(
    values: dict[str, Any],
) -> dict[str, float]:
    """Return the complete loss contract used to identify one rehearsal."""
    raw = dict(values)
    if set(raw) != set(REHEARSAL_LOSS_WEIGHT_KEYS):
        raise ValueError(
            "expert rehearsal loss keys mismatch: "
            f"expected={list(REHEARSAL_LOSS_WEIGHT_KEYS)} "
            f"actual={sorted(raw)}"
        )
    canonical = {
        name: float(raw[name]) for name in REHEARSAL_LOSS_WEIGHT_KEYS
    }
    if any(
        not math.isfinite(weight) or weight < 0.0
        for weight in canonical.values()
    ):
        raise ValueError("expert rehearsal loss weights must be finite/nonnegative")
    return canonical


def canonical_checkpoint_rehearsal_loss_weights(
    values: dict[str, Any],
    expanded_head_contract: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    """Validate checkpoint/result loss metadata and return its base losses.

    Expanded-head checkpoints bind the per-head weights both in the dedicated
    tensor-training record and under the historical rehearsal loss mapping.
    Durable receipts intentionally keep the base loss mapping flat and bind
    expanded heads in ``expanded_head_training``.  Accept that checkpoint-only
    nested field while still failing closed on missing or changed head weights.
    """

    raw = dict(values)
    embedded_expanded = raw.pop("expanded_strategic", None)
    expected_expanded = canonical_expanded_rehearsal_contract(
        expanded_head_contract
    )
    if expected_expanded:
        if not isinstance(embedded_expanded, dict):
            raise ValueError(
                "expanded rehearsal checkpoint is missing strategic weights"
            )
        from poke_bot.strategic_losses import canonical_expanded_loss_weights

        actual_expanded = canonical_expanded_loss_weights(
            dict(embedded_expanded)
        )
        if actual_expanded != expected_expanded["loss_weights"]:
            raise ValueError(
                "expanded rehearsal checkpoint strategic weights mismatch"
            )
    elif embedded_expanded is not None:
        raise ValueError(
            "legacy rehearsal checkpoint unexpectedly contains strategic weights"
        )
    return canonical_rehearsal_loss_weights(raw)


def canonical_expanded_rehearsal_contract(
    values: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Canonical identity of an expanded-head rehearsal, or an empty mapping."""

    if not values:
        return {}
    from poke_bot.strategic_losses import canonical_expanded_loss_weights

    raw = dict(values)
    if raw.get("schema") != "poke_bot.expanded_head_schedule/v1":
        raise ValueError("expanded rehearsal schedule schema mismatch")
    target_schema = str(raw.get("target_schema") or "")
    target_digest = str(raw.get("target_schema_digest") or "")
    schedule_digest = str(raw.get("schedule_digest") or "")
    if target_schema != "poke_bot.expanded_strategic_targets/v2":
        raise ValueError("expanded rehearsal target schema mismatch")
    for name, value in (
        ("target schema", target_digest),
        ("schedule", schedule_digest),
    ):
        if not value.startswith("sha256:") or len(value) != 71:
            raise ValueError(f"expanded rehearsal {name} digest is invalid")
    weights = canonical_expanded_loss_weights(
        dict(raw.get("loss_weights") or {})
    )
    enabled = [name for name, weight in weights.items() if weight > 0.0]
    if not enabled:
        raise ValueError("expanded rehearsal contract enables no heads")
    if raw.get("runtime_enabled_heads", []) != []:
        raise ValueError("expanded rehearsal must remain shadow-only")
    return {
        "schema": "poke_bot.expanded_head_schedule/v1",
        "target_schema": target_schema,
        "target_schema_digest": target_digest,
        "schedule_digest": schedule_digest,
        "loss_weights": weights,
        "enabled_heads": enabled,
        "runtime_enabled_heads": [],
    }


def _validate_expanded_training_record(
    record: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Validate a checkpoint-bound expanded record against its requested pass."""

    if record.get("schema") != "poke_bot.expanded_head_training/v1":
        raise RuntimeError("expanded rehearsal checkpoint schema mismatch")
    if record.get("target_schema_version") != expected["target_schema"]:
        raise RuntimeError("expanded rehearsal checkpoint target schema mismatch")
    if record.get("target_schema_digest") != expected["target_schema_digest"]:
        raise RuntimeError("expanded rehearsal checkpoint target digest mismatch")
    if record.get("schedule_digest") != expected["schedule_digest"]:
        raise RuntimeError("expanded rehearsal checkpoint schedule digest mismatch")
    try:
        actual_weights = canonical_expanded_rehearsal_contract(
            {
                **expected,
                "loss_weights": dict(record.get("loss_weights") or {}),
            }
        )["loss_weights"]
    except ValueError as exc:
        raise RuntimeError(
            f"expanded rehearsal checkpoint weights invalid: {exc}"
        ) from exc
    if actual_weights != expected["loss_weights"]:
        raise RuntimeError("expanded rehearsal checkpoint weights mismatch")
    gradient = {
        str(name) for name in record.get("gradient_enabled_heads") or ()
    }
    if gradient != set(expected["enabled_heads"]):
        raise RuntimeError("expanded rehearsal gradient-head set mismatch")
    if record.get("runtime_enabled_heads") != []:
        raise RuntimeError("expanded rehearsal unexpectedly enabled runtime heads")
    heads = dict(record.get("heads") or {})
    for name in expected["enabled_heads"]:
        row = dict(heads.get(name) or {})
        train_rows = int(row.get("train_labeled_rows", 0))
        validation_rows = int(row.get("validation_labeled_rows", 0))
        train_loss = row.get("train_loss")
        validation_loss = row.get("validation_loss")
        if row.get("present") is not True or row.get("gradient_enabled") is not True:
            raise RuntimeError(
                f"expanded rehearsal did not expose trainable {name}"
            )
        # Causal labels are allowed to be absent and are then masked. A head
        # with labeled training rows must, however, have finite loss evidence
        # from this exact checkpoint-producing pass. Validation coverage is
        # independently optional; a small deterministic split may contain no
        # validation labels for an otherwise trained head.
        if train_rows > 0 and (
            train_loss is None or not math.isfinite(float(train_loss))
        ):
            raise RuntimeError(
                f"expanded rehearsal did not train labeled {name} rows"
            )
        if validation_rows > 0 and (
            validation_loss is None
            or not math.isfinite(float(validation_loss))
        ):
            raise RuntimeError(
                f"expanded rehearsal did not validate labeled {name} rows"
            )
    return record


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_json_exclusive(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite expert receipt: {path}") from exc


@dataclass(frozen=True)
class ExpertManifestIdentity:
    path: str
    digest: str
    dates: tuple[str, ...]
    decisions: int
    records: int
    expanded_strategic_targets: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["dates"] = list(self.dates)
        if not self.expanded_strategic_targets:
            # Preserve byte-for-byte identity of legacy V5 rehearsal receipts.
            row.pop("expanded_strategic_targets", None)
        return row


def resolve_expert_manifest(
    source: Path,
    *,
    min_decisions: int = 1,
    require_protected: bool = False,
    required_archetype: str = "",
    required_compact_mode: str = "",
    required_max_context: Optional[int] = None,
    required_target_coverage: tuple[str, ...] = (),
    required_expanded_target_schema: str = "",
    required_expanded_target_digest: str = "",
    required_expanded_heads: tuple[str, ...] = (),
) -> ExpertManifestIdentity:
    """Resolve either a direct feature manifest or an atomic rolling pointer."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    outer = json.loads(source.read_text(encoding="utf-8"))
    if outer.get("format") == "pokebot-bootstrap-feature-manifest":
        if require_protected:
            raise ValueError("expert corpus must use a protected pointer")
        manifest_path = source
        expected_digest = ""
    else:
        if require_protected and (
            outer.get("schema") != "poke_bot.pinned_expert_corpus/v1"
            or outer.get("protected") is not True
        ):
            raise ValueError("expert corpus pointer is not protected")
        raw_manifest = str(outer.get("manifest") or "")
        if not raw_manifest:
            raise ValueError(f"rolling ladder pointer has no manifest: {source}")
        candidate = Path(raw_manifest).expanduser()
        manifest_path = (
            candidate.resolve()
            if candidate.is_absolute()
            else (source.parent / candidate).resolve()
        )
        expected_digest = str(outer.get("manifest_sha256") or "")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    digest = _sha256(manifest_path)
    if expected_digest and digest != expected_digest:
        raise ValueError(
            "rolling ladder pointer digest mismatch: "
            f"expected={expected_digest} actual={digest} path={manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("format") != "pokebot-bootstrap-feature-manifest":
        raise ValueError(f"invalid expert feature manifest: {manifest_path}")
    dates = tuple(str(value) for value in payload.get("dates") or ())
    if not dates or dates != tuple(sorted(set(dates))):
        raise ValueError(f"expert manifest dates are missing/invalid: {manifest_path}")
    totals = dict(payload.get("totals") or {})
    decisions = int(totals.get("decisions_kept") or 0)
    records = int(totals.get("records_kept") or 0)
    if decisions < int(min_decisions):
        raise ValueError(
            f"expert manifest has {decisions} decisions < minimum {min_decisions}"
        )
    if records <= 0:
        raise ValueError("expert manifest contains no usable records")
    required_archetype = str(required_archetype).strip().casefold()
    if required_archetype:
        selection = dict(payload.get("selection") or {})
        if (
            str(selection.get("value") or "").strip().casefold()
            != required_archetype
            or selection.get("seat_semantics") != "acting_seat_only"
        ):
            raise ValueError(
                "expert manifest acting-seat archetype contract mismatch"
            )
    required_compact_mode = str(required_compact_mode).strip()
    if required_compact_mode and (
        str(payload.get("compact_mode") or "") != required_compact_mode
    ):
        raise ValueError("expert manifest compact-mode contract mismatch")
    if required_max_context is not None and int(
        payload.get("max_context", -1)
    ) != int(required_max_context):
        raise ValueError("expert manifest max-context contract mismatch")
    required_targets = tuple(str(name).strip() for name in required_target_coverage)
    if any(not name for name in required_targets) or len(required_targets) != len(
        set(required_targets)
    ):
        raise ValueError("required expert target names must be unique/nonempty")
    if required_targets:
        quality = dict(payload.get("quality_gates") or {})
        coverage = dict(totals.get("target_coverage") or {})
        if quality.get("passed") is not True or quality.get(
            "hidden_targets_are_aux_only"
        ) is not True:
            raise ValueError("expert manifest quality gates are not authoritative")
        incomplete = {
            name: int(coverage.get(name, 0))
            for name in required_targets
            if int(coverage.get(name, 0)) != decisions
        }
        if incomplete:
            raise ValueError(
                "expert manifest target coverage is incomplete: "
                f"decisions={decisions} coverage={incomplete}"
            )
    expanded_targets = dict(payload.get("expanded_strategic_targets") or {})
    required_expanded_target_schema = str(
        required_expanded_target_schema
    ).strip()
    required_expanded_target_digest = str(
        required_expanded_target_digest
    ).strip()
    required_expanded_heads = tuple(
        str(name).strip() for name in required_expanded_heads
    )
    if (
        required_expanded_target_schema
        or required_expanded_target_digest
        or required_expanded_heads
    ):
        if not expanded_targets:
            raise ValueError(
                "expert manifest lacks expanded strategic target metadata"
            )
        if (
            required_expanded_target_schema
            and expanded_targets.get("schema")
            != required_expanded_target_schema
        ):
            raise ValueError("expert expanded strategic target schema mismatch")
        if (
            required_expanded_target_digest
            and expanded_targets.get("digest")
            != required_expanded_target_digest
        ):
            raise ValueError("expert expanded strategic target digest mismatch")
        if int(expanded_targets.get("decisions", -1)) != decisions:
            raise ValueError(
                "expert expanded strategic target decision count mismatch"
            )
        coverage = dict(expanded_targets.get("head_coverage") or {})
        for name in required_expanded_heads:
            row = dict(coverage.get(name) or {})
            labeled = int(row.get("labeled_rows", -1))
            masked = int(row.get("masked_rows", -1))
            total = int(row.get("total_rows", -1))
            if total != decisions or labeled < 0 or masked < 0:
                raise ValueError(
                    f"expert expanded target coverage is invalid for {name}"
                )
            if labeled + masked != total:
                raise ValueError(
                    f"expert expanded target coverage does not partition {name}"
                )
            if labeled <= 0:
                raise ValueError(
                    f"expert expanded target has no labeled rows for {name}"
                )
    return ExpertManifestIdentity(
        path=str(manifest_path),
        digest=digest,
        dates=dates,
        decisions=decisions,
        records=records,
        expanded_strategic_targets=expanded_targets,
    )


def rehearsal_due(before_iteration: int, every: int) -> bool:
    """A cadence of five means before iterations 5, 10, 15, ..."""
    return int(every) > 0 and int(before_iteration) > 0 and int(before_iteration) % int(every) == 0


def rehearsal_paths(run_dir: Path, before_iteration: int) -> tuple[Path, Path]:
    stem = f"before_iter_{int(before_iteration):05d}"
    return (
        Path(run_dir) / "checkpoints" / f"expert_{stem}.pt",
        Path(run_dir) / "rehearsals" / f"{stem}.json",
    )


def _validate_receipt(
    receipt: dict[str, Any],
    *,
    before_iteration: int,
    parent_digest: str,
    epochs: int,
    learning_rate: float,
    manifest_identity: ExpertManifestIdentity,
    loss_weights: dict[str, Any],
    corpus_split_seed: int,
    expanded_head_contract: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    from poke_bot.promotion import CheckpointIdentity

    expected_expanded = canonical_expanded_rehearsal_contract(
        expanded_head_contract
    )
    expected_schema = (
        EXPANDED_REHEARSAL_RECEIPT_SCHEMA_VERSION
        if expected_expanded
        else REHEARSAL_RECEIPT_SCHEMA_VERSION
    )
    if int(receipt.get("schema", -1)) != expected_schema:
        raise RuntimeError("expert receipt schema mismatch")
    if int(receipt.get("before_iteration", -1)) != int(before_iteration):
        raise RuntimeError("expert receipt iteration mismatch")
    if str(receipt.get("parent_digest") or "") != str(parent_digest):
        raise RuntimeError("expert receipt parent digest mismatch")
    if int(receipt.get("epochs", -1)) != int(epochs):
        raise RuntimeError("expert receipt epoch contract mismatch")
    if float(receipt.get("learning_rate", -1.0)) != float(learning_rate):
        raise RuntimeError("expert receipt learning-rate contract mismatch")
    manifest = dict(receipt.get("manifest") or {})
    if manifest != manifest_identity.as_dict():
        raise RuntimeError("expert receipt manifest contract mismatch")
    manifest_path = Path(str(manifest.get("path") or "")).expanduser().resolve()
    if not manifest_path.is_file() or _sha256(manifest_path) != str(
        manifest.get("digest") or ""
    ):
        raise RuntimeError("expert receipt manifest bytes are missing or changed")
    output = CheckpointIdentity.from_path(str(receipt.get("checkpoint") or ""))
    if output.digest != str(receipt.get("checkpoint_digest") or ""):
        raise RuntimeError("expert receipt checkpoint digest mismatch")
    try:
        actual_loss_weights = canonical_rehearsal_loss_weights(
            dict(receipt.get("loss_weights") or {})
        )
        expected_loss_weights = canonical_rehearsal_loss_weights(loss_weights)
    except ValueError as exc:
        raise RuntimeError(f"expert receipt loss contract invalid: {exc}") from exc
    if actual_loss_weights != expected_loss_weights:
        raise RuntimeError("expert receipt loss-weight contract mismatch")
    if int(receipt.get("corpus_split_seed", -1)) != int(corpus_split_seed):
        raise RuntimeError("expert receipt corpus split-seed mismatch")
    if expected_expanded:
        _validate_expanded_training_record(
            dict(receipt.get("expanded_head_training") or {}),
            expected_expanded,
        )
    return {**receipt, "checkpoint_identity": output.as_dict(), "reused": True}


def recover_rehearsal(
    run_dir: Path,
    *,
    before_iteration: int,
    parent_digest: str,
    epochs: int,
    learning_rate: float,
    manifest_identity: ExpertManifestIdentity,
    loss_weights: dict[str, Any],
    corpus_split_seed: int,
    expanded_head_contract: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Reuse a receipt, or reconstruct it after checkpoint-before-receipt crash."""
    from poke_bot import checkpoint
    from poke_bot.promotion import CheckpointIdentity

    checkpoint_path, receipt_path = rehearsal_paths(run_dir, before_iteration)
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        return _validate_receipt(
            receipt,
            before_iteration=before_iteration,
            parent_digest=parent_digest,
            epochs=epochs,
            learning_rate=learning_rate,
            manifest_identity=manifest_identity,
            loss_weights=loss_weights,
            corpus_split_seed=corpus_split_seed,
            expanded_head_contract=expanded_head_contract,
        )
    if not checkpoint_path.is_file():
        return None

    payload = checkpoint.load_checkpoint(checkpoint_path, map_location="cpu")
    record = dict((payload.get("extra") or {}).get("expert_rehearsal") or {})
    if int(record.get("before_iteration", -1)) != int(before_iteration):
        raise RuntimeError("orphan expert checkpoint iteration mismatch")
    if str(record.get("parent_digest") or "") != str(parent_digest):
        raise RuntimeError("orphan expert checkpoint parent mismatch")
    if int(record.get("epochs", -1)) != int(epochs):
        raise RuntimeError("orphan expert checkpoint epoch mismatch")
    if float(record.get("learning_rate", -1.0)) != float(learning_rate):
        raise RuntimeError("orphan expert checkpoint learning-rate mismatch")
    if dict(record.get("manifest") or {}) != manifest_identity.as_dict():
        raise RuntimeError("orphan expert checkpoint manifest mismatch")
    expected_expanded = canonical_expanded_rehearsal_contract(
        expanded_head_contract
    )
    try:
        record_loss_weights = canonical_checkpoint_rehearsal_loss_weights(
            dict(record.get("loss_weights") or {}),
            expected_expanded,
        )
        expected_loss_weights = canonical_rehearsal_loss_weights(loss_weights)
    except ValueError as exc:
        raise RuntimeError(
            f"orphan expert checkpoint loss contract invalid: {exc}"
        ) from exc
    if record_loss_weights != expected_loss_weights:
        raise RuntimeError("orphan expert checkpoint loss-weight mismatch")
    if int(record.get("corpus_split_seed", -1)) != int(corpus_split_seed):
        raise RuntimeError("orphan expert checkpoint corpus split-seed mismatch")
    expanded_training = dict(
        (payload.get("extra") or {}).get("expanded_head_training") or {}
    )
    if expected_expanded:
        _validate_expanded_training_record(
            expanded_training,
            expected_expanded,
        )
    identity = CheckpointIdentity.from_path(checkpoint_path)
    receipt = {
        "schema": (
            EXPANDED_REHEARSAL_RECEIPT_SCHEMA_VERSION
            if expected_expanded
            else REHEARSAL_RECEIPT_SCHEMA_VERSION
        ),
        "before_iteration": int(before_iteration),
        "parent_digest": str(parent_digest),
        "checkpoint": identity.path,
        "checkpoint_digest": identity.digest,
        "manifest": manifest_identity.as_dict(),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "loss_weights": expected_loss_weights,
        "corpus_split_seed": int(corpus_split_seed),
        "batch_size": int(record.get("batch_size") or 0),
        "metrics": {
            "train": record.get("train_metrics"),
            "validation": record.get("validation_metrics"),
        },
        "recovered_after_checkpoint_write": True,
        **(
            {"expanded_head_training": expanded_training}
            if expected_expanded
            else {}
        ),
    }
    _write_json_exclusive(receipt_path, receipt)
    return _validate_receipt(
        receipt,
        before_iteration=before_iteration,
        parent_digest=parent_digest,
        epochs=epochs,
        learning_rate=learning_rate,
        manifest_identity=manifest_identity,
        loss_weights=loss_weights,
        corpus_split_seed=corpus_split_seed,
        expanded_head_contract=expanded_head_contract,
    )


def commit_rehearsal_receipt(
    run_dir: Path,
    *,
    before_iteration: int,
    parent_digest: str,
    manifest: ExpertManifestIdentity,
    epochs: int,
    learning_rate: float,
    loss_weights: dict[str, Any],
    corpus_split_seed: int,
    result: dict[str, Any],
    expanded_head_contract: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Commit the small durable receipt after the immutable checkpoint exists."""
    from poke_bot.promotion import CheckpointIdentity

    _checkpoint_path, receipt_path = rehearsal_paths(run_dir, before_iteration)
    identity = CheckpointIdentity.from_path(str(result.get("candidate_path") or ""))
    if identity.digest != str(result.get("candidate_digest") or ""):
        raise RuntimeError("rehearsal result digest does not match saved checkpoint")
    if str(result.get("parent_digest") or "") != str(parent_digest):
        raise RuntimeError("rehearsal result parent digest mismatch")
    expected_loss_weights = canonical_rehearsal_loss_weights(loss_weights)
    rehearsal = dict(result.get("rehearsal") or {})
    if dict(rehearsal.get("manifest") or {}) != manifest.as_dict():
        raise RuntimeError("rehearsal result manifest contract mismatch")
    expected_expanded = canonical_expanded_rehearsal_contract(
        expanded_head_contract
    )
    try:
        actual_loss_weights = canonical_checkpoint_rehearsal_loss_weights(
            dict(rehearsal.get("loss_weights") or {}),
            expected_expanded,
        )
    except ValueError as exc:
        raise RuntimeError(
            f"rehearsal result loss contract invalid: {exc}"
        ) from exc
    if actual_loss_weights != expected_loss_weights:
        raise RuntimeError("rehearsal result loss-weight contract mismatch")
    if int(rehearsal.get("corpus_split_seed", -1)) != int(corpus_split_seed):
        raise RuntimeError("rehearsal result corpus split-seed mismatch")
    expanded_training = dict(result.get("expanded_head_training") or {})
    if expected_expanded:
        _validate_expanded_training_record(
            expanded_training,
            expected_expanded,
        )
    receipt = {
        "schema": (
            EXPANDED_REHEARSAL_RECEIPT_SCHEMA_VERSION
            if expected_expanded
            else REHEARSAL_RECEIPT_SCHEMA_VERSION
        ),
        "before_iteration": int(before_iteration),
        "parent_digest": str(parent_digest),
        "checkpoint": identity.path,
        "checkpoint_digest": identity.digest,
        "manifest": manifest.as_dict(),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "loss_weights": expected_loss_weights,
        "corpus_split_seed": int(corpus_split_seed),
        "batch_size": int(result.get("batch_size") or 0),
        "metrics": {
            "train": result.get("train_metrics"),
            "validation": result.get("validation_metrics"),
        },
        "recovered_after_checkpoint_write": False,
        **(
            {"expanded_head_training": expanded_training}
            if expected_expanded
            else {}
        ),
    }
    _write_json_exclusive(receipt_path, receipt)
    return _validate_receipt(
        receipt,
        before_iteration=before_iteration,
        parent_digest=parent_digest,
        epochs=epochs,
        learning_rate=learning_rate,
        manifest_identity=manifest,
        loss_weights=expected_loss_weights,
        corpus_split_seed=corpus_split_seed,
        expanded_head_contract=expanded_head_contract,
    )


class ResidentExpertCorpusCache:
    """Keep one validated top-ladder window packed on the trainer GPU."""

    def __init__(self, *, cpu_pack_root: Optional[Path] = None) -> None:
        from poke_bot import config
        from poke_bot.pure_rl.expert_cpu_pack import ExpertCpuPackCache

        self.identity: Optional[ExpertManifestIdentity] = None
        self.corpus: Any = None
        self.max_context: Optional[int] = None
        self.seed: Optional[int] = None
        self.val_frac: Optional[float] = None
        self.device: Optional[str] = None
        self.belief_card_vocab: Optional[int] = None
        self.pack_info: Optional[dict[str, Any]] = None
        root = (
            Path(cpu_pack_root).expanduser().resolve()
            if cpu_pack_root is not None
            else Path(config.HARDWARE.cache_dir).expanduser().resolve()
            / "expert_cpu_pack"
        )
        self.cpu_pack_cache = ExpertCpuPackCache(root)

    def release(self) -> None:
        self.corpus = None
        self.identity = None
        self.max_context = None
        self.seed = None
        self.val_frac = None
        self.device = None
        self.belief_card_vocab = None
        self.pack_info = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def prepare(
        self,
        identity: ExpertManifestIdentity,
        *,
        device: Any,
        seed: int,
        val_frac: float = 0.10,
        max_context: Optional[int] = None,
        belief_card_vocab: Optional[int] = None,
        pack_workers: int = 1,
        pack_memory_reserve_gib: float = 12.0,
        pack_disk_reserve_gib: float = 16.0,
    ) -> Any:
        if (
            self.corpus is not None
            and self.identity is not None
            and self.identity.digest == identity.digest
            and self.max_context
            == (int(max_context) if max_context is not None else None)
            and self.seed == int(seed)
            and self.val_frac == float(val_frac)
            and self.device == str(device)
            and self.belief_card_vocab
            == (
                int(belief_card_vocab)
                if belief_card_vocab is not None
                else None
            )
        ):
            return self.corpus

        import torch

        from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
        from poke_bot.feature_shards import COMPACT_MODE_TEMPORAL_EXPERT
        from poke_bot.pure_rl.expert_cpu_pack import ExpertCpuPackKey
        from poke_bot.pure_rl.expert_feature_stream import (
            EpisodeGroupedFeatureManifest,
        )
        from poke_bot.process_memory import release_process_heap

        self.release()
        key = ExpertCpuPackKey(
            manifest_digest=identity.digest,
            split_seed=int(seed),
            val_frac=float(val_frac),
            max_context=(
                int(max_context) if max_context is not None else None
            ),
            belief_card_vocab=(
                int(belief_card_vocab)
                if belief_card_vocab is not None
                else None
            ),
        )

        def build_cpu_pack() -> DeviceResidentBootstrapCorpus:
            plan = EpisodeGroupedFeatureManifest.open(
                Path(identity.path),
                expected_manifest_digest=identity.digest,
                val_frac=float(val_frac),
                seed=int(seed),
                max_context=(
                    int(max_context) if max_context is not None else None
                ),
                expected_compact_mode=(
                    COMPACT_MODE_TEMPORAL_EXPERT
                    if max_context is not None or belief_card_vocab is not None
                    else None
                ),
                workers=max(1, int(pack_workers)),
            )
            try:
                if plan.decisions != int(identity.decisions):
                    raise RuntimeError(
                        "loaded expert decision count differs from immutable "
                        f"manifest: loaded={plan.decisions} "
                        f"manifest={identity.decisions}"
                    )
                if plan.max_context is not None:
                    print(
                        "[device-corpus] expert temporal layout "
                        f"context={plan.max_context} "
                        f"truncated_sequences={plan.truncated_sequences}",
                        flush=True,
                    )
                train, val = plan.splits()
                if int(pack_workers) > 1:
                    from poke_bot.pure_rl.expert_parallel_pack import (
                        build_parallel_expert_cpu_pack,
                    )

                    return build_parallel_expert_cpu_pack(
                        plan,
                        workers=int(pack_workers),
                        exact_card_vocab=(
                            int(belief_card_vocab)
                            if belief_card_vocab is not None
                            else None
                        ),
                        spool_root=self.cpu_pack_cache.root,
                        memory_reserve_gib=float(
                            pack_memory_reserve_gib
                        ),
                        disk_reserve_gib=float(pack_disk_reserve_gib),
                    )
                return DeviceResidentBootstrapCorpus.from_splits(
                    train,
                    val,
                    device=torch.device("cpu"),
                    exact_card_vocab=(
                        int(belief_card_vocab)
                        if belief_card_vocab is not None
                        else None
                    ),
                )
            finally:
                del plan
                release_process_heap()

        cpu_corpus, pack_info = self.cpu_pack_cache.load_or_build(
            key, build_cpu_pack
        )
        try:
            corpus = cpu_corpus.to_device(torch.device(device))
        finally:
            del cpu_corpus
            release_process_heap()
        self.identity = identity
        self.corpus = corpus
        self.max_context = int(max_context) if max_context is not None else None
        self.seed = int(seed)
        self.val_frac = float(val_frac)
        self.device = str(device)
        self.belief_card_vocab = (
            int(belief_card_vocab)
            if belief_card_vocab is not None
            else None
        )
        self.pack_info = pack_info
        return corpus


def carry_learner_candidate(
    promotion: dict[str, Any],
    *,
    abort: bool,
    minimum_head_to_head_wr: float = 0.35,
) -> tuple[bool, str]:
    """Keep small non-regressing steps while rolling back obvious collapse."""
    if abort:
        return False, "training_abort"
    if not bool(promotion.get("valid", True)):
        return False, "invalid_promotion_evidence"
    try:
        wr = float(promotion.get("wr"))
    except (TypeError, ValueError):
        return False, "missing_head_to_head_wr"
    if wr < float(minimum_head_to_head_wr):
        return False, "head_to_head_collapse"
    return True, "promoted" if bool(promotion.get("passed")) else "continuous_learner"


def continuous_learner_carry_decision(
    *,
    candidate_safety_ok: bool,
    candidate_safety_reason: str,
    heldout_audit_ok: bool,
    promoted: bool,
) -> tuple[bool, str]:
    """Keep exploration independent from the protected heldout champion.

    A valid candidate that clears the configured head-to-head safety floor may
    become the next learner even when it does not set a new official heldout
    record.  The formal heldout champion remains separately ranked and is never
    replaced by this decision.  An invalid heldout execution cannot become a
    rollout policy because its exact checkpoint/seat/opponent contract was not
    proven.
    """
    if not bool(candidate_safety_ok):
        return False, str(candidate_safety_reason or "candidate_safety_failed")
    if not bool(heldout_audit_ok):
        return False, "heldout_contract_audit_failed"
    return (
        True,
        "promoted_safety_carry"
        if bool(promoted)
        else "continuous_learner_safety_carry",
    )
