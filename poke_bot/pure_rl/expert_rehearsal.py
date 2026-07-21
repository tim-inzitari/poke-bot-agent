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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


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

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["dates"] = list(self.dates)
        return row


def resolve_expert_manifest(
    source: Path,
    *,
    min_decisions: int = 1,
) -> ExpertManifestIdentity:
    """Resolve either a direct feature manifest or an atomic rolling pointer."""
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    outer = json.loads(source.read_text(encoding="utf-8"))
    if outer.get("format") == "pokebot-bootstrap-feature-manifest":
        manifest_path = source
        expected_digest = ""
    else:
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
    return ExpertManifestIdentity(
        path=str(manifest_path),
        digest=digest,
        dates=dates,
        decisions=decisions,
        records=records,
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
) -> dict[str, Any]:
    from poke_bot.promotion import CheckpointIdentity

    if int(receipt.get("before_iteration", -1)) != int(before_iteration):
        raise RuntimeError("expert receipt iteration mismatch")
    if str(receipt.get("parent_digest") or "") != str(parent_digest):
        raise RuntimeError("expert receipt parent digest mismatch")
    if int(receipt.get("epochs", -1)) != int(epochs):
        raise RuntimeError("expert receipt epoch contract mismatch")
    if float(receipt.get("learning_rate", -1.0)) != float(learning_rate):
        raise RuntimeError("expert receipt learning-rate contract mismatch")
    manifest = dict(receipt.get("manifest") or {})
    manifest_path = Path(str(manifest.get("path") or "")).expanduser().resolve()
    if not manifest_path.is_file() or _sha256(manifest_path) != str(
        manifest.get("digest") or ""
    ):
        raise RuntimeError("expert receipt manifest bytes are missing or changed")
    output = CheckpointIdentity.from_path(str(receipt.get("checkpoint") or ""))
    if output.digest != str(receipt.get("checkpoint_digest") or ""):
        raise RuntimeError("expert receipt checkpoint digest mismatch")
    return {**receipt, "checkpoint_identity": output.as_dict(), "reused": True}


def recover_rehearsal(
    run_dir: Path,
    *,
    before_iteration: int,
    parent_digest: str,
    epochs: int,
    learning_rate: float,
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
    identity = CheckpointIdentity.from_path(checkpoint_path)
    receipt = {
        "schema": 1,
        "before_iteration": int(before_iteration),
        "parent_digest": str(parent_digest),
        "checkpoint": identity.path,
        "checkpoint_digest": identity.digest,
        "manifest": dict(record.get("manifest") or {}),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(record.get("batch_size") or 0),
        "metrics": {
            "train": record.get("train_metrics"),
            "validation": record.get("validation_metrics"),
        },
        "recovered_after_checkpoint_write": True,
    }
    _write_json_exclusive(receipt_path, receipt)
    return _validate_receipt(
        receipt,
        before_iteration=before_iteration,
        parent_digest=parent_digest,
        epochs=epochs,
        learning_rate=learning_rate,
    )


def commit_rehearsal_receipt(
    run_dir: Path,
    *,
    before_iteration: int,
    parent_digest: str,
    manifest: ExpertManifestIdentity,
    epochs: int,
    learning_rate: float,
    result: dict[str, Any],
) -> dict[str, Any]:
    """Commit the small durable receipt after the immutable checkpoint exists."""
    from poke_bot.promotion import CheckpointIdentity

    _checkpoint_path, receipt_path = rehearsal_paths(run_dir, before_iteration)
    identity = CheckpointIdentity.from_path(str(result.get("candidate_path") or ""))
    if identity.digest != str(result.get("candidate_digest") or ""):
        raise RuntimeError("rehearsal result digest does not match saved checkpoint")
    if str(result.get("parent_digest") or "") != str(parent_digest):
        raise RuntimeError("rehearsal result parent digest mismatch")
    receipt = {
        "schema": 1,
        "before_iteration": int(before_iteration),
        "parent_digest": str(parent_digest),
        "checkpoint": identity.path,
        "checkpoint_digest": identity.digest,
        "manifest": manifest.as_dict(),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "batch_size": int(result.get("batch_size") or 0),
        "metrics": {
            "train": result.get("train_metrics"),
            "validation": result.get("validation_metrics"),
        },
        "recovered_after_checkpoint_write": False,
    }
    _write_json_exclusive(receipt_path, receipt)
    return _validate_receipt(
        receipt,
        before_iteration=before_iteration,
        parent_digest=parent_digest,
        epochs=epochs,
        learning_rate=learning_rate,
    )


class ResidentExpertCorpusCache:
    """Keep one validated top-ladder window packed on the trainer GPU."""

    def __init__(self) -> None:
        self.identity: Optional[ExpertManifestIdentity] = None
        self.corpus: Any = None

    def release(self) -> None:
        self.corpus = None
        self.identity = None
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
    ) -> Any:
        if (
            self.corpus is not None
            and self.identity is not None
            and self.identity.digest == identity.digest
        ):
            return self.corpus

        from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
        from poke_bot.feature_shards import load_feature_manifest
        from poke_bot.process_memory import release_process_heap
        from poke_bot.train import split_dataset

        self.release()
        dataset = load_feature_manifest(Path(identity.path), verify_hashes=True)
        if dataset.n_decisions != int(identity.decisions):
            raise RuntimeError(
                "loaded expert decision count differs from immutable manifest: "
                f"loaded={dataset.n_decisions} manifest={identity.decisions}"
            )
        train, val = split_dataset(
            dataset,
            float(val_frac),
            int(seed),
            group_by_episode=True,
        )
        corpus = DeviceResidentBootstrapCorpus.from_splits(
            train,
            val,
            device=device,
        )
        dataset.sequences.clear()
        train.clear()
        val.clear()
        del dataset, train, val
        release_process_heap()
        self.identity = identity
        self.corpus = corpus
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

