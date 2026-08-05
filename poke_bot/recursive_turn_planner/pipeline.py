"""Archetype-generic Recursive Turn Planner training pipeline.

One recipe for every specialist (Alakazam, Marnie, Crustle, future decks):

1. Bind a frozen parent CABT checkpoint + training shard to a ``specialist_id``
2. Encode features with the frozen parent
3. Train an RTP sidecar (optional PokeRLM sidecar)
4. Emit a per-archetype receipt + env load hints

Does not rewrite GOAL.md, parent tensors, selectors, or managed training.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from poke_bot.poke_rlm.training.shadow_train import (
    PokeRLMTrainConfig,
    train_poke_rlm_shadow,
)
from poke_bot.recursive_turn_planner.training.shadow_train import (
    RTPDecisionBatch,
    RTPTrainConfig,
    encode_sequences_to_batches,
    make_synthetic_batches,
    train_rtp_shadow,
)


PIPELINE_SCHEMA = "poke_bot.recursive_turn_planner.archetype_pipeline/v1"
REGISTRY_SCHEMA = "poke_bot.recursive_turn_planner.archetype_registry/v1"


@dataclass
class ArchetypeRTPJob:
    """One specialist's RTP train job. Paths may be empty for synthetic smoke."""

    specialist_id: str
    display_name: str = ""
    parent_checkpoint: str = ""
    training_shard: str = ""
    parent_digest: str = ""
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
    enabled: bool = True
    notes: str = ""

    def __post_init__(self) -> None:
        sid = str(self.specialist_id or "").strip().lower().replace("_", "-")
        if not sid:
            raise ValueError("specialist_id is required")
        self.specialist_id = sid
        if not self.display_name:
            self.display_name = sid

    @property
    def ready_for_host_train(self) -> bool:
        return bool(self.parent_checkpoint) and bool(self.training_shard)

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
) -> tuple[list[RTPDecisionBatch], str, dict[str, Any]]:
    """Return (batches, source, provenance)."""
    provenance: dict[str, Any] = {"specialist_id": job.specialist_id}
    if synthetic or not job.ready_for_host_train:
        batches = make_synthetic_batches(
            n_decisions=int(n_synthetic),
            d_model=int(job.d_model),
            seed=int(job.seed),
            option_threshold=int(job.complexity_option_threshold),
            entropy_threshold=float(job.complexity_entropy_threshold),
        )
        provenance["mode"] = "synthetic"
        return batches, "synthetic", provenance

    from poke_bot.pure_rl.dataset_bridge import compact_game_to_sequence
    from poke_bot.pure_rl.shards import iter_shard_games
    from poke_bot.train import load_model_from_checkpoint

    ckpt = Path(job.parent_checkpoint).expanduser().resolve()
    shard = Path(job.training_shard).expanduser().resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(f"parent checkpoint missing: {ckpt}")
    if not shard.exists():
        raise FileNotFoundError(f"training shard missing: {shard}")

    model = load_model_from_checkpoint(ckpt, device=job.device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    # Auto-bind width from live parent when possible.
    live_d = int(getattr(model, "d_model", job.d_model))
    if live_d != int(job.d_model):
        job.d_model = live_d
        job.profile = "pure_rl" if live_d == 96 else (
            "global_transformer" if live_d == 256 else job.profile
        )

    sequences = []
    for index, game in enumerate(iter_shard_games(shard)):
        if index >= int(job.max_games):
            break
        sequence = compact_game_to_sequence(
            game, verify_info_set=False, max_context=int(model.max_context)
        )
        if sequence is not None:
            sequences.append(sequence)
    if not sequences:
        raise RuntimeError(f"no trainable sequences for {job.specialist_id}")

    batches = encode_sequences_to_batches(
        model,
        sequences,
        option_threshold=int(job.complexity_option_threshold),
        entropy_threshold=float(job.complexity_entropy_threshold),
    )
    digest = job.parent_digest or _sha256_file(ckpt)
    provenance.update(
        {
            "mode": "host",
            "parent_checkpoint": str(ckpt),
            "parent_digest": digest,
            "training_shard": str(shard),
            "n_sequences": len(sequences),
            "d_model": int(job.d_model),
            "profile": job.profile,
        }
    )
    return batches, "shard", provenance


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
    batches, source, provenance = load_batches_for_job(
        job, synthetic=synthetic, n_synthetic=n_synthetic
    )

    rtp = train_rtp_shadow(
        batches,
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
        ),
    )

    poke_ckpt = ""
    poke_receipt = ""
    poke_metrics: dict[str, Any] = {}
    if job.also_poke_rlm:
        poke = train_poke_rlm_shadow(
            batches,
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
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "1",
        "POKEBOT_RTP_CHECKPOINT": rtp.checkpoint_path,
        "POKEBOT_RTP_SPECIALIST_ID": job.specialist_id,
    }
    result = ArchetypeRTPResult(
        specialist_id=job.specialist_id,
        source=source,
        out_dir=str(out_dir.resolve()),
        n_batches=len(batches),
        rtp_checkpoint=rtp.checkpoint_path,
        rtp_receipt=rtp.receipt_path,
        poke_rlm_checkpoint=poke_ckpt,
        poke_rlm_receipt=poke_receipt,
        metrics={
            "rtp": rtp.metrics,
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
        "notes": [
            "Sidecar only; parent CABT tensors are not rewritten.",
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
    "ArchetypeRTPJob",
    "ArchetypeRTPResult",
    "example_registry_jobs",
    "load_archetype_registry",
    "load_batches_for_job",
    "run_archetype_rtp_pipeline",
    "run_registry",
    "select_jobs",
]
