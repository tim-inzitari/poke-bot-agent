"""Streaming shadow trainer for the sealed Alakazam recent-20 RTP overlay."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
import torch.nn as nn

from poke_bot.recursive_turn_planner.config import RTPConfig
from poke_bot.recursive_turn_planner.planner import RecursiveTurnPlanner
from poke_bot.recursive_turn_planner.recent20_overlay import (
    Recent20RTPDataset,
    canonical_bytes,
    sha256_file,
)
from poke_bot.recursive_turn_planner.training.checkpoint import (
    load_rtp_checkpoint,
    save_rtp_checkpoint,
)
from poke_bot.recursive_turn_planner.training.shadow_train import (
    RTPDecisionBatch,
    RTPTrainConfig,
    train_step,
)


ADAPTER_SCHEMA = "poke_bot.alakazam_recent20_rtp_semantic_adapter/v1"
COMPLETION_SCHEMA = "poke_bot.alakazam_recent20_rtp_shadow_bootstrap/v1"
FEATURE_WIDTH = 40


@dataclass(frozen=True)
class Recent20ShadowConfig:
    d_model: int = 96
    hidden_width: int = 128
    epochs: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_accumulation_stages: int = 32
    max_train_programs: int = 0
    max_validation_programs: int = 4096
    max_train_programs_per_day: int = 0
    max_validation_programs_per_day: int = 0
    seed: int = 31816
    device: str = "cpu"
    verify_overlay_shards: bool = True

    def __post_init__(self) -> None:
        for name in (
            "d_model",
            "hidden_width",
            "epochs",
            "gradient_accumulation_stages",
            "max_validation_programs",
        ):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be positive")
        if int(self.max_train_programs) < 0:
            raise ValueError("max_train_programs must be nonnegative")
        if int(self.max_train_programs_per_day) < 0:
            raise ValueError("max_train_programs_per_day must be nonnegative")
        if int(self.max_validation_programs_per_day) < 0:
            raise ValueError("max_validation_programs_per_day must be nonnegative")
        if not math.isfinite(float(self.learning_rate)) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")


class Recent20SemanticAdapter(nn.Module):
    """Permutation-safe public semantic projection into RTP latent space."""

    def __init__(self, *, d_model: int = 96, hidden_width: int = 128) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.hidden_width = int(hidden_width)
        self.option_projection = nn.Sequential(
            nn.LayerNorm(FEATURE_WIDTH),
            nn.Linear(FEATURE_WIDTH, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, self.d_model),
            nn.Tanh(),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(FEATURE_WIDTH * 2),
            nn.Linear(FEATURE_WIDTH * 2, self.hidden_width),
            nn.GELU(),
            nn.Linear(self.hidden_width, self.d_model),
            nn.Tanh(),
        )

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or int(features.shape[1]) != FEATURE_WIDTH:
            raise ValueError("recent20 semantic features must be [options, 40]")
        if int(features.shape[0]) < 1 or not bool(torch.isfinite(features).all()):
            raise ValueError("recent20 semantic options must be nonempty and finite")
        option_hidden = self.option_projection(features)
        summary = torch.cat(
            (features.mean(dim=0), features.max(dim=0).values), dim=0
        )
        state = self.state_projection(summary)
        return state, option_hidden

    def config_json(self) -> dict[str, Any]:
        return {
            "schema": ADAPTER_SCHEMA,
            "input_width": FEATURE_WIDTH,
            "d_model": self.d_model,
            "hidden_width": self.hidden_width,
            "state_pool": "permutation_invariant_mean_and_max",
            "public_information_only": True,
        }


def _planner_config(d_model: int) -> RTPConfig:
    return RTPConfig(
        sizing_profile="pure_rl",
        d_model=int(d_model),
        dynamics_width=max(32, 2 * int(d_model)),
        num_plan_candidates=4,
        max_recursion_depth=2,
        max_neural_passes=4,
        complexity_option_threshold=8,
        complexity_entropy_threshold=1.5,
    )


def _write_create_only(path: Path, body: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        os.write(fd, body)
        os.fsync(fd)
    finally:
        os.close(fd)
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _stage_batches(
    dataset: Recent20RTPDataset,
    split: str,
    *,
    adapter: Recent20SemanticAdapter,
    device: torch.device,
    max_programs: int,
    max_programs_per_day: int = 0,
) -> Iterator[tuple[RTPDecisionBatch, dict[str, Any]]]:
    programs = 0
    per_day: dict[str, int] = {}
    for sample in dataset.iter_samples(split):
        if sample.get("public_information_only") is not True:
            raise RuntimeError("recent20 shadow sample crossed information boundary")
        program = sample["program"]
        day = str(program["utc_day"])
        if max_programs_per_day and per_day.get(day, 0) >= max_programs_per_day:
            continue
        stages = list(program.get("stages") or ())
        feature_stages = list(sample.get("base_option_features_by_stage") or ())
        if not stages or len(stages) != len(feature_stages):
            raise RuntimeError("recent20 program/base stage alignment drifted")
        multi_stage = len(stages) > 1
        # Encode each yielded stage in its own graph. A program can contain
        # more stages than the gradient-accumulation window, so retaining one
        # shared graph for the whole program would cross an optimizer step.
        for index, (stage, rows) in enumerate(zip(stages, feature_stages)):
            features = torch.as_tensor(rows, dtype=torch.float32, device=device)
            state, option_hidden = adapter.encode(features)
            legal = [list(action) for action in stage["ordered_legal_action_programs"]]
            chosen = int(stage["selected_option_index"])
            valid = list(stage.get("valid_option_mask") or ())
            if (
                len(legal) != int(option_hidden.shape[0])
                or valid != [True] * len(legal)
                or not 0 <= chosen < len(legal)
                or legal[chosen] != list(stage["selected_action_program"])
            ):
                raise RuntimeError("recent20 legal action/selection alignment drifted")
            next_state = None
            if index + 1 < len(feature_stages):
                next_features = torch.as_tensor(
                    feature_stages[index + 1], dtype=torch.float32, device=device
                )
                next_state = adapter.encode(next_features)[0]
            outcome = program.get("recorded_outcome")
            outcome_available = isinstance(outcome, (int, float)) and not isinstance(
                outcome, bool
            ) and float(outcome) in {-1.0, 0.0, 1.0}
            yield (
                RTPDecisionBatch(
                    state=state,
                    option_hidden=option_hidden,
                    legal_actions=legal,
                    chosen_index=chosen,
                    should_recurse=multi_stage,
                    next_state=next_state,
                    game_value=float(outcome) if outcome_available else None,
                    outcome_available=outcome_available,
                    episode_id=str(program["episode_id"]),
                    sequence_window_id=str(program["program_identity"]),
                    action_space_source=(
                        "runtime_factorized_stage_grouped_under_complete_recorded_program"
                    ),
                ),
                {
                    "program_identity": program["program_identity"],
                    "factorized_stage": int(stage["factorized_stage"]),
                    "split": split,
                },
            )
        programs += 1
        per_day[day] = per_day.get(day, 0) + 1
        if max_programs and programs >= max_programs:
            return


def _mean(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: sum(float(row.get(key, 0.0)) for row in rows) / len(rows)
        for key in keys
    }


@torch.no_grad()
def _evaluate(
    dataset: Recent20RTPDataset,
    planner: RecursiveTurnPlanner,
    adapter: Recent20SemanticAdapter,
    *,
    cfg: Recent20ShadowConfig,
    train_cfg: RTPTrainConfig,
) -> dict[str, Any]:
    planner.eval()
    adapter.eval()
    rows: list[dict[str, float]] = []
    top1 = 0
    stages = 0
    programs: set[str] = set()
    for batch, identity in _stage_batches(
        dataset,
        "validation",
        adapter=adapter,
        device=torch.device(cfg.device),
        max_programs=int(cfg.max_validation_programs),
        max_programs_per_day=int(cfg.max_validation_programs_per_day),
    ):
        loss, metrics = train_step(planner, batch, cfg=train_cfg)
        rows.append({"loss": float(loss.item()), **metrics})
        with torch.enable_grad():
            # Rank is computed from the same differentiable serving-shaped
            # scorer; no optimizer step or target fabrication occurs.
            from poke_bot.recursive_turn_planner.training.shadow_train import (
                _action_outputs_with_grad,
            )

            outputs = _action_outputs_with_grad(
                planner, batch.state, batch.option_hidden, batch.legal_actions
            )
        top1 += int(int(outputs["scores"].argmax().item()) == batch.chosen_index)
        stages += 1
        programs.add(str(identity["program_identity"]))
    planner.train()
    adapter.train()
    return {
        "available": stages > 0,
        "programs": len(programs),
        "stages": stages,
        "chosen_top1_rate": top1 / max(1, stages),
        **{f"mean_{key}": value for key, value in _mean(rows).items()},
    }


def train_recent20_shadow(
    *,
    manifest_path: Path | str,
    manifest_sha256: str,
    base_pack_root: Path | str,
    base_completion_sha256: str,
    parent_checkpoint: Path | str,
    parent_checkpoint_sha256: str,
    output_root: Path | str,
    config: Recent20ShadowConfig,
    warm_start_checkpoint: Path | str | None = None,
    warm_start_checkpoint_sha256: str = "",
) -> dict[str, Any]:
    """Train a shadow-only planner without loading or mutating the parent."""
    started = time.time()
    parent = Path(parent_checkpoint).expanduser().resolve()
    if sha256_file(parent) != str(parent_checkpoint_sha256):
        raise RuntimeError("frozen parent checkpoint digest mismatch")
    out = Path(output_root).expanduser().resolve()
    if out.exists() or out.is_symlink():
        raise FileExistsError("recent20 shadow output root must be create-only")
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(int(config.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.seed))
    device = torch.device(config.device)

    dataset = Recent20RTPDataset(
        manifest_path,
        base_pack_root=base_pack_root,
        expected_manifest_sha256=manifest_sha256,
        expected_base_completion_sha256=base_completion_sha256,
        verify_overlay_shards=bool(config.verify_overlay_shards),
    )
    planner_cfg = _planner_config(config.d_model)
    warm_path = ""
    warm_digest = ""
    warm_adapter_restored = False
    if warm_start_checkpoint:
        warm = Path(warm_start_checkpoint).expanduser().resolve()
        warm_digest = sha256_file(warm)
        if warm_digest != str(warm_start_checkpoint_sha256):
            raise RuntimeError("RTP warm-start checkpoint digest mismatch")
        planner = load_rtp_checkpoint(
            warm,
            device=device,
            expected_config=planner_cfg,
            serving_qualified=False,
        )
        try:
            adapter = load_recent20_adapter_from_checkpoint(warm, device=device)
        except ValueError:
            adapter = Recent20SemanticAdapter(
                d_model=config.d_model, hidden_width=config.hidden_width
            ).to(device)
        else:
            if (
                adapter.d_model != int(config.d_model)
                or adapter.hidden_width != int(config.hidden_width)
            ):
                raise RuntimeError(
                    "RTP warm-start semantic adapter configuration mismatch"
                )
            warm_adapter_restored = True
        warm_path = str(warm)
    else:
        planner = RecursiveTurnPlanner(planner_cfg).to(device)
        adapter = Recent20SemanticAdapter(
            d_model=config.d_model, hidden_width=config.hidden_width
        ).to(device)
    planner.train()
    adapter.train()
    train_cfg = RTPTrainConfig(
        d_model=config.d_model,
        profile="pure_rl",
        epochs=config.epochs,
        lr=config.learning_rate,
        seed=config.seed,
        device=config.device,
        num_plan_candidates=4,
        max_recursion_depth=2,
        max_neural_passes=4,
        # The semantic adapter starts in a new latent basis. Keep dynamics
        # supervision bounded during bootstrap while action/value losses align
        # that basis to the warm planner.
        dynamics_weight=0.05,
    )
    optimizer = torch.optim.AdamW(
        list(planner.parameters()) + list(adapter.parameters()),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    epoch_history: list[dict[str, Any]] = []
    total_optimizer_steps = 0
    total_stages = 0
    total_programs: set[str] = set()
    for epoch in range(int(config.epochs)):
        optimizer.zero_grad(set_to_none=True)
        rows: list[dict[str, float]] = []
        accumulated = 0
        epoch_programs: set[str] = set()
        epoch_stages = 0
        for batch, identity in _stage_batches(
            dataset,
            "train",
            adapter=adapter,
            device=device,
            max_programs=int(config.max_train_programs),
            max_programs_per_day=int(config.max_train_programs_per_day),
        ):
            loss, metrics = train_step(planner, batch, cfg=train_cfg)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("recent20 RTP shadow loss is non-finite")
            (loss / int(config.gradient_accumulation_stages)).backward()
            rows.append({"loss": float(loss.detach().item()), **metrics})
            accumulated += 1
            epoch_stages += 1
            epoch_programs.add(str(identity["program_identity"]))
            if accumulated == int(config.gradient_accumulation_stages):
                torch.nn.utils.clip_grad_norm_(
                    list(planner.parameters()) + list(adapter.parameters()), 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_optimizer_steps += 1
                accumulated = 0
        if accumulated:
            torch.nn.utils.clip_grad_norm_(
                list(planner.parameters()) + list(adapter.parameters()), 1.0
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_optimizer_steps += 1
        if not epoch_stages:
            raise RuntimeError("recent20 shadow train split produced no stages")
        summary = {
            "epoch": epoch + 1,
            "programs": len(epoch_programs),
            "stages": epoch_stages,
            **{f"mean_{key}": value for key, value in _mean(rows).items()},
        }
        epoch_history.append(summary)
        total_stages += epoch_stages
        total_programs.update(epoch_programs)
        print(json.dumps({"phase": "train", **summary}, sort_keys=True), flush=True)

    validation = _evaluate(
        dataset, planner, adapter, cfg=config, train_cfg=train_cfg
    )
    checkpoint = out / "rtp_shadow_planner.pt"
    save_rtp_checkpoint(
        planner,
        checkpoint,
        metrics={
            "epoch": float(config.epochs),
            "mean_loss": float(epoch_history[-1]["mean_loss"]),
            "n_steps": float(total_stages),
            "optimizer_steps": float(total_optimizer_steps),
        },
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        shadow_only=True,
        research_only=False,
        extra={
            "recent20_semantic_adapter_config": adapter.config_json(),
            "recent20_semantic_adapter_state_dict": {
                key: value.detach().cpu() for key, value in adapter.state_dict().items()
            },
            "recent20_overlay_manifest_sha256": manifest_sha256,
            "recent20_base_completion_sha256": base_completion_sha256,
            "warm_start_checkpoint_sha256": warm_digest,
            "warm_start_semantic_adapter_restored": warm_adapter_restored,
            "training_unit": (
                "recorded_factorized_stage_grouped_under_complete_recorded_program"
            ),
            "evaluation_split_consumed": False,
            "serving_eligible": False,
            "action_authority_enabled": False,
        },
    )
    checkpoint.chmod(0o444)
    checkpoint_receipt = checkpoint.with_suffix(checkpoint.suffix + ".receipt.json")
    checkpoint_receipt.chmod(0o444)
    completion = {
        "schema": COMPLETION_SCHEMA,
        "status": "passed_shadow_only_bootstrap",
        "manifest_path": str(Path(manifest_path).resolve()),
        "manifest_sha256": manifest_sha256,
        "base_pack_root": str(Path(base_pack_root).resolve()),
        "base_pack_completion_sha256": base_completion_sha256,
        "parent_checkpoint_path": str(parent),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_checkpoint_loaded_or_mutated": False,
        "warm_start_checkpoint_path": warm_path,
        "warm_start_checkpoint_sha256": warm_digest,
        "warm_start_semantic_adapter_restored": warm_adapter_restored,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_receipt_path": str(checkpoint_receipt),
        "checkpoint_receipt_sha256": sha256_file(checkpoint_receipt),
        "adapter": {
            **adapter.config_json(),
            "parameters": sum(parameter.numel() for parameter in adapter.parameters()),
        },
        "planner_parameters": sum(parameter.numel() for parameter in planner.parameters()),
        "config": asdict(config),
        "train_programs": len(total_programs),
        "train_stages": total_stages,
        "optimizer_steps": total_optimizer_steps,
        "epoch_history": epoch_history,
        "validation": validation,
        "evaluation_split_consumed": False,
        "hidden_information_inputs_present": False,
        "unchosen_counterfactual_targets_present": False,
        "simulator_search_mcts_or_recollection_performed": False,
        "policy_checkpoint_or_active_optimizer_updated": False,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "elapsed_seconds": time.time() - started,
        "completed_at_unix_seconds": time.time(),
    }
    body = canonical_bytes(completion)
    digest = "sha256:" + hashlib.sha256(body).hexdigest()
    completion_path = out / f"sha256-{digest.removeprefix('sha256:')}.COMPLETE.json"
    _write_create_only(completion_path, body)
    result = {
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": completion["checkpoint_sha256"],
        "checkpoint_receipt_path": str(checkpoint_receipt),
        "completion_path": str(completion_path),
        "completion_sha256": digest,
        "metrics": epoch_history[-1],
        "validation": validation,
        "train_programs": len(total_programs),
        "train_stages": total_stages,
    }
    print(json.dumps({"phase": "complete", **result}, sort_keys=True), flush=True)
    return result


def load_recent20_adapter_from_checkpoint(
    checkpoint: Path | str, *, device: str | torch.device = "cpu"
) -> Recent20SemanticAdapter:
    """Load the bound semantic adapter for shadow evaluation only."""
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    extra = payload.get("extra") or {}
    config = extra.get("recent20_semantic_adapter_config") or {}
    if config.get("schema") != ADAPTER_SCHEMA:
        raise ValueError("checkpoint has no recognized recent20 semantic adapter")
    adapter = Recent20SemanticAdapter(
        d_model=int(config["d_model"]), hidden_width=int(config["hidden_width"])
    ).to(device)
    adapter.load_state_dict(extra["recent20_semantic_adapter_state_dict"], strict=True)
    return adapter
