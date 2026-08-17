"""Atomic checkpoint save/load with resume helpers.

Every training loop (bootstrap, round-robin RL, self-play) should checkpoint
periodically and resume from the newest matching file. This module owns:

  - atomic writes (temp path → ``os.replace``)
  - ``latest.pt`` / ``best.pt`` / rolling last-K
  - full train state: model, optim, scaler, RNG, step/epoch, config snapshot
  - SIGINT/SIGTERM flush helper
  - ``--resume auto`` path resolution
"""

from __future__ import annotations

import os
import hashlib
import random
import re
import signal
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch

from . import config, paths
from .matchup_adapters import ZERO_DORMANT_CHECKPOINT_SCHEMA

PathLike = Union[str, Path]


def _as_path(p: PathLike) -> Path:
    return Path(p).expanduser().resolve()


def _config_snapshot(obj: Any) -> Any:
    if obj is None:
        return None
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    return obj


def _matchup_adapter_bank(model: torch.nn.Module) -> Any:
    for module in model.modules():
        adapter_bank = getattr(module, "matchup_adapter_bank", None)
        if adapter_bank is not None:
            return adapter_bank
    return None


def validate_matchup_adapter_contract(
    ckpt: dict[str, Any],
    *,
    model: torch.nn.Module,
    source: Any = "checkpoint",
) -> None:
    """Fail closed when adapter weights lack their pinned route contract."""

    state = ckpt.get("model_state_dict")
    has_adapter_state = isinstance(state, dict) and any(
        "matchup_adapter_bank." in key for key in state
    )
    extra = ckpt.get("extra")
    saved_config = (
        extra.get("matchup_adapter_config")
        if isinstance(extra, dict)
        else None
    )
    if has_adapter_state and saved_config is None:
        raise ValueError(
            f"checkpoint {source} has matchup adapter state but is missing "
            "the matchup adapter routing contract"
        )

    adapter_bank = _matchup_adapter_bank(model)
    if saved_config is not None and adapter_bank is not None:
        current = adapter_bank.config_dict()
        if saved_config != current:
            raise ValueError(
                f"checkpoint {source} matchup adapter routing contract mismatch"
            )


def atomic_torch_save(obj: Any, path: PathLike) -> Path:
    """Write ``obj`` via a temp file then atomically replace ``path``."""
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return path


def immutable_torch_save(obj: Any, path: PathLike) -> Path:
    """Create ``path`` atomically and refuse to replace an existing artifact."""
    path = _as_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"immutable checkpoint already exists: {path}")
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp)
        # Hard-link publication is atomic and fails if another writer won.
        os.link(tmp, path)
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
    return path


def checkpoint_digest(path: PathLike, algorithm: str = "sha256") -> str:
    """Return a content digest used to bind evaluation to exact weights."""
    h = hashlib.new(algorithm)
    with _as_path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return f"{algorithm}:{h.hexdigest()}"


def candidate_path(
    run_name: str,
    iteration: int,
    root: Optional[PathLike] = None,
) -> Path:
    """Immutable iteration-specific candidate path."""
    return checkpoint_dir(root) / f"{run_name}.candidate.iter{int(iteration):06d}.pt"


def capture_rng_state() -> dict[str, Any]:
    """Snapshot python / numpy / torch / cuda RNG states."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except Exception:
        state["numpy"] = None
    if torch.cuda.is_available():
        try:
            state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["cuda"] = None
    else:
        state["cuda"] = None
    return state


def restore_rng_state(state: Optional[dict[str, Any]]) -> None:
    if not state:
        return
    if state.get("python") is not None:
        random.setstate(state["python"])
    if state.get("torch") is not None:
        tstate = state["torch"]
        if not isinstance(tstate, torch.Tensor):
            tstate = torch.tensor(tstate, dtype=torch.uint8)
        else:
            tstate = tstate.detach().cpu().to(dtype=torch.uint8)
        torch.set_rng_state(tstate)
    if state.get("numpy") is not None:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except Exception:
            pass
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            cuda_states = state["cuda"]
            fixed = []
            for s in cuda_states:
                if not isinstance(s, torch.Tensor):
                    s = torch.tensor(s, dtype=torch.uint8)
                else:
                    s = s.detach().cpu().to(dtype=torch.uint8)
                fixed.append(s)
            torch.cuda.set_rng_state_all(fixed)
        except Exception:
            pass


def checkpoint_dir(root: Optional[PathLike] = None) -> Path:
    d = _as_path(root) if root else paths.CHECKPOINTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def latest_path(run_name: str, root: Optional[PathLike] = None) -> Path:
    return checkpoint_dir(root) / f"{run_name}.latest.pt"


def best_path(run_name: str, root: Optional[PathLike] = None) -> Path:
    return checkpoint_dir(root) / f"{run_name}.best.pt"


def step_path(run_name: str, step: int, root: Optional[PathLike] = None) -> Path:
    return checkpoint_dir(root) / f"{run_name}.step{step:08d}.pt"


_STEP_RE = re.compile(r"\.step(\d+)\.pt$")


def list_rolling(run_name: str, root: Optional[PathLike] = None) -> list[Path]:
    d = checkpoint_dir(root)
    files = sorted(d.glob(f"{run_name}.step*.pt"))
    return files


def prune_rolling(run_name: str, keep_last_k: Optional[int] = None, root: Optional[PathLike] = None) -> None:
    keep = keep_last_k if keep_last_k is not None else config.CHECKPOINT.keep_last_k
    files = list_rolling(run_name, root)
    if keep <= 0 or len(files) <= keep:
        return
    for old in files[:-keep]:
        try:
            old.unlink()
        except OSError:
            pass


def build_checkpoint(
    *,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Any = None,
    scheduler: Any = None,
    step: int = 0,
    epoch: int = 0,
    rl_iteration: int = 0,
    best_metric: Optional[float] = None,
    early_stop_state: Optional[dict] = None,
    model_config: Any = None,
    extra: Optional[dict] = None,
    archetype_id: Optional[str] = None,
    model_id: Optional[str] = None,
) -> dict[str, Any]:
    """Assemble a full training-state checkpoint dict."""
    from . import features

    model_snapshot = _config_snapshot(model_config or config.MODEL)
    search_snapshot = _config_snapshot(config.SEARCH)
    model_state = model.state_dict()
    inventory_fn = getattr(model, "expanded_head_inventory", None)
    expanded_head_inventory = (
        inventory_fn()
        if callable(inventory_fn)
        else {
            "schema": "poke_bot.expanded_strategic_heads/v1",
            "version": 0,
            "enabled": False,
            "runtime_enabled_heads": [],
            "modules": {},
        }
    )
    fusion_inventory_fn = getattr(model, "decision_fusion_inventory", None)
    decision_fusion_inventory = (
        fusion_inventory_fn()
        if callable(fusion_inventory_fn)
        else {
            "schema": "poke_bot.causal_decision_fusion/v1",
            "enabled": False,
            "runtime_enabled": False,
            "required_heads": [],
            "parameters": 0,
        }
    )
    ckpt: dict[str, Any] = {
        "model_state_dict": model_state,
        "step": int(step),
        "epoch": int(epoch),
        "rl_iteration": int(rl_iteration),
        "best_metric": best_metric,
        "early_stop_state": early_stop_state,
        "model_config": model_snapshot,
        "search_config": search_snapshot,
        "hardware_config": _config_snapshot(config.HARDWARE),
        "archetype_id": archetype_id,
        "model_id": model_id,
        "rng_state": capture_rng_state(),
        "saved_at": time.time(),
        "torch_version": torch.__version__,
        "provenance": {
            "schema": 1,
            "feature_schema": features.FEATURE_SCHEMA_VERSION,
            "decision_context": model_snapshot.get("decision_context"),
            "search_mode": search_snapshot.get("mode"),
            "trusted_policy_path": (
                model_snapshot.get("decision_context") in {"history", "stateless"}
                and search_snapshot.get("mode") == "policy"
            ),
            "simulator": "competition_cg",
            "aux_heads_present": list(
                getattr(model, "aux_heads_present", ("aux_head",))
            ),
            "warm_started_belief_heads": list(
                getattr(model, "warm_started_belief_heads", ())
            ),
            "expanded_heads": expanded_head_inventory,
            "decision_fusion": decision_fusion_inventory,
            "warm_started_expanded_heads": list(
                getattr(model, "warm_started_expanded_heads", ())
            ),
            "warm_started_decision_fusion": bool(
                getattr(model, "warm_started_decision_fusion", False)
            ),
        },
    }
    if optimizer is not None:
        ckpt["optimizer_state_dict"] = optimizer.state_dict()
    if scaler is not None:
        ckpt["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    extra_payload = dict(extra or {})
    adapter_bank = _matchup_adapter_bank(model)
    if adapter_bank is not None and any(
        "matchup_adapter_bank." in key for key in model_state
    ):
        expected_adapter_config = adapter_bank.config_dict()
        supplied_adapter_config = extra_payload.get("matchup_adapter_config")
        if (
            supplied_adapter_config is not None
            and supplied_adapter_config != expected_adapter_config
        ):
            raise ValueError(
                "checkpoint matchup adapter routing contract mismatch"
            )
        extra_payload["matchup_adapter_config"] = expected_adapter_config
        runtime_enabled = bool(getattr(adapter_bank, "enabled", False))
        adapter_parameters = list(adapter_bank.parameters())
        training_enabled = any(
            bool(parameter.requires_grad) for parameter in adapter_parameters
        )
        adapter_parameter_ids = {id(parameter) for parameter in adapter_parameters}
        optimizer_parameter_ids = (
            {
                id(parameter)
                for group in optimizer.param_groups
                for parameter in group.get("params", ())
            }
            if optimizer is not None
            else set()
        )
        optimizer_included = bool(adapter_parameter_ids & optimizer_parameter_ids)
        up_tensors = [
            value
            for name, value in adapter_bank.state_dict().items()
            if name.endswith("up.weight") or name.endswith("up.bias")
        ]
        zero_output = bool(
            up_tensors
            and all(
                int(value.detach().count_nonzero().item()) == 0
                for value in up_tensors
            )
        )
        extra_payload["matchup_adapters_runtime_enabled"] = runtime_enabled
        extra_payload["matchup_adapter_training_enabled"] = training_enabled
        extra_payload["matchup_adapter_optimizer_included"] = optimizer_included
        if not runtime_enabled and not training_enabled:
            if optimizer_included:
                raise ValueError(
                    "frozen matchup adapters leaked into the ordinary optimizer"
                )
            if not zero_output and extra_payload.get("pure_rl") is True:
                fit = dict(
                    extra_payload.get("dormant_matchup_adapter_fit") or {}
                )
                optimizer_state = dict(
                    extra_payload.get(
                        "dormant_matchup_adapter_optimizer_state"
                    )
                    or {}
                )
                route_rows = dict(fit.get("route_decisions") or {})
                if not (
                    fit.get("schema")
                    == "poke_bot.dormant_matchup_adapter_fit/v1"
                    and fit.get("runtime_enabled") is False
                    and fit.get("base_frozen") is True
                    and fit.get("optimizer_scope")
                    == "matchup_adapter_bank_only"
                    and int(fit.get("epochs", 0)) > 0
                    and int(fit.get("steps", 0)) > 0
                    and int(fit.get("rows", 0)) > 0
                    and sum(int(value) for value in route_rows.values()) > 0
                    and optimizer_state
                ):
                    raise ValueError(
                        "ordinary pure-RL checkpoint cannot persist non-zero "
                        "dormant adapters without a complete "
                        "isolated fit receipt and continuation optimizer state"
                    )
            inherited_dormant = dict(
                extra_payload.get("dormant_matchup_adapter_bank") or {}
            )
            provenance = dict(
                getattr(adapter_bank, "dormant_provenance", {}) or {}
            )
            extra_payload["dormant_matchup_adapter_bank"] = {
                **inherited_dormant,
                **provenance,
                "schema": (
                    "poke_bot.trained_dormant_matchup_adapter/v1"
                    if not zero_output
                    else ZERO_DORMANT_CHECKPOINT_SCHEMA
                ),
                "runtime_enabled": False,
                "training_enabled": False,
                "optimizer_imported": False,
                "optimizer_present": optimizer is not None,
                "optimizer_included": False,
                "frozen": True,
                "zero_output": zero_output,
                "parameter_count": sum(
                    int(parameter.numel()) for parameter in adapter_parameters
                ),
                "adapter_config": expected_adapter_config,
            }
    if extra_payload:
        ckpt["extra"] = extra_payload
    return ckpt


def save_checkpoint(
    ckpt: dict[str, Any],
    run_name: str,
    *,
    root: Optional[PathLike] = None,
    is_best: bool = False,
    write_step_copy: bool = True,
    keep_last_k: Optional[int] = None,
) -> dict[str, Path]:
    """Atomically write latest (+ optional best + rolling step copy)."""
    paths_out: dict[str, Path] = {}
    latest = latest_path(run_name, root)
    atomic_torch_save(ckpt, latest)
    paths_out["latest"] = latest

    if is_best:
        best = best_path(run_name, root)
        atomic_torch_save(ckpt, best)
        paths_out["best"] = best

    if write_step_copy:
        step = int(ckpt.get("step", 0))
        sp = step_path(run_name, step, root)
        atomic_torch_save(ckpt, sp)
        paths_out["step"] = sp
        prune_rolling(run_name, keep_last_k=keep_last_k, root=root)

    return paths_out


def load_checkpoint(
    path: PathLike,
    *,
    map_location: Union[str, torch.device] = "cpu",
) -> dict[str, Any]:
    path = _as_path(path)
    return torch.load(path, map_location=map_location, weights_only=False)


def assert_trusted_policy_checkpoint(path: PathLike) -> dict[str, Any]:
    """Fail closed unless ``path`` can serve a non-privileged policy.

    Realized-history and state-only policies both consume deployment-visible
    observations. Oracle search and checkpoints explicitly marked untrusted
    remain rejected.
    """
    ckpt = load_checkpoint(path, map_location="cpu")
    if ckpt.get("schema") == (
        "poke_bot.alakazam_rule_derivative_composite_candidate_initialization/v1"
    ):
        revision = int(ckpt.get("goal_revision", -1))
        if (
            revision not in {9, 10}
            or not str(ckpt.get("goal_contract_sha256", "")).startswith("sha256:")
            or not isinstance(ckpt.get("base_model_state_dict"), dict)
            or not isinstance(
                ckpt.get("public_rule_semantic_projection_state_dict"), dict
            )
            or "public_rule_semantic_projection"
            not in list(ckpt.get("eligible_trainable_branches") or ())
        ):
            raise ValueError("rule-derivative candidate checkpoint is not trusted")
    model_cfg = dict(ckpt.get("model_config") or {})
    provenance = dict(ckpt.get("provenance") or {})
    decision_context = str(
        model_cfg.get("decision_context", provenance.get("decision_context", "history"))
    ).lower()
    if decision_context not in {"history", "stateless"}:
        raise ValueError(
            f"checkpoint decision_context={decision_context!r}; trusted deployment "
            "requires history or stateless policy context"
        )
    if provenance and provenance.get("trusted_policy_path") is False:
        raise ValueError("checkpoint provenance explicitly marks path untrusted")
    search_cfg = dict(ckpt.get("search_config") or {})
    if str(search_cfg.get("mode", "policy")).lower() == "oracle_mcts":
        raise ValueError("oracle-MCTS checkpoint provenance cannot be deployed")
    return {
        "decision_context": decision_context,
        "provenance": provenance,
        "model_config": model_cfg,
    }


def apply_checkpoint(
    ckpt: dict[str, Any],
    *,
    model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Any = None,
    scheduler: Any = None,
    restore_rng: bool = True,
    strict: bool = True,
) -> dict[str, Any]:
    """Load weights / optim / scaler / RNG from ``ckpt`` into live objects."""
    if model is not None and "model_state_dict" in ckpt:
        validate_matchup_adapter_contract(ckpt, model=model)
        model.load_state_dict(ckpt["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scaler is not None and "scaler_state_dict" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if restore_rng:
        restore_rng_state(ckpt.get("rng_state"))
    return {
        "step": int(ckpt.get("step", 0)),
        "epoch": int(ckpt.get("epoch", 0)),
        "rl_iteration": int(ckpt.get("rl_iteration", 0)),
        "best_metric": ckpt.get("best_metric"),
        "early_stop_state": ckpt.get("early_stop_state"),
        "extra": ckpt.get("extra") or {},
    }


def resolve_resume_path(
    run_name: str,
    resume: Optional[Union[str, bool]] = None,
    *,
    root: Optional[PathLike] = None,
) -> Optional[Path]:
    """Resolve ``--resume`` into a checkpoint path.

    - ``None`` / ``"auto"`` → latest if present, else None
    - ``False`` / ``"0"`` / ``"none"`` → never resume
    - ``True`` / ``"1"`` / ``"require"`` → latest, error if missing
    - otherwise treat as an explicit filesystem path
    """
    if resume is None:
        resume = config.CHECKPOINT.resume

    if isinstance(resume, bool):
        mode = "require" if resume else "0"
    else:
        mode = str(resume).strip()

    lower = mode.lower()
    latest = latest_path(run_name, root)

    if lower in {"0", "false", "no", "off", "none"}:
        return None
    if lower in {"auto", ""}:
        return latest if latest.is_file() else None
    if lower in {"1", "true", "yes", "on", "require", "force", "latest"}:
        if not latest.is_file():
            raise FileNotFoundError(
                f"resume={mode!r} requested but no checkpoint at {latest}"
            )
        return latest
    if lower == "best":
        best = best_path(run_name, root)
        if not best.is_file():
            raise FileNotFoundError(f"resume=best but missing {best}")
        return best

    path = _as_path(mode)
    if not path.is_file():
        raise FileNotFoundError(f"resume path not found: {path}")
    return path


class CheckpointManager:
    """Convenience wrapper: cadence checks + SIGINT flush."""

    def __init__(
        self,
        run_name: str,
        *,
        root: Optional[PathLike] = None,
        every_steps: Optional[int] = None,
        every_minutes: Optional[float] = None,
        keep_last_k: Optional[int] = None,
    ):
        self.run_name = run_name
        self.root = root
        self.every_steps = (
            every_steps if every_steps is not None else config.CHECKPOINT.every_steps
        )
        self.every_minutes = (
            every_minutes
            if every_minutes is not None
            else config.CHECKPOINT.every_minutes
        )
        self.keep_last_k = (
            keep_last_k if keep_last_k is not None else config.CHECKPOINT.keep_last_k
        )
        self._last_save_t = time.time()
        self._last_save_step = 0
        self._pending_builder: Optional[Callable[[], dict[str, Any]]] = None
        self._installed_signals = False
        self._prev_handlers: dict[int, Any] = {}

    def should_save(self, step: int) -> bool:
        if self.every_steps > 0 and (step - self._last_save_step) >= self.every_steps:
            return True
        if self.every_minutes > 0:
            if (time.time() - self._last_save_t) >= self.every_minutes * 60.0:
                return True
        return False

    def save(
        self,
        ckpt: dict[str, Any],
        *,
        is_best: bool = False,
    ) -> dict[str, Path]:
        out = save_checkpoint(
            ckpt,
            self.run_name,
            root=self.root,
            is_best=is_best,
            write_step_copy=True,
            keep_last_k=self.keep_last_k,
        )
        self._last_save_t = time.time()
        self._last_save_step = int(ckpt.get("step", 0))
        return out

    def maybe_save(
        self,
        step: int,
        builder: Callable[[], dict[str, Any]],
        *,
        is_best: bool = False,
        force: bool = False,
    ) -> Optional[dict[str, Path]]:
        self._pending_builder = builder
        if force or self.should_save(step):
            return self.save(builder(), is_best=is_best)
        return None

    def install_signal_flush(self, builder: Callable[[], dict[str, Any]]) -> None:
        """Best-effort checkpoint on SIGINT/SIGTERM then re-raise default."""
        self._pending_builder = builder
        if self._installed_signals:
            return

        def _handler(signum, frame):  # noqa: ARG001
            try:
                if self._pending_builder is not None:
                    self.save(self._pending_builder())
            except Exception as exc:  # noqa: BLE001
                print(f"[checkpoint] signal flush failed: {exc}", flush=True)
            # Restore and re-raise.
            prev = self._prev_handlers.get(signum, signal.SIG_DFL)
            signal.signal(signum, prev)
            if callable(prev):
                prev(signum, frame)
            elif prev == signal.SIG_DFL:
                signal.raise_signal(signum)

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._prev_handlers[sig] = signal.signal(sig, _handler)
            except Exception:
                pass
        self._installed_signals = True

    def uninstall_signal_flush(self) -> None:
        for sig, prev in self._prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except Exception:
                pass
        self._prev_handlers.clear()
        self._installed_signals = False
