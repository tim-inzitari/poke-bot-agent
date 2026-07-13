"""Deck-agnostic **core kernel** trunk shared across all archetypes.

This module is *additive*: it sits alongside the per-archetype bootstrap flow
(``poke_bot.train`` / ``scripts/train_bootstrap.py``) and never mutates it. The
idea is "generalist trunk, specialist fine-tune":

  * A single :class:`~poke_bot.model.TemporalCabtTransformer` is trained across
    the **whole classified ladder corpus** (every archetype bucket at once).
  * It carries **no archetype-specific priors**: card-id embeddings are shared
    across all decks (already true of the base model — the board/option bags are
    keyed by card id), and the only archetype signal is an *optional*
    conditioning embedding that defaults to ``"unknown"`` (index 0, initialised
    to zero) so the exact same trunk works for any deck with no conditioning.
  * Phase-6 specialists (``starmie``, ``lucario``, ``hammer-pult`` …) warm-start
    from the kernel: the trunk weights are copied, heads are optionally reinit'd,
    and the deck's conditioning offset is folded into ``cls_proj.bias`` so the
    resulting checkpoint is an ordinary ``TemporalCabtTransformer`` that the
    existing ``train_bootstrap`` loop can fine-tune with ``--resume auto``.

Why a wrapper and not a subclass?
---------------------------------
The base model is consumed directly by the (actively-developed) training loop in
``poke_bot.train`` via ``encode_board`` / ``pool_cls`` / ``temporal_encode`` /
``decode_options``. To avoid coupling to that churn we keep :class:`CoreKernel`
a thin wrapper around a plain ``TemporalCabtTransformer`` (``self.net``) plus one
extra ``archetype_embed`` table, and we provide our *own* streaming corpus and
loss/train loop here. Warm-started specialists are emitted as plain
``TemporalCabtTransformer`` state dicts so nothing downstream needs to know about
this module.
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import archetypes, checkpoint, config, device as device_mod, features, paths
from . import dataset as dataset_mod
from .dataset import GameSequence
from .model import TemporalCabtTransformer, build_model

Tensor = torch.Tensor
PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Archetype conditioning vocabulary (unknown == index 0 == no conditioning)
# ---------------------------------------------------------------------------

def kernel_archetype_ids() -> list[str]:
    """Conditioning vocabulary: ``["unknown", <registered archetypes...>]``.

    ``unknown`` is index 0 so the default conditioning is deck-agnostic (and the
    embedding is zero-initialised, i.e. a true no-op at start of training).
    """
    return [archetypes.UNKNOWN] + list(archetypes.archetype_ids())


def archetype_index(name: Optional[str]) -> int:
    """Map an archetype id to its conditioning index (unknown / unregistered → 0)."""
    if not name:
        return 0
    ids = kernel_archetype_ids()
    try:
        return ids.index(name)
    except ValueError:
        return 0


def _aux_archetype_label(name: str, num_classes: int) -> int:
    """Aux head target index (matches the ``archetype_ids()+["unknown"]`` order).

    Kept local so we do not import the actively-edited ``poke_bot.train``.
    """
    ids = list(archetypes.archetype_ids()) + [archetypes.UNKNOWN]
    idx = ids.index(name) if name in ids else len(ids) - 1
    return idx if idx < num_classes else num_classes - 1


# ---------------------------------------------------------------------------
# Hardware-tuned sizing
# ---------------------------------------------------------------------------

def core_kernel_config_3080ti() -> "config.ModelConfig":
    """Model sizing for the RTX 3080 Ti (~12 GB) core-kernel/specialist track.

    Lean generalist trunk that (a) trains comfortably on 12 GB with whole-game
    ``max_context=320`` sequences — a real probe measured peak ≈ 2 GB reserved,
    so VRAM is not the bottleneck — and (b) matches the ``d_model`` the Phase-6
    3080 Ti specialists use, so ``warm_start_specialist`` transfers the trunk 1:1.

    ``n_heads=6`` → head_dim=32 (even), which keeps RoPE valid.
    """
    return config.ModelConfig(
        d_model=192,
        spatial_layers=3,
        temporal_layers=4,
        option_decoder_layers=2,
        n_heads=6,
        ff_dim=768,
        max_context=320,
    )


# ---------------------------------------------------------------------------
# The core kernel wrapper
# ---------------------------------------------------------------------------

class CoreKernel(nn.Module):
    """Deck-agnostic trunk = ``TemporalCabtTransformer`` + archetype conditioning.

    The wrapped network (:attr:`net`) is architecturally identical to what
    :func:`poke_bot.model.build_model` produces, so its ``state_dict`` transfers
    1:1 into a Phase-6 specialist. The extra :attr:`archetype_embed` adds a
    per-archetype offset to the per-timestep ``[CLS]`` token; index 0 is
    ``unknown`` and is initialised to zero.
    """

    def __init__(
        self,
        cfg: Optional[config.ModelConfig] = None,
        *,
        device: Optional[torch.device] = None,
        aux_archetype_classes: Optional[int] = None,
        condition_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.cfg = cfg or config.MODEL
        if aux_archetype_classes is None:
            aux_archetype_classes = len(archetypes.archetype_ids()) + 1
        self.aux_archetype_classes = aux_archetype_classes
        self.net = build_model(
            self.cfg, device=None, aux_archetype_classes=aux_archetype_classes
        )
        self.condition_scale = float(condition_scale)
        self.n_archetype_tokens = len(kernel_archetype_ids())
        self.archetype_embed = nn.Embedding(self.n_archetype_tokens, self.cfg.d_model)
        nn.init.zeros_(self.archetype_embed.weight)  # unknown/default = no-op
        if device is not None:
            self.to(device)

    # ----- convenience delegates -----

    @property
    def d_model(self) -> int:
        return self.net.d_model

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ----- archetype-conditioned CLS -----

    def pool_cls_conditioned(
        self, spatial_memory: Tensor, arch_ids: Union[int, Tensor]
    ) -> Tensor:
        """Pool spatial memory then add the archetype conditioning offset.

        ``spatial_memory``: ``[B, 24, D]``. ``arch_ids`` is either a scalar
        conditioning index (broadcast over the batch) or a ``[B]`` LongTensor.
        Returns ``[B, D]``.
        """
        base = self.net.pool_cls(spatial_memory)  # [B, D]
        if not torch.is_tensor(arch_ids):
            arch_ids = torch.full(
                (base.size(0),), int(arch_ids), dtype=torch.long, device=base.device
            )
        arch_ids = arch_ids.to(base.device).clamp_(0, self.n_archetype_tokens - 1)
        return base + self.condition_scale * self.archetype_embed(arch_ids)

    # ----- forward (mirrors net.forward with conditioning) -----

    def forward(
        self,
        board,
        options,
        kv_cache=None,
        *,
        archetype_id: Union[str, int] = 0,
        append_cache: bool = True,
        n_options: Optional[Sequence[int]] = None,
    ) -> dict[str, Any]:
        """Full forward with archetype conditioning (default ``unknown``)."""
        arch_idx = (
            archetype_index(archetype_id)
            if isinstance(archetype_id, str)
            else int(archetype_id)
        )
        spatial = self.net.encode_board(board)
        cls = self.pool_cls_conditioned(spatial, arch_idx).unsqueeze(1)  # [B,1,D]
        state_vec, new_cache = self.net.temporal_encode(
            cls, kv_cache, append=append_cache
        )
        logits = self.net.decode_options(
            options, spatial, state_vec, n_options=n_options
        )
        value = torch.tanh(self.net.value_head(state_vec)).squeeze(-1)
        aux = self.net.aux_head(state_vec)
        return {
            "policy_logits": logits,
            "value": value,
            "aux_logits": aux,
            "state_vec": state_vec,
            "spatial_memory": spatial,
            "kv_cache": new_cache,
        }

    # ----- checkpoint I/O -----

    def save_core_kernel(
        self,
        path: PathLike,
        *,
        step: int = 0,
        epoch: int = 0,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scaler: Any = None,
        scheduler: Any = None,
        best_metric: Optional[float] = None,
        early_stop_state: Optional[dict] = None,
        extra: Optional[dict] = None,
    ) -> Path:
        """Atomically save a core-kernel checkpoint.

        The dict is a superset of a normal training checkpoint so that:
          * ``model_state_dict`` (the plain trunk) is directly loadable by any
            ``TemporalCabtTransformer`` (warm-start / inference), and
          * ``core_kernel_state_dict`` (trunk + archetype embedding) restores the
            full kernel for resumable core-kernel training.
        """
        ckpt = checkpoint.build_checkpoint(
            model=self.net,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            step=step,
            epoch=epoch,
            best_metric=best_metric,
            early_stop_state=early_stop_state,
            model_config=self.cfg,
            archetype_id=archetypes.UNKNOWN,
            model_id="core_kernel",
            extra=extra,
        )
        ckpt["is_core_kernel"] = True
        ckpt["core_kernel_state_dict"] = self.state_dict()
        ckpt["kernel_archetypes"] = kernel_archetype_ids()
        ckpt["condition_scale"] = self.condition_scale
        ckpt["aux_archetype_classes"] = self.aux_archetype_classes
        return checkpoint.atomic_torch_save(ckpt, path)

    @classmethod
    def load_core_kernel(
        cls,
        path: PathLike,
        *,
        device: Optional[torch.device] = None,
        cfg: Optional[config.ModelConfig] = None,
    ) -> "CoreKernel":
        """Load a :class:`CoreKernel` saved by :meth:`save_core_kernel`.

        If ``cfg`` is omitted, rebuilds :class:`~poke_bot.config.ModelConfig`
        from the checkpoint's ``model_config`` snapshot (required for smoke /
        lean checkpoints that differ from the live ``config.MODEL`` defaults).
        """
        ckpt = checkpoint.load_checkpoint(path, map_location=device or "cpu")
        if cfg is None:
            snap = ckpt.get("model_config")
            if isinstance(snap, dict):
                known = {
                    f.name for f in config.ModelConfig.__dataclass_fields__.values()  # type: ignore[attr-defined]
                }
                cfg = config.ModelConfig(
                    **{k: v for k, v in snap.items() if k in known and not isinstance(v, dict)}
                )
            else:
                cfg = config.MODEL
        kernel = cls(
            cfg=cfg,
            device=device,
            aux_archetype_classes=int(
                ckpt.get("aux_archetype_classes")
                or (len(archetypes.archetype_ids()) + 1)
            ),
            condition_scale=float(ckpt.get("condition_scale", 1.0)),
        )
        if "core_kernel_state_dict" in ckpt:
            kernel.load_state_dict(ckpt["core_kernel_state_dict"], strict=False)
        elif "model_state_dict" in ckpt:
            kernel.net.load_state_dict(ckpt["model_state_dict"], strict=True)
        if device is not None:
            kernel.to(device)
        return kernel

    # ----- warm-start API (Phase 6) -----

    def warm_start_specialist(
        self,
        archetype_id: str,
        *,
        reinit_heads: Union[bool, Sequence[str]] = ("policy_head", "aux_head"),
        fold_archetype: bool = True,
        device: Optional[torch.device] = None,
    ) -> TemporalCabtTransformer:
        """Create a Phase-6 specialist ``TemporalCabtTransformer`` from this trunk.

        Parameters
        ----------
        archetype_id:
            The specialist's archetype (e.g. ``"starmie"``, ``"hammer-pult"``).
            Only used for folding conditioning; it need not be registered.
        reinit_heads:
            Which output heads to reinitialise. Default ``("policy_head",
            "aux_head")`` keeps the **value head** — ladder evidence and
            BC→RL literature both say an accurate value is what makes search
            useful, and "first imitate then improve" needs a pretrained critic.
            Pass ``True`` to also reinit ``value_head``, an explicit name tuple
            for finer control, or ``False`` to keep all kernel heads.
        fold_archetype:
            If ``True`` (default), the archetype conditioning offset is folded
            into ``cls_proj.bias`` (exact, since the conditioned CLS is
            ``cls_proj(mean) + archetype_embed[idx]``). The specialist is then a
            plain trunk that carries the deck's conditioning with no extra module.

        Returns a plain :class:`~poke_bot.model.TemporalCabtTransformer` — the
        same class Phase-6 fine-tuning (``train_bootstrap``) expects.
        """
        spec = build_model(
            self.cfg,
            device=None,
            aux_archetype_classes=self.aux_archetype_classes,
        )
        spec.load_state_dict(self.net.state_dict(), strict=True)

        if reinit_heads:
            names = (
                ("policy_head", "value_head", "aux_head")
                if reinit_heads is True
                else tuple(reinit_heads)
            )
            for name in names:
                mod = getattr(spec, name, None)
                if mod is not None:
                    _reinit_module(mod)

        if fold_archetype:
            idx = archetype_index(archetype_id)
            with torch.no_grad():
                offset = self.archetype_embed.weight[idx].detach().to(
                    spec.cls_proj.bias.device
                )
                spec.cls_proj.bias.add_(self.condition_scale * offset)

        if device is not None:
            spec = spec.to(device)
        return spec


def _reinit_module(module: nn.Module) -> None:
    """Reset parameters of a head submodule (Linear / LayerNorm / Embedding)."""
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.LayerNorm, nn.Embedding)):
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()


def warm_start_specialist_from_checkpoint(
    core_ckpt_path: PathLike,
    archetype_id: str,
    *,
    run_name: Optional[str] = None,
    root: Optional[PathLike] = None,
    reinit_heads: Union[bool, Sequence[str]] = ("policy_head", "aux_head"),
    fold_archetype: bool = True,
    device: Optional[torch.device] = None,
    write_checkpoint: bool = True,
) -> dict[str, Any]:
    """Warm-start a specialist from a saved core kernel and (optionally) persist it.

    When ``write_checkpoint`` is True, a ``<run_name>.latest.pt`` checkpoint is
    written with ``step=0`` and **no optimizer state**. The existing
    ``scripts/train_bootstrap.py --archetype <id> --resume auto`` then resumes
    from it: the trunk weights load, but a fresh optimizer / schedule / epoch-0
    fine-tune begins (because no ``optimizer_state_dict`` is present).

    Returns a dict with the specialist model, run name and written paths.
    """
    kernel = CoreKernel.load_core_kernel(core_ckpt_path, device=device)
    spec = kernel.warm_start_specialist(
        archetype_id,
        reinit_heads=reinit_heads,
        fold_archetype=fold_archetype,
        device=device,
    )
    run_name = run_name or f"{archetype_id}_bootstrap"
    result: dict[str, Any] = {
        "run_name": run_name,
        "archetype_id": archetype_id,
        "specialist": spec,
        "reinit_heads": list(reinit_heads) if not isinstance(reinit_heads, bool) else reinit_heads,
        "fold_archetype": fold_archetype,
        "source": str(core_ckpt_path),
    }
    if write_checkpoint:
        ckpt = checkpoint.build_checkpoint(
            model=spec,
            optimizer=None,
            step=0,
            epoch=0,
            best_metric=None,
            model_config=kernel.cfg,
            archetype_id=archetype_id,
            model_id=run_name,
            extra={
                "warm_started_from": str(core_ckpt_path),
                "fold_archetype": fold_archetype,
            },
        )
        paths_out = checkpoint.save_checkpoint(
            ckpt, run_name, root=root, is_best=False, write_step_copy=False
        )
        result["paths"] = {k: str(v) for k, v in paths_out.items()}
    return result


# ---------------------------------------------------------------------------
# Streaming multi-archetype corpus (RAM-bounded: the corpus won't fit in RAM)
# ---------------------------------------------------------------------------

@dataclass
class CorpusConfig:
    """Configuration for the streaming multi-archetype corpus."""

    jsonl_paths: list[Path]
    val_frac: float = 0.05
    max_context: Optional[int] = None
    verify_info_set: bool = True
    #: Streaming shuffle buffer (number of GameSequences held in RAM at once).
    shuffle_buffer: int = 256
    seed: int = 0
    #: Cap sequences yielded per epoch (0 = all). Handy for smoke / debugging.
    max_sequences: int = 0


class StreamingArchetypeCorpus:
    """Streams :class:`GameSequence` objects from many archetype bucket JSONLs.

    Memory-bounded: records are read line-by-line and featurised on the fly; only
    a shuffle buffer of ``shuffle_buffer`` sequences ever lives in RAM. The
    whole corpus is re-streamed each epoch (cheap vs. holding it resident).

    Train/val split is **deterministic per (episode_id, seat)** via a stable
    hash, so a game never straddles the split and no full-corpus shuffle/index
    is required.
    """

    def __init__(self, cfg: CorpusConfig) -> None:
        self.cfg = cfg
        self.jsonl_paths = [Path(p) for p in cfg.jsonl_paths]
        missing = [p for p in self.jsonl_paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"corpus JSONL not found: {missing}")
        self.max_ctx = int(
            cfg.max_context if cfg.max_context is not None else config.MODEL.max_context
        )

    # ----- discovery -----

    @staticmethod
    def discover_bucket_jsonls(
        bucket_dir: Optional[PathLike] = None,
        *,
        include_smoke: bool = False,
        prefer_capped: bool = True,
    ) -> list[Path]:
        """Discover one JSONL per archetype bucket under ``bucket_dir``.

        Consumes *all* archetype buckets (the whole classified corpus). For each
        registered archetype it prefers ``<arch>.jsonl`` then ``<arch>.capped.jsonl``
        then any other ``<arch>*.jsonl`` (excluding ``*.smoke.jsonl`` unless
        ``include_smoke``). If no per-archetype files match, it falls back to
        every non-smoke ``*.jsonl`` in the directory.
        """
        bucket_dir = Path(bucket_dir) if bucket_dir else (paths.DATA_DIR / "bootstrap")
        if not bucket_dir.is_dir():
            return []

        def _ok(p: Path) -> bool:
            return include_smoke or ".smoke." not in p.name

        chosen: list[Path] = []
        seen: set[Path] = set()
        arch_ids = list(archetypes.archetype_ids())
        for arch in arch_ids:
            exact = bucket_dir / f"{arch}.jsonl"
            capped = bucket_dir / f"{arch}.capped.jsonl"
            pick: Optional[Path] = None
            if exact.is_file():
                pick = exact
            elif prefer_capped and capped.is_file():
                pick = capped
            else:
                cands = sorted(
                    p for p in bucket_dir.glob(f"{arch}*.jsonl") if _ok(p)
                )
                if cands:
                    pick = cands[0]
            if pick is not None and pick not in seen:
                chosen.append(pick)
                seen.add(pick)

        if chosen:
            return chosen

        # Fallback: any bootstrap JSONL present under the directory.
        return sorted(p for p in bucket_dir.glob("*.jsonl") if _ok(p))

    # ----- split -----

    def _split_of(self, episode_id: str, seat: int) -> str:
        raw = f"{episode_id}|{seat}".encode("utf-8")
        h = int(hashlib.sha1(raw).hexdigest()[:8], 16) % 10000
        return "val" if h < int(self.cfg.val_frac * 10000) else "train"

    # ----- streaming -----

    def _raw_sequences(self, split: str) -> Iterator[GameSequence]:
        for path in self.jsonl_paths:
            for record in dataset_mod.iter_jsonl(path):
                ep = str(record.get("episode_id", ""))
                seat = int(record.get("seat", 0))
                if split in ("train", "val") and self._split_of(ep, seat) != split:
                    continue
                seq = dataset_mod.record_to_sequence(
                    record,
                    max_context=self.max_ctx,
                    verify_info_set=self.cfg.verify_info_set,
                )
                if seq is None:
                    continue
                if not seq.source:
                    seq.source = path.name
                yield seq

    def iter_sequences(
        self,
        split: str = "train",
        *,
        epoch: int = 0,
        shuffle: bool = True,
    ) -> Iterator[GameSequence]:
        """Yield sequences for ``split`` ('train' | 'val' | 'all').

        With ``shuffle`` a streaming shuffle buffer of ``cfg.shuffle_buffer``
        sequences is used (bounded RAM). ``max_sequences`` caps the count.
        """
        rng = random.Random(self.cfg.seed + epoch * 100003)
        n_yielded = 0
        cap = self.cfg.max_sequences

        def _emit(seq: GameSequence) -> Optional[GameSequence]:
            nonlocal n_yielded
            if cap and n_yielded >= cap:
                return None
            n_yielded += 1
            return seq

        if not shuffle or self.cfg.shuffle_buffer <= 1:
            for seq in self._raw_sequences(split):
                out = _emit(seq)
                if out is None:
                    return
                yield out
            return

        buffer: list[GameSequence] = []
        buf_size = self.cfg.shuffle_buffer
        for seq in self._raw_sequences(split):
            if cap and n_yielded >= cap:
                break
            buffer.append(seq)
            if len(buffer) >= buf_size:
                j = rng.randrange(len(buffer))
                out = _emit(buffer.pop(j))
                if out is None:
                    return
                yield out
        rng.shuffle(buffer)
        for seq in buffer:
            out = _emit(seq)
            if out is None:
                return
            yield out


# ---------------------------------------------------------------------------
# Core-kernel training (streaming, AMP, early stop, resumable checkpoints)
# ---------------------------------------------------------------------------

@dataclass
class CoreTrainConfig:
    """Core-kernel supervised training knobs.

    Loss mix notes (from external research):
      * Value weight ≥ policy weight is justified — Kaggle ladder analysis
        (discussion/724362) finds search only helps with an accurate value
        head; BC→RL ("first imitate, then improve") also needs a pretrained
        critic for stable offline→online improvement.
      * Aux weight stays small (UNREAL-style representation learning via
        opponent-archetype prediction) without dominating the BC objective.
      * Soft ``policy_targets`` (MCTS visit distributions) are preferred over
        one-hot BC when present — AWR-ready path for later offline RL.
    """

    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    games_per_batch: int = 4
    max_decisions_per_batch: int = 256
    early_stop_patience: int = 5
    #: Prefer ≥1.0 so the shared value head is MCTS-ready for specialists.
    value_loss_weight: float = 1.5
    aux_loss_weight: float = 0.1
    grad_clip: float = 1.0
    amp: bool = True
    #: AMP compute dtype: ``bf16`` (Ampere+; no loss scaler needed, fp32 range)
    #: or ``fp16`` (uses GradScaler) or ``fp32`` (amp off).
    amp_dtype: str = "bf16"
    #: Optimizer step every N streamed micro-batches (effective batch = N × batch).
    grad_accum_steps: int = 1
    seed: int = 0
    #: Condition each sequence on its own archetype token during training.
    #: At inference / unknown decks, index 0 (``unknown``) is a zero-init no-op.
    condition_archetype: bool = True
    #: Cap val batches per epoch (0 = all val). Streaming val can be large.
    max_val_batches: int = 0


def resolve_amp_dtype(name: str) -> Optional[torch.dtype]:
    """Map an ``amp_dtype`` string to a torch dtype (``None`` → amp disabled)."""
    n = (name or "bf16").strip().lower()
    if n in ("bf16", "bfloat16"):
        return torch.bfloat16
    if n in ("fp16", "float16", "half"):
        return torch.float16
    if n in ("fp32", "float32", "none", "off"):
        return None
    raise ValueError(f"unknown amp_dtype {name!r}")


@dataclass
class BatchMetrics:
    policy_loss: float = 0.0
    value_loss: float = 0.0
    aux_loss: float = 0.0
    total_loss: float = 0.0
    policy_acc: float = 0.0
    n_decisions: int = 0
    n_games: int = 0


def core_sequence_losses(
    kernel: CoreKernel,
    seq: GameSequence,
    *,
    value_weight: float = 1.0,
    aux_weight: float = 0.1,
    condition: bool = True,
) -> tuple[Tensor, BatchMetrics]:
    """Whole-game forward + BC/value/aux losses for one sequence (conditioned).

    Mirrors ``poke_bot.train.sequence_losses`` but routes the CLS through the
    archetype-conditioned pooling so the trunk learns a shared representation
    plus per-archetype offsets.
    """
    net = kernel.net
    decisions = seq.decisions
    device = next(kernel.parameters()).device
    if not decisions:
        return torch.zeros((), device=device), BatchMetrics()

    boards = [d.board for d in decisions]
    spatial = net.encode_board(boards)  # [T, 24, D]
    arch_idx = archetype_index(seq.archetype) if condition else 0
    cls = kernel.pool_cls_conditioned(spatial, arch_idx).unsqueeze(0)  # [1, T, D]
    states, _ = net.temporal_encode(cls, None, append=False, return_all=True)
    target_value = torch.tensor(float(seq.value), device=states.device, dtype=states.dtype)

    policy_losses: list[Tensor] = []
    value_losses: list[Tensor] = []
    correct = 0
    n_valid = 0

    for t, d in enumerate(decisions):
        state_t = states[0, t]  # [D]
        spatial_t = spatial[t : t + 1]  # [1, 24, D]
        n_opt = d.options.num_words
        if n_opt <= 0:
            continue
        logits = net.decode_options(
            d.options, spatial_t, state_t.unsqueeze(0), n_options=[n_opt]
        )[0, :n_opt]

        if (
            seq.policy_targets is not None
            and t < len(seq.policy_targets)
            and seq.policy_targets[t] is not None
        ):
            target = torch.tensor(
                seq.policy_targets[t][:n_opt], device=logits.device, dtype=logits.dtype
            )
            if target.numel() != n_opt:
                continue
            target = target / target.sum().clamp_min(1e-8)
            log_p = F.log_softmax(logits, dim=-1)
            policy_losses.append(-(target * log_p).sum())
            correct += int(int(logits.argmax().item()) == int(target.argmax().item()))
        else:
            idx = int(d.action_combo_index)
            if idx < 0 or idx >= n_opt:
                continue
            policy_losses.append(
                F.cross_entropy(logits.unsqueeze(0), torch.tensor([idx], device=logits.device))
            )
            correct += int(int(logits.argmax().item()) == idx)

        value_pred = torch.tanh(net.value_head(state_t)).squeeze()
        value_losses.append(F.smooth_l1_loss(value_pred, target_value))
        n_valid += 1

    if n_valid == 0:
        return torch.zeros((), device=states.device, requires_grad=True), BatchMetrics(n_games=1)

    p_loss = torch.stack(policy_losses).mean()
    v_loss = torch.stack(value_losses).mean()
    aux_loss = torch.zeros((), device=states.device)
    if aux_weight > 0:
        aux_logits = net.aux_head(states[0, -1])
        label = torch.tensor(
            [_aux_archetype_label(seq.opp_archetype, kernel.aux_archetype_classes)],
            device=aux_logits.device,
            dtype=torch.long,
        )
        aux_loss = F.cross_entropy(aux_logits.unsqueeze(0), label)

    total = p_loss + value_weight * v_loss + aux_weight * aux_loss
    metrics = BatchMetrics(
        policy_loss=float(p_loss.detach().item()),
        value_loss=float(v_loss.detach().item()),
        aux_loss=float(aux_loss.detach().item()),
        total_loss=float(total.detach().item()),
        policy_acc=correct / max(n_valid, 1),
        n_decisions=n_valid,
        n_games=1,
    )
    return total, metrics


def _merge_metrics(parts: Sequence[BatchMetrics]) -> BatchMetrics:
    if not parts:
        return BatchMetrics()
    nd = sum(p.n_decisions for p in parts)
    ng = sum(p.n_games for p in parts)
    if nd == 0:
        return BatchMetrics(n_games=ng)

    def wavg(attr: str) -> float:
        return sum(getattr(p, attr) * p.n_decisions for p in parts) / nd

    return BatchMetrics(
        policy_loss=wavg("policy_loss"),
        value_loss=wavg("value_loss"),
        aux_loss=wavg("aux_loss"),
        total_loss=wavg("total_loss"),
        policy_acc=wavg("policy_acc"),
        n_decisions=nd,
        n_games=ng,
    )


def _stream_batches(
    seq_iter: Iterable[GameSequence],
    games_per_batch: int,
    max_decisions: int,
) -> Iterator[list[GameSequence]]:
    """Group a stream of sequences into decision-budgeted batches."""
    cur: list[GameSequence] = []
    cur_dec = 0
    for seq in seq_iter:
        n = len(seq)
        if cur and (len(cur) >= games_per_batch or cur_dec + n > max_decisions):
            yield cur
            cur, cur_dec = [], 0
        cur.append(seq)
        cur_dec += n
    if cur:
        yield cur


@torch.no_grad()
def evaluate_core(
    kernel: CoreKernel,
    corpus: StreamingArchetypeCorpus,
    *,
    cfg: CoreTrainConfig,
    epoch: int,
    desc: str = "val",
) -> BatchMetrics:
    from tqdm.auto import tqdm

    kernel.eval()
    parts: list[BatchMetrics] = []
    val_iter = corpus.iter_sequences("val", epoch=epoch, shuffle=False)
    batches = _stream_batches(val_iter, cfg.games_per_batch, cfg.max_decisions_per_batch)
    n_batches = 0
    for batch in tqdm(batches, desc=desc, leave=False, unit="batch"):
        for seq in batch:
            _, m = core_sequence_losses(
                kernel,
                seq,
                value_weight=cfg.value_loss_weight,
                aux_weight=cfg.aux_loss_weight,
                condition=cfg.condition_archetype,
            )
            parts.append(m)
        n_batches += 1
        if cfg.max_val_batches and n_batches >= cfg.max_val_batches:
            break
    return _merge_metrics(parts)


def train_core_kernel(
    corpus: StreamingArchetypeCorpus,
    *,
    run_name: str = "core_kernel",
    train_cfg: Optional[CoreTrainConfig] = None,
    resume: Union[str, bool, None] = "auto",
    device: Optional[torch.device] = None,
    cfg: Optional[config.ModelConfig] = None,
) -> dict[str, Any]:
    """Streaming supervised training of the deck-agnostic core kernel.

    AMP, early stopping, resumable checkpoints (``--resume auto``), tqdm and a
    RAM-bounded streaming corpus. Returns a JSON-safe result dict.
    """
    from tqdm.auto import tqdm

    tcfg = train_cfg or CoreTrainConfig()
    device = device or device_mod.training_device(
        prefer_name=config.HARDWARE.train_gpu_name, allow_cpu=False
    )
    torch.manual_seed(tcfg.seed)
    random.seed(tcfg.seed)

    kernel = CoreKernel(cfg=cfg or config.MODEL, device=device)
    optimizer = torch.optim.AdamW(
        kernel.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay
    )
    amp_dtype = resolve_amp_dtype(tcfg.amp_dtype)
    use_amp = bool(tcfg.amp and device.type == "cuda" and amp_dtype is not None)
    # GradScaler only needed for fp16; bf16 has fp32 dynamic range.
    use_scaler = bool(use_amp and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    accum = max(1, int(tcfg.grad_accum_steps))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(tcfg.epochs, 1)
    )

    step = 0
    start_epoch = 0
    best_metric = float("inf")
    patience_left = tcfg.early_stop_patience
    history: list[dict[str, Any]] = []

    mgr = checkpoint.CheckpointManager(run_name)
    resume_path = checkpoint.resolve_resume_path(run_name, resume)
    if resume_path is not None:
        print(f"[core-kernel] resuming from {resume_path}", flush=True)
        ckpt = checkpoint.load_checkpoint(resume_path, map_location=device)
        if "core_kernel_state_dict" in ckpt:
            kernel.load_state_dict(ckpt["core_kernel_state_dict"], strict=False)
        elif "model_state_dict" in ckpt:
            kernel.net.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if use_scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        checkpoint.restore_rng_state(ckpt.get("rng_state"))
        step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))
        if ckpt.get("best_metric") is not None:
            best_metric = float(ckpt["best_metric"])
        es = ckpt.get("early_stop_state") or {}
        patience_left = int(es.get("patience_left", patience_left))
        history = list((ckpt.get("extra") or {}).get("history") or [])

    def build_ckpt() -> dict[str, Any]:
        base = checkpoint.build_checkpoint(
            model=kernel.net,
            optimizer=optimizer,
            scaler=scaler if use_scaler else None,
            scheduler=scheduler,
            step=step,
            epoch=cur_epoch,
            best_metric=best_metric,
            early_stop_state={"patience_left": patience_left, "best_metric": best_metric},
            model_config=kernel.cfg,
            archetype_id=archetypes.UNKNOWN,
            model_id=run_name,
            extra={"history": history, "train_cfg": tcfg.__dict__},
        )
        base["is_core_kernel"] = True
        base["core_kernel_state_dict"] = kernel.state_dict()
        base["kernel_archetypes"] = kernel_archetype_ids()
        base["condition_scale"] = kernel.condition_scale
        base["aux_archetype_classes"] = kernel.aux_archetype_classes
        return base

    cur_epoch = start_epoch
    mgr.install_signal_flush(build_ckpt)
    try:
        epoch_bar = tqdm(
            range(start_epoch, tcfg.epochs),
            desc="epochs",
            initial=start_epoch,
            total=tcfg.epochs,
            unit="ep",
        )
        for epoch in epoch_bar:
            cur_epoch = epoch
            kernel.train()
            train_iter = corpus.iter_sequences("train", epoch=epoch, shuffle=True)
            batches = _stream_batches(
                train_iter, tcfg.games_per_batch, tcfg.max_decisions_per_batch
            )
            epoch_parts: list[BatchMetrics] = []
            batch_bar = tqdm(batches, desc=f"train ep{epoch}", leave=False, unit="batch")
            micro = 0  # micro-batches since last optimizer step (grad accumulation)
            optimizer.zero_grad(set_to_none=True)

            def _optim_step() -> None:
                nonlocal step
                if tcfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(kernel.parameters(), tcfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1

            for batch in batch_bar:
                losses: list[Tensor] = []
                metrics_parts: list[BatchMetrics] = []
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    for seq in batch:
                        loss, m = core_sequence_losses(
                            kernel,
                            seq,
                            value_weight=tcfg.value_loss_weight,
                            aux_weight=tcfg.aux_loss_weight,
                            condition=tcfg.condition_archetype,
                        )
                        losses.append(loss)
                        metrics_parts.append(m)
                    if not losses:
                        continue
                    total = torch.stack(losses).mean() / accum

                scaler.scale(total).backward()
                micro += 1
                if micro % accum == 0:
                    _optim_step()
                    saved = mgr.maybe_save(step, build_ckpt)
                    if saved:
                        tqdm.write(
                            f"[checkpoint] step={step} saved → "
                            + ", ".join(f"{k}={v.name}" for k, v in saved.items())
                        )

                bm = _merge_metrics(metrics_parts)
                epoch_parts.append(bm)
                batch_bar.set_postfix(
                    loss=f"{bm.total_loss:.3f}",
                    p=f"{bm.policy_loss:.3f}",
                    v=f"{bm.value_loss:.3f}",
                    acc=f"{bm.policy_acc:.2%}",
                    step=step,
                )

            # Flush any remaining accumulated grads at epoch end.
            if micro % accum != 0:
                _optim_step()

            train_m = _merge_metrics(epoch_parts)
            val_m = evaluate_core(kernel, corpus, cfg=tcfg, epoch=epoch, desc=f"val ep{epoch}")
            metric = val_m.total_loss if val_m.n_decisions else train_m.total_loss

            scheduler.step()
            history.append(
                {
                    "epoch": epoch,
                    "step": step,
                    "train": train_m.__dict__,
                    "val": val_m.__dict__,
                    "lr": optimizer.param_groups[0]["lr"],
                    "t": time.time(),
                }
            )

            is_best = metric < best_metric - 1e-5
            if is_best:
                best_metric = metric
                patience_left = tcfg.early_stop_patience
                mgr.save(build_ckpt(), is_best=True)
                tqdm.write(
                    f"[checkpoint] NEW BEST epoch={epoch} val_loss={metric:.4f} "
                    f"val_acc={val_m.policy_acc:.2%}"
                )
            else:
                patience_left -= 1
                mgr.save(build_ckpt(), is_best=False)
                tqdm.write(
                    f"[core-kernel] epoch={epoch} train_loss={train_m.total_loss:.4f} "
                    f"val_loss={metric:.4f} val_acc={val_m.policy_acc:.2%} "
                    f"patience={patience_left}"
                )

            epoch_bar.set_postfix(
                val_loss=f"{metric:.4f}", best=f"{best_metric:.4f}", pat=patience_left
            )
            if patience_left <= 0:
                tqdm.write(
                    f"[early-stop] patience exhausted at epoch={epoch} "
                    f"best_val_loss={best_metric:.4f}"
                )
                break
    finally:
        mgr.uninstall_signal_flush()
        try:
            mgr.save(build_ckpt(), is_best=False)
        except Exception as exc:  # noqa: BLE001
            print(f"[checkpoint] final save failed: {exc}", flush=True)

    best = checkpoint.best_path(run_name)
    latest = checkpoint.latest_path(run_name)
    return {
        "run_name": run_name,
        "best_metric": best_metric,
        "step": step,
        "epoch": cur_epoch,
        "best_path": str(best) if best.is_file() else None,
        "latest_path": str(latest) if latest.is_file() else None,
        "history": history,
    }
