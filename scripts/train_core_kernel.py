#!/usr/bin/env python
"""Train the deck-agnostic **core kernel** over ALL archetype buckets.

The core kernel is a shared "generalist trunk" (see ``poke_bot.core_kernel``)
trained across the whole classified ladder corpus. Phase-6 specialists
(``starmie``, ``lucario``, ``hammer-pult`` …) then warm-start from it.

This is *additive* — it does not touch the per-archetype bootstrap flow.

Device / dual-GPU
-----------------
Per the dual-GPU plan the RTX 3080 Ti handles kernel / specialist training while
the Blackwell runs the primary Dragapult job. This script forces
``CUDA_DEVICE_ORDER=PCI_BUS_ID`` so torch indices match ``nvidia-smi`` (index 0 =
3080 Ti), and ``--device auto`` pins to the 3080 Ti *by name*. Pin explicitly::

    # default: auto-pins to the 3080 Ti by name
    python scripts/train_core_kernel.py --gpu-profile 3080ti --device auto
    # or isolate the 3080 Ti entirely (PCI order → index 0 = 3080 Ti):
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        python scripts/train_core_kernel.py --device cuda
    # NOTE: a bare CUDA_VISIBLE_DEVICES=0 (default torch order) selects the
    # BLACKWELL, not the 3080 Ti — always set CUDA_DEVICE_ORDER=PCI_BUS_ID.

Examples
--------
Smoke test (build model, forward, save ckpt, warm-start a specialist)::

    /home/inzi/miniconda3/envs/poke-bot-agent/bin/python scripts/train_core_kernel.py --smoke

Throughput / VRAM probe on the 3080 Ti (a handful of real steps)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        /home/inzi/miniconda3/envs/poke-bot-agent/bin/python \\
        scripts/train_core_kernel.py --probe --gpu-profile 3080ti --device cuda

Full run (documented; do NOT launch while Phase 3-5 owns the GPUs)::

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \\
        /home/inzi/miniconda3/envs/poke-bot-agent/bin/python \\
        scripts/train_core_kernel.py --device cuda --gpu-profile 3080ti \\
        --run-name core_kernel --epochs 20 --resume auto
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make torch's device indices match ``nvidia-smi`` (PCI bus order) so a
# ``CUDA_VISIBLE_DEVICES=0`` pin is unambiguous. MUST be set before torch
# initialises CUDA. Default torch order is "fastest first" → Blackwell would be
# index 0 and the 3080 Ti index 1; PCI order puts the 3080 Ti at index 0.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from poke_bot import config, device as device_mod, paths
from poke_bot.core_kernel import (
    CoreKernel,
    CoreTrainConfig,
    CorpusConfig,
    StreamingArchetypeCorpus,
    core_kernel_config_3080ti,
    core_kernel_config_small_3080ti,
    core_batch_losses,
    core_sequence_losses,
    resolve_amp_dtype,
    train_core_kernel,
    warm_start_specialist_from_checkpoint,
)


# ---------------------------------------------------------------------------
# GPU profiles — sizing presets tuned per card (dual-GPU plan)
# ---------------------------------------------------------------------------

def _profile_3080ti() -> tuple[config.ModelConfig, dict, str]:
    """RTX 3080 Ti (~12 GB) preset: lean generalist trunk, bf16, grad-accum.

    Sized so the kernel + AdamW states + whole-game (MAX_CONTEXT=320) activations
    fit comfortably in 12 GB with the card to itself, while matching the lean
    ``d_model`` the Phase-6 3080 specialists want (warm-start shares this arch).
    """
    cfg = core_kernel_config_3080ti()  # d_model=192, 3/4/2, heads=6, ff=768, ctx=320
    # Batch sizes come from the centralized 3080 Ti profile (config.batch_profile)
    # so this stays the single source of truth. Probe on the 3080 Ti showed peak
    # reserved ≈ 2 GB (the lean d_model=192 kernel is CPU-featurization-bound), so
    # the bold profile sits comfortably under the ~10 GB (~85%) VRAM target; the
    # OOM guard in the train loop is the crash-safety net if a spike exceeds it.
    bp = config.batch_profile("3080ti")
    train = dict(
        games_per_batch=bp.games_per_batch,
        max_decisions_per_batch=bp.max_decisions_per_batch,
        shuffle_buffer=bp.shuffle_buffer,
        amp_dtype=bp.amp_dtype,
        grad_accum_steps=bp.grad_accum_steps,
        value_loss_weight=1.5,
        aux_loss_weight=0.1,
    )
    return cfg, train, config.HARDWARE.leaf_gpu_name  # "3080"


def _profile_small_3080ti() -> tuple[config.ModelConfig, dict, str]:
    """Small search-oriented kernel that stays shape-compatible with its specialist."""
    cfg = core_kernel_config_small_3080ti()
    bp = config.batch_profile("3080ti")
    train = dict(
        games_per_batch=min(bp.games_per_batch, 16),
        max_decisions_per_batch=min(bp.max_decisions_per_batch, 1024),
        shuffle_buffer=min(bp.shuffle_buffer, 1024),
        amp_dtype=bp.amp_dtype,
        grad_accum_steps=1,
        value_loss_weight=1.5,
        aux_loss_weight=0.1,
    )
    return cfg, train, config.HARDWARE.leaf_gpu_name


def _profile_blackwell() -> tuple[config.ModelConfig, dict, str]:
    """RTX PRO 5000 Blackwell (~48 GB) preset: full-width trunk (side use only).

    Batch sizes come from the centralized Blackwell profile (config.batch_profile),
    sized toward the ~40 GB (~85%) VRAM target; the train-loop OOM guard backstops
    any spike beyond it.
    """
    cfg = config.ModelConfig()  # defaults (d_model=256, 4/4/2, heads=8, ff=1024)
    bp = config.batch_profile("blackwell")
    train = dict(
        games_per_batch=bp.games_per_batch,
        max_decisions_per_batch=bp.max_decisions_per_batch,
        shuffle_buffer=bp.shuffle_buffer,
        amp_dtype=bp.amp_dtype,
        grad_accum_steps=bp.grad_accum_steps,
        value_loss_weight=1.5,
        aux_loss_weight=0.1,
    )
    return cfg, train, config.HARDWARE.train_gpu_name  # "Blackwell"


def gpu_profile(name: str) -> tuple[config.ModelConfig | None, dict, str | None]:
    n = (name or "3080ti").strip().lower()
    if n in ("small-3080ti", "small", "search-small"):
        return _profile_small_3080ti()
    if n in ("3080ti", "3080", "ampere"):
        return _profile_3080ti()
    if n in ("blackwell", "5000", "pro5000"):
        return _profile_blackwell()
    if n in ("none", "default", "cpu"):
        return None, {}, None
    raise SystemExit(f"unknown --gpu-profile {name!r}")


def _resolve_device(spec: str, prefer_name: str | None = None) -> torch.device:
    """Resolve a ``--device`` spec into a torch device.

    ``auto`` pins to ``prefer_name`` (e.g. the 3080 Ti for the 3080ti profile) by
    GPU name so a bare invocation lands on the intended card even without
    ``CUDA_VISIBLE_DEVICES``. Honours an explicit ``cuda:N`` or a caller-set
    ``CUDA_VISIBLE_DEVICES`` (which remaps indices, so cuda:0 = the visible card).
    """
    spec = (spec or "auto").strip().lower()
    if spec == "cpu":
        return torch.device("cpu")
    if spec == "cuda":
        return torch.device("cuda")
    if spec.startswith("cuda:"):
        return torch.device(spec)
    if spec in ("auto", ""):
        if not device_mod.cuda_available():
            return torch.device("cpu")
        # If the user already isolated a card via CVD, cuda:0 IS that card.
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            return torch.device("cuda:0")
        if prefer_name:
            idx = device_mod.find_gpu_by_name(prefer_name)
            if idx is not None:
                return torch.device(f"cuda:{idx}")
        return torch.device("cuda:0")
    return torch.device(spec)


def _resolve_jsonls(args: argparse.Namespace) -> list[Path]:
    if args.jsonl:
        paths_out: list[Path] = []
        for pattern in args.jsonl:
            p = Path(pattern)
            if p.is_file():
                paths_out.append(p)
            else:
                paths_out.extend(sorted(Path().glob(pattern)))
        return paths_out
    return StreamingArchetypeCorpus.discover_bucket_jsonls(
        args.bucket_dir, include_smoke=args.include_smoke
    )


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--jsonl", nargs="*", default=None,
                   help="Explicit bucket JSONL paths/globs (default: auto-discover buckets).")
    p.add_argument("--bucket-dir", type=Path, default=paths.DATA_DIR / "bootstrap",
                   help="Directory of per-archetype bucket JSONLs.")
    p.add_argument("--include-smoke", action="store_true",
                   help="Include *.smoke.jsonl buckets in discovery.")
    p.add_argument("--run-name", default="core_kernel")
    p.add_argument("--resume", default="auto", help="auto | 0 | path | best")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    p.add_argument("--gpu-profile", default="3080ti",
                   help="Sizing preset: small-3080ti | 3080ti (default) | blackwell | none")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--games-per-batch", type=int, default=None,
                   help="Override profile games-per-batch.")
    p.add_argument("--max-decisions-per-batch", type=int, default=None,
                   help="Override profile max-decisions-per-batch.")
    p.add_argument("--grad-accum-steps", type=int, default=None,
                   help="Override profile grad-accum (optimizer step every N micro-batches).")
    p.add_argument("--amp-dtype", default=None, help="bf16 | fp16 | fp32 (default from profile).")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--shuffle-buffer", type=int, default=None,
                   help="Override profile shuffle buffer.")
    p.add_argument("--max-sequences", type=int, default=0, help="Cap seqs/epoch (0=all).")
    p.add_argument("--max-val-batches", type=int, default=0, help="Cap val batches (0=all).")
    p.add_argument("--no-condition", action="store_true",
                   help="Disable archetype conditioning token (pure deck-agnostic).")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-verify-info-set", action="store_true",
                   help="Skip per-step info-set verification (faster; already verified upstream).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true",
                   help="Run the verification smoke test instead of training.")
    p.add_argument("--probe", action="store_true",
                   help="Short real throughput/VRAM probe on the target GPU (no full run).")
    p.add_argument("--probe-steps", type=int, default=12,
                   help="Optimizer steps for --probe (kept small to spare CPU workers).")
    p.add_argument("--warm-start", default=None,
                   help="Archetype id: warm-start a specialist from the kernel and exit.")
    p.add_argument("--core-ckpt", type=Path, default=None,
                   help="Core kernel checkpoint for --warm-start (default: <run-name>.best/latest).")
    p.add_argument(
        "--reinit-heads",
        default="policy,aux",
        help=(
            "Comma list of heads to reinit on --warm-start "
            "(default: policy,aux — keep value for MCTS). "
            "Use 'all' to also reinit value; 'none' to keep all heads."
        ),
    )
    p.add_argument("--value-loss-weight", type=float, default=None,
                   help="Value loss weight (profile default 1.5; ladder/MCTS needs accurate value).")
    p.add_argument("--aux-loss-weight", type=float, default=None,
                   help="Aux (opponent-archetype) loss weight (profile default 0.1).")
    p.add_argument(
        "--opp-hand-loss-weight",
        type=float,
        default=None,
        help="opp_hand_head multilabel BCE weight (default 0.2; masked if labels absent).",
    )
    p.add_argument(
        "--opp-remainder-loss-weight",
        type=float,
        default=None,
        help="opp_remainder_head multilabel BCE weight (default 0.15; masked if absent).",
    )
    p.add_argument(
        "--lethal-threat-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Scope B only — keep 0 on core_kernel (Blackwell Hammer strategy "
            "heads are not trained here)."
        ),
    )
    p.add_argument(
        "--prize-race-loss-weight",
        type=float,
        default=0.0,
        help="Scope B only — keep 0 on core_kernel.",
    )
    return p.parse_args(argv)


def _parse_reinit_heads(spec: str):
    s = (spec or "policy,aux").strip().lower()
    if s in ("none", "0", "false", "keep"):
        return False
    if s in ("all", "true", "1"):
        return True
    mapping = {
        "policy": "policy_head",
        "policy_head": "policy_head",
        "value": "value_head",
        "value_head": "value_head",
        "aux": "aux_head",
        "aux_head": "aux_head",
        "opp_hand": "opp_hand_head",
        "opp_hand_head": "opp_hand_head",
        "opp_remainder": "opp_remainder_head",
        "opp_remainder_head": "opp_remainder_head",
        # Scope B heads exist on the architecture for warm-start but core
        # training does not reinit/require them by default.
        "lethal": "lethal_threat_head",
        "lethal_threat": "lethal_threat_head",
        "lethal_threat_head": "lethal_threat_head",
        "prize_race": "prize_race_head",
        "prize_race_head": "prize_race_head",
    }
    names = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in mapping:
            raise SystemExit(f"Unknown --reinit-heads token: {part!r}")
        names.append(mapping[part])
    return tuple(names) or False


def _run_smoke(args: argparse.Namespace) -> int:
    """Build model, forward pass, save checkpoint, warm-start a specialist."""
    import poke_bot.paths as _paths

    _paths.ensure_runtime_dirs()
    device = _resolve_device(args.device)
    print(f"== core_kernel SMOKE  device={device}  ({device_mod.describe()})", flush=True)

    # 1) Build a tiny kernel (small config so it is fast on CPU/GPU).
    cfg = config.ModelConfig(
        d_model=64, spatial_layers=1, temporal_layers=1,
        option_decoder_layers=1, n_heads=4, ff_dim=128, max_context=32,
    )
    kernel = CoreKernel(cfg=cfg, device=device)
    n_params = sum(p.numel() for p in kernel.parameters())
    print(f">> built CoreKernel: {n_params:,} params, "
          f"archetype_tokens={kernel.n_archetype_tokens}, d_model={cfg.d_model}", flush=True)

    # 2) Forward pass on synthetic SparseVectors (no engine dependency).
    from poke_bot import features as feat

    def _rand_board() -> feat.SparseVector:
        sv = feat.SparseVector()
        for _ in range(feat.NUM_BOARD_TOKENS):
            sv.word_start()
            sv.add(int(torch.randint(0, kernel.net.encoder_vocab, (1,)).item()), 1.0)
        return sv

    def _rand_options(n: int) -> feat.SparseVector:
        sv = feat.SparseVector()
        for _ in range(n):
            sv.word_start()
            sv.add(int(torch.randint(0, kernel.net.decoder_vocab, (1,)).item()), 1.0)
        return sv

    board = _rand_board()
    opts = _rand_options(5)
    kernel.eval()
    with torch.no_grad():
        out_unknown = kernel(board, opts, archetype_id="unknown")
        out_hammer = kernel(board, opts, archetype_id="hammer-pult")
    print(f">> forward OK: policy_logits={tuple(out_unknown['policy_logits'].shape)} "
          f"value={float(out_unknown['value'].reshape(-1)[0]):+.4f} "
          f"aux={tuple(out_unknown['aux_logits'].shape)}", flush=True)
    # unknown conditioning starts as a no-op (embedding zero-initialised).
    same = torch.allclose(out_unknown["policy_logits"], out_hammer["policy_logits"])
    print(f">> conditioning: unknown≈hammer at init (zero-embed) = {same}", flush=True)

    # 3) Save a core-kernel checkpoint.
    ckpt_path = paths.CHECKPOINTS_DIR / f"{args.run_name}_smoke.pt"
    kernel.save_core_kernel(ckpt_path, step=0)
    print(f">> saved core kernel → {ckpt_path} ({ckpt_path.stat().st_size/1e6:.2f} MB)", flush=True)

    # 4) Warm-start a specialist and verify trunk transfer + value kept + fold.
    reloaded = CoreKernel.load_core_kernel(ckpt_path, device=device, cfg=cfg)
    # Default: reinit policy+aux, KEEP value (research-driven for MCTS).
    spec = reloaded.warm_start_specialist("hammer-pult", fold_archetype=True)
    trunk_ok = torch.equal(
        spec.board_bag.weight.detach().cpu(),
        reloaded.net.board_bag.weight.detach().cpu(),
    )
    value_kept = torch.equal(
        spec.value_head[0].weight.detach().cpu(),
        reloaded.net.value_head[0].weight.detach().cpu(),
    )
    # cls_proj.bias got the (zero at init) archetype fold — equal when embed is zero:
    fold_ok = torch.allclose(
        spec.cls_proj.bias.detach().cpu(),
        reloaded.net.cls_proj.bias.detach().cpu(),
    )
    print(f">> warm-start specialist(hammer-pult): trunk_transfer={trunk_ok} "
          f"value_head_kept={value_kept} cls_bias_fold_consistent={fold_ok} "
          f"type={type(spec).__name__}", flush=True)

    # 5) Warm-start + persist a resumable specialist checkpoint (Phase-6 entry).
    ws = warm_start_specialist_from_checkpoint(
        ckpt_path, "hammer-pult",
        run_name=f"{args.run_name}_smoke_hammer-pult_bootstrap",
        device=device,
    )
    print(f">> wrote specialist warm-start ckpt: {ws.get('paths')}", flush=True)

    # 6) Optional: exercise the streaming corpus + one train batch on real data.
    jsonls = _resolve_jsonls(args) or StreamingArchetypeCorpus.discover_bucket_jsonls(
        args.bucket_dir, include_smoke=True
    )
    if jsonls:
        corpus = StreamingArchetypeCorpus(
            CorpusConfig(
                jsonl_paths=jsonls, val_frac=0.5, shuffle_buffer=4,
                max_sequences=4, verify_info_set=True, max_context=cfg.max_context,
            )
        )
        seqs = list(corpus.iter_sequences("all", shuffle=False))
        print(f">> streaming corpus: files={[p.name for p in jsonls]} "
              f"sampled_sequences={len(seqs)}", flush=True)
        if seqs:
            kernel.train()
            opt = torch.optim.AdamW(kernel.parameters(), lr=1e-3)
            from poke_bot.core_kernel import core_sequence_losses

            loss, m = core_sequence_losses(kernel, seqs[0], condition=True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            print(f">> one real train step: loss={float(loss.detach()):.4f} "
                  f"policy_acc={m.policy_acc:.2%} decisions={m.n_decisions} "
                  f"archetype={seqs[0].archetype!r}", flush=True)
    else:
        print(">> no bucket JSONL found — skipped real-data step (synthetic path OK).", flush=True)

    print("\n>> SMOKE OK: build + forward + conditioning + save + warm-start verified.",
          flush=True)
    return 0


def _build_train_setup(args):
    """Merge GPU profile + CLI overrides into (model_cfg, train_cfg, prefer_name, shuffle_buffer)."""
    prof_cfg, prof_train, prefer_name = gpu_profile(args.gpu_profile)

    def pick(cli, key, fallback):
        if cli is not None:
            return cli
        return prof_train.get(key, fallback)

    shuffle_buffer = pick(args.shuffle_buffer, "shuffle_buffer", 256)
    tcfg = CoreTrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        games_per_batch=pick(args.games_per_batch, "games_per_batch", 4),
        max_decisions_per_batch=pick(args.max_decisions_per_batch, "max_decisions_per_batch", 256),
        grad_accum_steps=pick(args.grad_accum_steps, "grad_accum_steps", 1),
        early_stop_patience=args.patience,
        value_loss_weight=(args.value_loss_weight
                           if args.value_loss_weight is not None
                           else prof_train.get("value_loss_weight", 1.5)),
        aux_loss_weight=(args.aux_loss_weight
                         if args.aux_loss_weight is not None
                         else prof_train.get("aux_loss_weight", 0.1)),
        opp_hand_loss_weight=(
            args.opp_hand_loss_weight
            if args.opp_hand_loss_weight is not None
            else 0.2
        ),
        opp_remainder_loss_weight=(
            args.opp_remainder_loss_weight
            if args.opp_remainder_loss_weight is not None
            else 0.15
        ),
        # Scope B strategy heads: never required on core_kernel (weights stay 0).
        lethal_threat_loss_weight=float(args.lethal_threat_loss_weight),
        prize_race_loss_weight=float(args.prize_race_loss_weight),
        amp=not args.no_amp,
        amp_dtype=(args.amp_dtype or prof_train.get("amp_dtype", "bf16")),
        seed=args.seed,
        condition_archetype=not args.no_condition,
        max_val_batches=args.max_val_batches,
    )
    return prof_cfg, tcfg, prefer_name, int(shuffle_buffer)


def _run_probe(args) -> int:
    """Short real throughput / peak-VRAM probe on the target GPU."""
    prof_cfg, tcfg, prefer_name, shuffle_buffer = _build_train_setup(args)
    device = _resolve_device(args.device, prefer_name=prefer_name)
    print(f"== core_kernel PROBE  device={device}  ({device_mod.describe()})", flush=True)
    if device.type != "cuda":
        print("WARN: probe not on CUDA — VRAM numbers only meaningful on GPU.", flush=True)
    cfg = prof_cfg or config.MODEL
    print(f">> profile={args.gpu_profile} d_model={cfg.d_model} "
          f"layers(s/t/o)={cfg.spatial_layers}/{cfg.temporal_layers}/{cfg.option_decoder_layers} "
          f"heads={cfg.n_heads} ff={cfg.ff_dim} max_context={cfg.max_context}", flush=True)
    _bp = config.batch_profile(device)
    print(f">> gpu_kind={config.gpu_kind(device)} "
          f"vram_target={_bp.vram_target_gb:g}/{_bp.vram_total_gb:g}GB "
          f"leaf_batch={config.leaf_batch_for_device(device)} "
          f"ram_cap={config.HARDWARE.ram_cache_gb:g}GB", flush=True)
    print(f">> train: gpb={tcfg.games_per_batch} maxdec={tcfg.max_decisions_per_batch} "
          f"accum={tcfg.grad_accum_steps} amp={tcfg.amp_dtype} "
          f"shuffle_buffer={shuffle_buffer}", flush=True)

    jsonls = _resolve_jsonls(args) or StreamingArchetypeCorpus.discover_bucket_jsonls(
        args.bucket_dir, include_smoke=True
    )
    if not jsonls:
        print("ERROR: no bucket JSONL found for probe.", file=sys.stderr)
        return 2

    n_seq_cap = max(tcfg.games_per_batch * (args.probe_steps + 2) * tcfg.grad_accum_steps, 8)
    corpus = StreamingArchetypeCorpus(
        CorpusConfig(
            jsonl_paths=jsonls, val_frac=0.0, shuffle_buffer=min(shuffle_buffer, 64),
            seed=args.seed, max_sequences=n_seq_cap,
            verify_info_set=not args.no_verify_info_set, max_context=cfg.max_context,
        )
    )

    kernel = CoreKernel(cfg=cfg, device=device)
    n_params = sum(p.numel() for p in kernel.parameters())
    optimizer_kwargs = dict(lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    if device.type == "cuda":
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(kernel.parameters(), **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(kernel.parameters(), **optimizer_kwargs)
    amp_dtype = resolve_amp_dtype(tcfg.amp_dtype)
    use_amp = bool(tcfg.amp and device.type == "cuda" and amp_dtype is not None)
    use_scaler = bool(use_amp and amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    accum = max(1, tcfg.grad_accum_steps)

    from poke_bot.core_kernel import _stream_batches  # local: batch grouping

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    kernel.train()

    steps = 0
    micro = 0
    dec_total = 0
    optimizer.zero_grad(set_to_none=True)
    # Warm-up one optimizer step (cuDNN/alloc) before timing.
    warmed = False
    t0 = time.time()
    done = False
    epoch = 0
    while not done:  # cycle the (small) corpus until probe_steps reached
        batches = _stream_batches(
            corpus.iter_sequences("all", epoch=epoch, shuffle=True),
            tcfg.games_per_batch, tcfg.max_decisions_per_batch,
        )
        any_batch = False
        for batch in batches:
            any_batch = True
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                total, metrics = core_batch_losses(
                    kernel,
                    batch,
                    value_weight=tcfg.value_loss_weight,
                    aux_weight=tcfg.aux_loss_weight,
                    opp_hand_weight=tcfg.opp_hand_loss_weight,
                    opp_remainder_weight=tcfg.opp_remainder_loss_weight,
                    condition=tcfg.condition_archetype,
                )
                dec_total += metrics.n_decisions
                if metrics.n_decisions <= 0:
                    continue
                total = total / accum
            scaler.scale(total).backward()
            micro += 1
            if micro % accum == 0:
                if tcfg.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(kernel.parameters(), tcfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                steps += 1
                if not warmed and device.type == "cuda":
                    torch.cuda.synchronize(device)
                    warmed = True
                    t0 = time.time()  # start timing after warm-up step
                    dec_total = 0
                if steps >= args.probe_steps + (1 if warmed else 0):
                    done = True
                    break
        epoch += 1
        if not any_batch:
            break
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    dt = time.time() - t0
    steps = max(steps - (1 if warmed else 0), 1)  # exclude warm-up step from rate

    peak_alloc = peak_reserved = 0.0
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated(device) / 1e9
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1e9

    sps = steps / dt if dt > 0 else 0.0
    dps = dec_total / dt if dt > 0 else 0.0
    print(f">> params={n_params/1e6:.1f}M  optim_steps={steps}  micro_batches={micro} "
          f"decisions={dec_total}", flush=True)
    print(f">> wall={dt:.1f}s  steps/s={sps:.2f}  decisions/s={dps:.0f}", flush=True)
    print(f">> PEAK VRAM: allocated={peak_alloc:.2f} GB  reserved={peak_reserved:.2f} GB "
          f"(card total ~12 GB)", flush=True)
    headroom = 12.0 - peak_reserved
    print(f">> headroom vs 12 GB: ~{headroom:.2f} GB  → "
          f"{'FITS comfortably' if headroom > 2 else 'TIGHT — reduce batch'}", flush=True)
    print(">> PROBE OK (no full run launched).", flush=True)
    return 0


def main(argv=None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    config.apply_runtime_perf()  # TF32 / cuDNN benchmark / thread pins.

    if args.smoke:
        return _run_smoke(args)

    if args.probe:
        return _run_probe(args)

    if args.warm_start:
        core_ckpt = args.core_ckpt
        if core_ckpt is None:
            from poke_bot import checkpoint as ckpt_mod

            best = ckpt_mod.best_path(args.run_name)
            latest = ckpt_mod.latest_path(args.run_name)
            core_ckpt = best if best.is_file() else latest
        if not Path(core_ckpt).is_file():
            print(f"ERROR: core kernel checkpoint not found: {core_ckpt}", file=sys.stderr)
            return 2
        device = _resolve_device(args.device)
        ws = warm_start_specialist_from_checkpoint(
            core_ckpt,
            args.warm_start,
            device=device,
            reinit_heads=_parse_reinit_heads(args.reinit_heads),
        )
        print(f">> warm-started {args.warm_start} from {core_ckpt}", flush=True)
        print(f">> specialist checkpoint: {ws.get('paths')}", flush=True)
        print(f">> now fine-tune with: scripts/train_bootstrap.py "
              f"--archetype {args.warm_start} --resume auto", flush=True)
        return 0

    prof_cfg, tcfg, prefer_name, shuffle_buffer = _build_train_setup(args)
    device = _resolve_device(args.device, prefer_name=prefer_name)
    jsonls = _resolve_jsonls(args)
    if not jsonls:
        print(f"ERROR: no bucket JSONL found under {args.bucket_dir} "
              f"(and none passed via --jsonl)", file=sys.stderr)
        return 2

    cfg = prof_cfg or config.MODEL
    _bp = config.batch_profile(device)
    print(f"== train_core_kernel  device={device}  ({device_mod.describe()})", flush=True)
    print(f">> gpu_profile={args.gpu_profile} gpu_kind={config.gpu_kind(device)} "
          f"vram_target={_bp.vram_target_gb:g}/{_bp.vram_total_gb:g}GB "
          f"ram_cap={config.HARDWARE.ram_cache_gb:g}GB "
          f"torch_threads={config.HARDWARE.torch_threads}", flush=True)
    print(f">> d_model={cfg.d_model} amp={tcfg.amp_dtype} gpb={tcfg.games_per_batch} "
          f"maxdec={tcfg.max_decisions_per_batch} accum={tcfg.grad_accum_steps} "
          f"shuffle_buffer={shuffle_buffer}", flush=True)
    print(f">> buckets ({len(jsonls)}): {[p.name for p in jsonls]}", flush=True)
    print(f">> run_name={args.run_name} resume={args.resume} "
          f"condition_archetype={not args.no_condition}", flush=True)

    corpus = StreamingArchetypeCorpus(
        CorpusConfig(
            jsonl_paths=jsonls,
            val_frac=args.val_frac,
            shuffle_buffer=shuffle_buffer,
            seed=args.seed,
            max_sequences=args.max_sequences,
            verify_info_set=not args.no_verify_info_set,
            max_context=cfg.max_context,
        )
    )
    result = train_core_kernel(
        corpus,
        run_name=args.run_name,
        train_cfg=tcfg,
        resume=args.resume,
        device=device,
        cfg=cfg,
    )
    out = paths.OUTPUTS_DIR / "train" / f"{args.run_name}_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f">> result → {out}", flush=True)
    print(f">> best={result.get('best_path')} metric={result.get('best_metric')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
