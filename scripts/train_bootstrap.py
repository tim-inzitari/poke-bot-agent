#!/usr/bin/env python
"""Supervised bootstrap training on per-archetype JSONL (Phase 3/4).

Default: pure Dragapult deck/archetype, Blackwell AMP, ``--resume auto``.

Example::

    /home/pokebot/miniconda3/envs/poke-bot-agent/bin/python scripts/train_bootstrap.py \\
        --jsonl data/bootstrap/dragapult.jsonl --archetype dragapult --resume auto
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import paths  # noqa: E402
from poke_bot.dataset import load_bootstrap_dataset  # noqa: E402
from poke_bot.train import TrainConfig, train_bootstrap  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _adapter_implementation_digest() -> str:
    """Bind training to the exact adapter/routing implementation snapshot."""

    relative_paths = (
        "poke_bot/dataset.py",
        "poke_bot/matchup_adapters.py",
        "poke_bot/matchup_adapter_activation.py",
        "poke_bot/model.py",
        "poke_bot/train.py",
        "scripts/train_bootstrap.py",
    )
    payload = {
        relative: _sha256(ROOT / relative) for relative in relative_paths
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="Bootstrap JSONL (default data/bootstrap/<archetype>.jsonl)",
    )
    p.add_argument(
        "--feature-manifest",
        type=Path,
        default=None,
        help="Validated compact feature-shard manifest (mutually exclusive with --jsonl).",
    )
    p.add_argument("--archetype", default="dragapult")
    p.add_argument("--run-name", default=None, help="Checkpoint run name")
    p.add_argument("--resume", default="auto", help="auto | 0 | path | best")
    p.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Weights-only initialization; optimizer/epoch state starts fresh.",
    )
    p.add_argument(
        "--model-profile",
        choices=("default", "pure-rl"),
        default="default",
        help="Use the exact small pure-RL architecture for a compatible hot start.",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--games-per-batch", type=int, default=4)
    p.add_argument("--max-decisions-per-batch", type=int, default=256)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument(
        "--split-by-episode",
        action="store_true",
        help="Keep both seats from an episode wholly in train or validation.",
    )
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--max-games", type=int, default=0, help="Cap games loaded (0=all)")
    p.add_argument(
        "--min-usable-record-frac",
        type=float,
        default=0.0,
        help="Fail before training if featurization keeps less than this fraction.",
    )
    p.add_argument(
        "--min-decisions",
        type=int,
        default=0,
        help="Fail before training below this many usable decisions.",
    )
    p.add_argument(
        "--aux-loss-weight",
        type=float,
        default=0.1,
        help="Archetype aux_head CE weight (default 0.1).",
    )
    p.add_argument(
        "--opp-hand-loss-weight",
        type=float,
        default=0.2,
        help="opp_hand_head multilabel BCE weight (default 0.2; masked if labels absent).",
    )
    p.add_argument(
        "--opp-remainder-loss-weight",
        type=float,
        default=0.15,
        help="opp_remainder_head multilabel BCE weight (default 0.15; masked if absent).",
    )
    p.add_argument(
        "--lethal-threat-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Scope B lethal_threat_head weight (default 0 — core/bootstrap off; "
            "enable on Blackwell Hammer specialist trains)."
        ),
    )
    p.add_argument(
        "--prize-race-loss-weight",
        type=float,
        default=0.0,
        help="Scope B prize_race_head weight (default 0 — core/bootstrap off).",
    )
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument(
        "--device-resident",
        action="store_true",
        help=(
            "Pack the complete hard-target stateless corpus onto the training "
            "GPU once and train without per-batch host transfers."
        ),
    )
    p.add_argument(
        "--matchup-adapter-training",
        action="store_true",
        help=(
            "Legacy bounded smoke path for oracle-ticketed matchup adapters. "
            "Requires --max-games 1..128. Production-scale fitting must use "
            "stage_matchup_adapter_corpus.py + train_matchup_adapters.py."
        ),
    )
    p.add_argument(
        "--matchup-adapter-activation-receipt",
        type=Path,
        default=None,
        help="Immutable post-iteration-15 activation receipt (required in adapter mode).",
    )
    p.add_argument(
        "--matchup-adapter-corpus-manifest",
        type=Path,
        default=None,
        help="Package/digest/archetype manifest for adapter corpus auditing.",
    )
    p.add_argument(
        "--active-gate-contract",
        type=Path,
        default=None,
        help="Active formal-gate contract whose packages/aliases must be excluded.",
    )
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    paths.ensure_runtime_dirs()
    if args.jsonl is not None and args.feature_manifest is not None:
        print("ERROR: --jsonl and --feature-manifest are mutually exclusive", file=sys.stderr)
        return 2
    if args.matchup_adapter_training:
        required = {
            "--init-checkpoint": args.init_checkpoint,
            "--matchup-adapter-activation-receipt": (
                args.matchup_adapter_activation_receipt
            ),
            "--matchup-adapter-corpus-manifest": (
                args.matchup_adapter_corpus_manifest
            ),
            "--active-gate-contract": args.active_gate_contract,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            print(
                "ERROR: adapter training is missing " + ", ".join(missing),
                file=sys.stderr,
            )
            return 2
        if args.feature_manifest is not None:
            print(
                "ERROR: adapter training requires raw exact-history JSONL; "
                "compact feature manifests cannot reconstruct public prefix masks",
                file=sys.stderr,
            )
            return 2
        if int(args.max_games) <= 0 or int(args.max_games) > 128:
            print(
                "ERROR: the legacy in-memory adapter path is smoke-only and "
                "requires --max-games in [1, 128]. Use "
                "scripts/stage_matchup_adapter_corpus.py followed by "
                "scripts/train_matchup_adapters.py for the real corpus.",
                file=sys.stderr,
            )
            return 2
    jsonl = args.jsonl or (paths.DATA_DIR / "bootstrap" / f"{args.archetype}.jsonl")
    feature_manifest = args.feature_manifest
    if feature_manifest is None and not jsonl.is_file():
        print(f"ERROR: missing JSONL {jsonl}", file=sys.stderr)
        return 2
    if feature_manifest is not None and not feature_manifest.is_file():
        print(f"ERROR: missing feature manifest {feature_manifest}", file=sys.stderr)
        return 2
    if args.matchup_adapter_training:
        from poke_bot.dataset import BootstrapDataset, iter_jsonl
        from poke_bot.matchup_adapter_activation import (
            assert_prepared_adapter_corpus_coverage,
            prepare_adapter_corpus_records,
        )
        from poke_bot.checkpoint import load_checkpoint

        corpus_manifest = json.loads(
            args.matchup_adapter_corpus_manifest.read_text(encoding="utf-8")
        )
        gate_contract = json.loads(
            args.active_gate_contract.read_text(encoding="utf-8")
        )
        parent_payload = load_checkpoint(args.init_checkpoint, map_location="cpu")
        parent_model_config = dict(parent_payload.get("model_config") or {})
        parent_max_context = int(parent_model_config.get("max_context") or 0)
        if parent_max_context <= 0:
            print(
                "ERROR: adapter parent lacks a valid temporal max_context",
                file=sys.stderr,
            )
            return 2
        # The explicit smoke cap is enforced above and applied before any
        # GameSequence objects are accumulated. Never iterate the full source
        # and truncate afterward: that recreates the prior host-memory failure.
        records = itertools.islice(iter_jsonl(jsonl), int(args.max_games))
        prepared = prepare_adapter_corpus_records(
            records,
            corpus_manifest=corpus_manifest,
            gate_contract=gate_contract,
            max_context=parent_max_context,
        )
        assert_prepared_adapter_corpus_coverage(prepared)
        ds = BootstrapDataset(
            prepared.sequences,
            jsonl_path=jsonl,
            conversion_stats={
                "records_total": prepared.total_records,
                "records_kept": len(prepared.sequences),
                "records_dropped": (
                    prepared.excluded_gate_records
                    + prepared.excluded_unsupported_records
                    + prepared.duplicate_records
                ),
                "excluded_gate_records": prepared.excluded_gate_records,
                "excluded_unsupported_records": (
                    prepared.excluded_unsupported_records
                ),
                "duplicate_records": prepared.duplicate_records,
                "adapter_active_positions": prepared.active_positions,
                "adapter_runtime_recognized_positions": (
                    prepared.runtime_recognized_positions
                ),
                "adapter_partition_sizes": {
                    key: len(value) for key, value in prepared.partitions.items()
                },
            },
        )
    elif feature_manifest is not None:
        compact_incompatible = (
            args.model_profile != "pure-rl"
            or float(args.aux_loss_weight) != 0.0
            or float(args.opp_hand_loss_weight) != 0.0
            or float(args.opp_remainder_loss_weight) != 0.0
            or float(args.lethal_threat_loss_weight) != 0.0
            or float(args.prize_race_loss_weight) != 0.0
        )
        if compact_incompatible:
            print(
                "ERROR: compact feature manifests require --model-profile pure-rl "
                "and all auxiliary loss weights set to zero",
                file=sys.stderr,
            )
            return 2

    run_name = args.run_name or f"{args.archetype}_bootstrap"
    print(f"== train_bootstrap archetype={args.archetype}", flush=True)
    print(
        f"   dataset={feature_manifest if feature_manifest is not None else jsonl}",
        flush=True,
    )
    print(f"   run_name={run_name} resume={args.resume}", flush=True)
    print(f"   device_resident={bool(args.device_resident)}", flush=True)

    if args.matchup_adapter_training:
        # ``ds`` was built above by the only path that derives public-prefix
        # masks and attaches package/gate audit tickets.  Never replace it with
        # the generic cached loader, which would erase both contracts.
        pass
    elif feature_manifest is not None:
        if args.max_games > 0:
            print("ERROR: --max-games is not supported with --feature-manifest", file=sys.stderr)
            return 2
        from poke_bot.feature_shards import load_feature_manifest

        ds = load_feature_manifest(feature_manifest, verify_hashes=True)
    else:
        ds = load_bootstrap_dataset(
            jsonl,
            max_games=args.max_games,
            use_cache=not args.no_cache,
            verify_info_set=True,
        )
    summary = ds.summary()
    print(f"   dataset: {json.dumps(summary)}", flush=True)
    if len(ds) == 0:
        print("ERROR: empty dataset", file=sys.stderr)
        return 1
    if not ds.info_set_ok_all:
        print("ERROR: info-set integrity failed in dataset", file=sys.stderr)
        return 1
    conversion = dict(summary.get("conversion") or {})
    records_total = int(conversion.get("records_total") or len(ds))
    usable_record_frac = len(ds) / max(1, records_total)
    if usable_record_frac < float(args.min_usable_record_frac):
        print(
            f"ERROR: usable record fraction {usable_record_frac:.4f} < "
            f"minimum {float(args.min_usable_record_frac):.4f}",
            file=sys.stderr,
        )
        return 1
    if ds.n_decisions < int(args.min_decisions):
        print(
            f"ERROR: usable decisions {ds.n_decisions} < "
            f"minimum {int(args.min_decisions)}",
            file=sys.stderr,
        )
        return 1

    cfg = TrainConfig(
        lr=args.lr,
        epochs=args.epochs,
        games_per_batch=args.games_per_batch,
        max_decisions_per_batch=args.max_decisions_per_batch,
        val_frac=args.val_frac,
        split_by_episode=(
            True if args.matchup_adapter_training else bool(args.split_by_episode)
        ),
        early_stop_patience=args.patience,
        aux_loss_weight=float(args.aux_loss_weight),
        opp_hand_loss_weight=float(args.opp_hand_loss_weight),
        opp_remainder_loss_weight=float(args.opp_remainder_loss_weight),
        lethal_threat_loss_weight=float(args.lethal_threat_loss_weight),
        prize_race_loss_weight=float(args.prize_race_loss_weight),
        amp=not args.no_amp,
        seed=args.seed,
        matchup_adapter_training=bool(args.matchup_adapter_training),
        matchup_adapter_activation_receipt=(
            str(args.matchup_adapter_activation_receipt.resolve())
            if args.matchup_adapter_activation_receipt is not None
            else ""
        ),
    )
    model_cfg = None
    checkpoint_extra: dict[str, object] = {}
    if args.matchup_adapter_training:
        assert args.matchup_adapter_corpus_manifest is not None
        assert args.active_gate_contract is not None
        checkpoint_extra["matchup_adapter_input_provenance"] = {
            "schema": "poke_bot.matchup_adapter_input_provenance/v1",
            "source_jsonl": str(jsonl.resolve()),
            "source_jsonl_digest": _sha256(jsonl),
            "corpus_manifest": str(
                args.matchup_adapter_corpus_manifest.resolve()
            ),
            "corpus_manifest_file_digest": _sha256(
                args.matchup_adapter_corpus_manifest
            ),
            "active_gate_contract": str(args.active_gate_contract.resolve()),
            "active_gate_contract_file_digest": _sha256(
                args.active_gate_contract
            ),
            "implementation_digest": _adapter_implementation_digest(),
        }
    if args.model_profile == "pure-rl":
        from poke_bot.pure_rl.model_profile import (
            model_config_dict,
            pure_rl_model_config,
        )

        model_cfg = pure_rl_model_config()
        provenance_path = feature_manifest if feature_manifest is not None else jsonl
        checkpoint_extra.update({
            "pure_rl": True,
            "smoke": False,
            "model_profile": model_config_dict(model_cfg),
            "top_ladder_hot_start": {
                "dataset": str(provenance_path.resolve()),
                "dataset_sha256": _sha256(provenance_path),
                "split_by_episode": bool(
                    args.split_by_episode or args.matchup_adapter_training
                ),
                "archetype": str(args.archetype),
            },
        })
    result = train_bootstrap(
        ds,
        run_name=run_name,
        archetype_id=args.archetype,
        train_cfg=cfg,
        resume=args.resume,
        # A weights-only hot start owns its serialized architecture.  Passing
        # today's profile here would reject historical pure-RL champions whose
        # layer geometry is intentionally preserved for bootstrap.
        model_cfg=None if args.init_checkpoint else model_cfg,
        init_checkpoint=args.init_checkpoint,
        checkpoint_extra=checkpoint_extra or None,
        device_resident=bool(args.device_resident),
    )
    out = paths.OUTPUTS_DIR / "train" / f"{run_name}_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Drop huge history tensors if any; history is JSON-safe already.
    out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(f">> result → {out}", flush=True)
    print(f">> best={result.get('best_path')} metric={result.get('best_metric')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
