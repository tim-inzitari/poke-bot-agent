#!/usr/bin/env python3
"""Run Chao-hard CE continue for Slop Box H10 (r170).

Trains on the FULL expanded teal-mask / Slop Box acting-seat expert corpus
with Chao×N (+ optional James Cox×M) importance upweight — never a Chao-only
subset. Hot-starts from best Chao-held deep-CE checkpoint when available.
Does not publish ready/RL. Gate remains exact \"James Cox & Henry Chao\" held.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import checkpoint  # noqa: E402
from scripts.run_slop_box_h10_rtp_bootstrap import (  # noqa: E402
    IDENTITY_SCHEMA,
    SPECIALIST_ID,
    read_json,
    sha256,
)
from scripts.run_starmie_expert_bootstrap import (  # noqa: E402
    TARGETS,
    load_expanded_head_contract,
    main as bootstrap_main,
)
from scripts.validate_future_specialist_strategic_curriculum import (  # noqa: E402
    materialize,
)

DEEP_CE_RUN = Path(
    "/home/inzi/poke-bot-agent/outputs/bootstrap/final_format_slop_box_h10_rtp"
)
DECISION_FUSION_V2 = "poke_bot.causal_decision_fusion/v2"
DECISION_FUSION_V3 = "poke_bot.causal_decision_fusion/v3"


def _fusion_schema(payload: dict) -> str | None:
    inv = payload.get("decision_fusion_inventory")
    if isinstance(inv, dict) and inv.get("schema"):
        return str(inv.get("schema"))
    provenance = payload.get("provenance") or {}
    if isinstance(provenance, dict):
        fusion = provenance.get("decision_fusion") or {}
        if isinstance(fusion, dict) and fusion.get("schema"):
            return str(fusion.get("schema"))
    fusion = payload.get("decision_fusion")
    if isinstance(fusion, dict) and fusion.get("schema"):
        return str(fusion.get("schema"))
    for key in ("model_config", "config"):
        cfg = payload.get(key) or {}
        if isinstance(cfg, dict) and cfg.get("decision_fusion_schema"):
            return str(cfg.get("decision_fusion_schema"))
    return None


def _fusion_enabled(payload: dict) -> bool:
    for key in ("model_config", "config"):
        cfg = payload.get(key) or {}
        if isinstance(cfg, dict) and cfg.get("decision_fusion_enabled") is True:
            return True
    provenance = payload.get("provenance") or {}
    if isinstance(provenance, dict):
        fusion = provenance.get("decision_fusion") or {}
        if isinstance(fusion, dict) and fusion.get("runtime_enabled") is True:
            return True
    return False


def _is_fusion_valid(payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False
    schema = _fusion_schema(payload)
    if schema in {DECISION_FUSION_V2, DECISION_FUSION_V3} and _fusion_enabled(payload):
        return True
    state_dict = payload.get("model_state_dict")
    if isinstance(state_dict, dict) and any(
        k.startswith("decision_fusion.") for k in state_dict
    ):
        return True
    return False


def resolve_hot_start(explicit: Path | None) -> Path:
    if explicit is not None and explicit.is_file():
        return explicit.expanduser().resolve()

    # Prefer an explicit Chao-held selection receipt when present.
    select_receipt = Path(
        "/home/inzi/poke-bot-agent/outputs/state/"
        "slop-box-chao-hard-held-select-prestart-r170.json"
    )
    if select_receipt.is_file():
        try:
            selected = read_json(select_receipt)
            for key in (
                "selected_checkpoint",
                "checkpoint",
                "hot_start_recommendation",
            ):
                value = selected.get(key)
                if isinstance(value, dict):
                    value = value.get("checkpoint") or value.get("path")
                if isinstance(value, str) and Path(value).is_file():
                    payload = checkpoint.load_checkpoint(value, map_location="cpu")
                    if _is_fusion_valid(payload if isinstance(payload, dict) else {}):
                        return Path(value).resolve()
        except Exception:
            pass

    state_path = DEEP_CE_RUN / "state.json"
    cks = DEEP_CE_RUN / "checkpoints"
    candidates: list[Path] = []
    if state_path.is_file():
        state = read_json(state_path)
        best = state.get("best_path")
        if isinstance(best, str) and Path(best).is_file():
            candidates.append(Path(best))
    if cks.is_dir():
        epochs = sorted(
            cks.glob("epoch_*.pt"),
            key=lambda p: int(p.stem.split("_")[1]),
        )
        # Prefer later deep-CE epochs (fusion trained) over early val-loss best.
        candidates.extend(reversed(epochs))
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            payload = checkpoint.load_checkpoint(path, map_location="cpu")
        except Exception:
            continue
        if _is_fusion_valid(payload if isinstance(payload, dict) else {}):
            return path.resolve()
    raise SystemExit(
        "no fusion-valid deep-CE hot-start checkpoint found under "
        f"{DEEP_CE_RUN}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert", type=Path, required=True)
    parser.add_argument("--guide-ready", type=Path, required=True)
    parser.add_argument("--pilot-importance-index", type=Path, required=True)
    parser.add_argument(
        "--hot-start",
        type=Path,
        default=None,
        help="Optional explicit hot-start; default resolves fusion-valid deep CE ckpt",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ready", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument(
        "--select-chao-held-after",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After CE, sweep run_dir checkpoints by Chao held and write receipt",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--identity",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/state/slop_box_h10_rtp_prestage_identity_r170.json"
        ),
    )
    parser.add_argument(
        "--guide",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/config/deck_guides/slop-box-h10-rtp-north-star-v1.yaml"
        ),
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path("/home/inzi/poke-bot-agent/config/rl_protocol.yaml"),
    )
    parser.add_argument(
        "--registry-root",
        type=Path,
        default=Path("/home/inzi/poke-bot-agent/outputs/pure_rl/_protected/models"),
    )
    parser.add_argument(
        "--cpu-pack-root",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/bootstrap/cpu-packs/"
            "final_format_slop_box_chao_hard_r170"
        ),
    )
    parser.add_argument(
        "--curriculum-root",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "final_format_slop_box_chao_hard_curriculum_r170"
        ),
    )
    parser.add_argument(
        "--family",
        default="final-format-slop-box-chao-hard-ce-r170",
    )
    args = parser.parse_args()

    expert = args.expert.expanduser().resolve()
    guide_ready = args.guide_ready.expanduser().resolve()
    guide = args.guide.expanduser().resolve()
    pilot = args.pilot_importance_index.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    ready = args.ready.expanduser().resolve()
    identity_path = args.identity.expanduser().resolve()
    hot_start = resolve_hot_start(
        args.hot_start.expanduser().resolve() if args.hot_start else None
    )

    for path in (expert, guide_ready, guide, hot_start, pilot, identity_path):
        if not path.is_file():
            raise SystemExit(f"missing required path: {path}")

    identity = read_json(identity_path)
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise SystemExit("Slop Box identity schema mismatch")

    # Fail closed: never mark production ready from Chao-hard until gate script.
    if ready.is_file():
        ready.unlink()

    curriculum = materialize(
        specialist_id=SPECIALIST_ID,
        guide_contract=guide,
        guide_ready_receipt=guide_ready,
        output_root=args.curriculum_root.resolve(),
        training_implementation=(ROOT / "poke_bot" / "train.py"),
        training_mode="strategic_directional_v2",
        include_combo_state=True,
    )
    _expanded_raw, expanded = load_expanded_head_contract(args.protocol.resolve())
    del _expanded_raw

    parent_family = args.registry_root.resolve() / "marnie-iter9-training-freeze-r163"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    log_dir = Path(
        "/home/inzi/poke-bot-agent/outputs/final_format_slop_box_chao_hard_r170/logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    seed = run_dir / "checkpoints" / "seed.pt"
    if seed.exists() or seed.is_symlink():
        seed.unlink()
    os.symlink(hot_start, seed)

    staging = {
        "schema": "poke_bot.slop_box_chao_hard_ce_run_r170/v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "starting",
        "gate_team": "James Cox & Henry Chao",
        "no_ready": True,
        "no_rl": True,
        "hot_start": str(hot_start),
        "hot_start_sha256": sha256(hot_start),
        "expert": str(expert),
        "pilot_importance_index": str(pilot),
        "pilot_importance_index_sha256": sha256(pilot),
        "epochs": int(args.epochs),
        "run_dir": str(run_dir),
    }
    receipt = Path(
        "/home/inzi/poke-bot-agent/outputs/state/slop-box-chao-hard-ce-run-r170.json"
    )
    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(staging, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(receipt)

    argv = [
        "--expert-corpus",
        str(expert),
        "--archetype",
        SPECIALIST_ID,
        "--family",
        args.family,
        "--display-name",
        "Slop Box Chao-hard CE Continue r170",
        "--core-family",
        str(parent_family),
        "--registry-root",
        str(args.registry_root.resolve()),
        "--ready",
        str(ready),
        "--run-name",
        "final_format_slop_box_chao_hard_r170",
        "--run-dir",
        str(run_dir),
        "--epochs",
        str(args.epochs),
        "--owner-slop-box-ce-intensify-epochs",
        str(args.epochs),
        "--pilot-importance-index",
        str(pilot),
        "--batch-size",
        str(args.batch_size),
        "--min-decisions",
        "90000",
        "--cpu-pack-root",
        str(args.cpu_pack_root.resolve()),
        "--expanded-heads",
        "--decision-fusion",
        "--allow-h10-specialist-parent",
        "--retain-inherited-h10-combo-state-head",
        "--rl-protocol",
        str(args.protocol.resolve()),
        "--expected-expanded-schedule-digest",
        str(expanded["schedule_digest"]),
        "--expected-expanded-target-digest",
        str(expanded["target_schema_digest"]),
        "--current-deck-guide-contract",
        str(guide),
        "--expected-current-deck-guide-sha256",
        sha256(guide),
        "--current-deck-guide-version",
        "teal-mask-ogerpon-ex-slop-box-north-star-v3",
        "--current-deck-guide-corpus-ready",
        str(guide_ready),
        "--expected-current-deck-guide-corpus-ready-sha256",
        sha256(guide_ready),
        "--strategic-curriculum-spec",
        str(curriculum["curriculum_spec"]),
        "--strategic-curriculum-spec-sha256",
        sha256(curriculum["curriculum_spec"]),
        "--strategic-head-role-map",
        str(curriculum["head_role_map"]),
        "--strategic-head-role-map-sha256",
        sha256(curriculum["head_role_map"]),
        "--strategic-validation-receipt",
        str(curriculum["validation_receipt"]),
        "--strategic-validation-receipt-sha256",
        sha256(curriculum["validation_receipt"]),
    ]
    for target in TARGETS:
        argv.extend(("--required-target", target))

    print(
        "[chao-hard] starting CE continue",
        json.dumps(
            {
                "hot_start": str(hot_start),
                "epochs": args.epochs,
                "pilot": str(pilot),
                "run_dir": str(run_dir),
            }
        ),
        flush=True,
    )
    rc = int(bootstrap_main(argv))

    staging["status"] = "completed" if rc == 0 else "failed"
    staging["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    staging["exit_code"] = rc

    if ready.is_file():
        quarantine = ready.with_name(ready.name + ".chao-hard-pre-gate-quarantine")
        shutil.move(str(ready), str(quarantine))
        print(f"[chao-hard] quarantined ready marker -> {quarantine}", flush=True)

    if args.select_chao_held_after and (run_dir / "checkpoints").is_dir():
        select_out = Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-chao-hard-held-select-r170.json"
        )
        select_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "select_slop_box_chao_held_checkpoint_r170.py"),
            "--checkpoint-dir",
            str(run_dir / "checkpoints"),
            "--epochs",
            "all",
            "--expert-pointer",
            str(expert),
            "--targets",
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-cox-chao-held-targets-expanded-r170.json",
            "--pilot-map",
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-cox-chao-held-pilot-map-expanded-r170.json",
            "--held-split",
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-cox-chao-held-split-pilot-map-expanded-r170.json",
            "--cpu-pack-root",
            str(args.cpu_pack_root.resolve()),
            "--output",
            str(select_out),
        ]
        print("[chao-hard] selecting best Chao-held checkpoint", flush=True)
        select_rc = int(subprocess.run(select_cmd, check=False).returncode)
        staging["chao_held_select_rc"] = select_rc
        staging["chao_held_select_receipt"] = str(select_out)
        if select_out.is_file():
            staging["chao_held_select"] = json.loads(
                select_out.read_text(encoding="utf-8")
            ).get("hot_start_recommendation")

    tmp = receipt.with_suffix(".tmp")
    tmp.write_text(json.dumps(staging, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(receipt)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
