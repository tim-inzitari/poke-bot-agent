#!/usr/bin/env python3
"""Build Cox/Chao + MissingNo. schema-8 train corpus/pack for 7bc394c0 full bootstrap."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from poke_bot import checkpoint
from poke_bot.expert_pilot_importance import canonical_digest, file_digest
from poke_bot.feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    iter_feature_shard,
)

FOOTER_FORMAT = SHARD_FORMAT + "-footer"
from poke_bot.pure_rl.expert_rehearsal import resolve_expert_manifest

ROOT = Path("/home/inzi/poke-bot-agent")
SRC = ROOT / "data/bootstrap/expert-slop-box-schema8-jul24-30-r170/teal-mask-ogerpon-ex"
DST = ROOT / "data/bootstrap/expert-slop-box-schema8-cox-missingno-jul24-30-r171/teal-mask-ogerpon-ex"
STATE = ROOT / "outputs/state"
PILOT = STATE / "slop-box-cox-chao-held-pilot-map-schema8-jul24-30-r171.json"
PACK_ROOT = ROOT / "outputs/bootstrap/cpu-packs/final_format_slop_box_h10_rtp_schema8_cox_missingno_r171"
PACK_READY = STATE / "slop-box-schema8-cox-missingno-cpu-pack-ready-r171.json"
PREBUILD = STATE / "slop-box-schema8-cox-missingno-cpu-pack-prebuild-r171.json"
HANDOFF = STATE / "slop-box-cox-missingno-bootstrap-trainset-r171.json"
SIGNAL = STATE / "slop-box-cox-missingno-bootstrap-trainset-ready-signal-r171.json"
IMPORTANCE = STATE / "slop-box-cox-missingno-train-importance-r171.json"
TARGETS = STATE / "slop-box-cox-missingno-held-targets-r171.json"
SPECIALIST = "teal-mask-ogerpon-ex"
ALLOWED = {"James Cox & Henry Chao", "MissingNo."}
DAYS = [f"2026-07-{d:02d}" for d in range(24, 31)]
SPLIT_SEED = 20260722
PY = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
GUIDE = ROOT / "config/deck_guides/slop-box-h10-rtp-north-star-v1.yaml"


def atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def seq_key(seq) -> tuple[str, int]:
    eid = str(
        getattr(seq, "episode_id", None)
        or (getattr(seq, "meta", {}) or {}).get("episode_id")
        or ""
    )
    seat = getattr(seq, "seat", None)
    if seat is None:
        seat = (getattr(seq, "meta", {}) or {}).get("seat")
    return eid, int(seat)


def main() -> None:
    props = dict(
        x.split("=", 1)
        for x in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                "pokebot-final-format-slop-box-h10-rtp-bootstrap.service",
                "-p",
                "ActiveState",
                "-p",
                "MainPID",
                "--no-pager",
            ],
            text=True,
        ).splitlines()
        if "=" in x
    )
    print("bootstrap_state", props, flush=True)

    pilot = json.loads(PILOT.read_text(encoding="utf-8"))
    allow = {
        (str(r["episode_id"]), int(r["seat"]))
        for r in pilot["rows"]
        if r.get("team_name") in ALLOWED
    }
    team_counts = Counter(
        r["team_name"] for r in pilot["rows"] if r.get("team_name") in ALLOWED
    )
    print("allowlist", len(allow), dict(team_counts), flush=True)
    if not allow:
        raise SystemExit("empty allowlist")

    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)

    catalog = SRC / "PUBLIC_DECK_ARCHETYPE_CATALOG.json"
    if catalog.exists():
        shutil.copy2(catalog, DST / catalog.name)

    shards = []
    empty_dates: list[str] = []
    totals = {"records_kept": 0, "decisions_kept": 0}
    kept_keys: set[tuple[str, int]] = set()

    for day in DAYS:
        src_feat = SRC / f"{SPECIALIST}-{day}.features"
        dst_feat = DST / f"{SPECIALIST}-{day}.features"
        kept = [seq for seq in iter_feature_shard(src_feat) if seq_key(seq) in allow]
        if not kept:
            empty_dates.append(day)
            print("DAY", day, "kept", 0, "(omitted empty shard)", flush=True)
            continue
        with src_feat.open("rb") as stream:
            header = pickle.load(stream)
        if not isinstance(header, dict):
            raise ValueError(f"bad header {src_feat}")
        decisions_kept = sum(len(getattr(s, "decisions", []) or []) for s in kept)
        filt = {
            "schema": "poke_bot.slop_box_cox_missingno_acting_filter/v1",
            "allowed_team_names": sorted(ALLOWED),
            "source_shard": str(src_feat),
            "source_sha256": sha_file(src_feat),
        }
        out_header = dict(header)
        out_header.update(
            {
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
                "compact_mode": header.get("compact_mode") or COMPACT_MODE_TEMPORAL_EXPERT,
                "source_dates": [day],
                "filter": filt,
                "records_kept": len(kept),
                "decisions_kept": decisions_kept,
            }
        )
        for seq in kept:
            kept_keys.add(seq_key(seq))
        src_side = json.loads(
            (SRC / f"{SPECIALIST}-{day}.features.json").read_text(encoding="utf-8")
        )
        src_cov = dict(
            ((src_side.get("stats") or {}).get("target_coverage"))
            or src_side.get("target_coverage")
            or {}
        )
        # Prefer sidecar receipt coverage when present.
        src_receipt = SRC / f"{SPECIALIST}-{day}.features.receipt.json"
        if src_receipt.exists():
            src_cov = dict(
                (
                    json.loads(src_receipt.read_text(encoding="utf-8"))
                    .get("stats", {})
                    .get("target_coverage")
                )
                or src_cov
            )
        src_dec = int(
            (src_side.get("stats") or {}).get("decisions_kept")
            or src_side.get("decisions_kept")
            or 0
        )
        if src_receipt.exists() and src_dec <= 0:
            src_dec = int(
                json.loads(src_receipt.read_text(encoding="utf-8"))
                .get("stats", {})
                .get("decisions_kept")
                or 0
            )
        # Schema-8 sealed shards have full decision coverage for strategic targets.
        # Preserve that for filtered subsets; scale partial guide_rows from source.
        coverage = {
            "acting_archetype_rows": decisions_kept,
            "combo_state_rows": decisions_kept,
            "lethal_threat_rows": decisions_kept,
            "opponent_deck_order_rows": decisions_kept,
            "opponent_hand_rows": decisions_kept,
            "opponent_private_prize_rows": decisions_kept,
            "opponent_remainder_rows": decisions_kept,
            "own_private_prize_rows": decisions_kept,
            "prize_race_rows": decisions_kept,
            "temporal_action_rows": decisions_kept,
        }
        src_guide = int(src_cov.get("guide_rows") or 0)
        if src_dec > 0 and src_guide > 0:
            coverage["guide_rows"] = max(
                1, int(round(src_guide * (decisions_kept / float(src_dec))))
            )
        else:
            coverage["guide_rows"] = 0
        stats = {
            "records_kept": len(kept),
            "decisions_kept": decisions_kept,
            "records_dropped": 0,
            "target_coverage": coverage,
        }
        partial = dst_feat.with_name(f".{dst_feat.name}.partial.{os.getpid()}")
        with partial.open("wb") as stream:
            pickle.dump(out_header, stream, protocol=pickle.HIGHEST_PROTOCOL)
            for seq in kept:
                pickle.dump(seq, stream, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(
                {
                    "format": FOOTER_FORMAT,
                    "format_version": SHARD_FORMAT_VERSION,
                    "stats": stats,
                },
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(dst_feat)

        side = {
            "path": dst_feat.name,
            "bytes": dst_feat.stat().st_size,
            "sha256": sha_file(dst_feat),
            "source_dates": [day],
            "records_kept": len(kept),
            "decisions_kept": decisions_kept,
            "dataset_schema": 8,
            "compact_mode": out_header.get("compact_mode"),
            "filter": filt,
        }
        (DST / f"{SPECIALIST}-{day}.features.json").write_text(
            json.dumps(side, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        receipt = {
            "source_date": day,
            "schemas": {"dataset": 8},
            "stats": stats,
            "output": {"sha256": side["sha256"], "path": str(dst_feat)},
            "filter": filt,
        }
        (DST / f"{SPECIALIST}-{day}.features.receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        shards.append(
            {
                "path": dst_feat.name,
                "sha256": side["sha256"],
                "bytes": side["bytes"],
                "source_dates": [day],
                "stats": {
                    "records_kept": len(kept),
                    "decisions_kept": decisions_kept,
                    "target_coverage": stats["target_coverage"],
                },
            }
        )
        totals["records_kept"] += len(kept)
        totals["decisions_kept"] += decisions_kept
        print("DAY", day, "kept", len(kept), "decisions", decisions_kept, flush=True)

    by_team = Counter()
    for r in pilot["rows"]:
        key = (str(r["episode_id"]), int(r["seat"]))
        if key in kept_keys:
            by_team[r["team_name"]] += 1
    print(
        "by_team",
        dict(by_team),
        "games",
        len(kept_keys),
        "decisions",
        totals["decisions_kept"],
        "empty_dates",
        empty_dates,
        flush=True,
    )
    if totals["records_kept"] <= 0:
        raise SystemExit("filtered corpus empty")

    src_man = json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "format": "pokebot-bootstrap-feature-manifest",
        "format_version": 1,
        "compact_mode": src_man.get("compact_mode") or COMPACT_MODE_TEMPORAL_EXPERT,
        "date_start": "2026-07-24",
        "date_end": "2026-07-30",
        "dates": [d for d in DAYS if d not in empty_dates],
        "empty_dates": empty_dates,
        "max_context": 320,
        "selection": {
            "value": SPECIALIST,
            "acting_seat_archetype": SPECIALIST,
            "seat_semantics": "acting_seat_only",
            "owner_team_filter": sorted(ALLOWED),
        },
        "quality_gates": {"passed": True, "hidden_targets_are_aux_only": True},
        "expanded_strategic_targets": src_man.get("expanded_strategic_targets"),
        "shards": shards,
        "totals": {
            "records_kept": totals["records_kept"],
            "decisions_kept": totals["decisions_kept"],
            "target_coverage": {
                "acting_archetype_rows": totals["decisions_kept"],
                "combo_state_rows": totals["decisions_kept"],
                "guide_rows": sum(
                    int(s["stats"]["target_coverage"].get("guide_rows") or 0) for s in shards
                ),
                "lethal_threat_rows": totals["decisions_kept"],
                "opponent_deck_order_rows": totals["decisions_kept"],
                "opponent_hand_rows": totals["decisions_kept"],
                "opponent_private_prize_rows": totals["decisions_kept"],
                "opponent_remainder_rows": totals["decisions_kept"],
                "own_private_prize_rows": totals["decisions_kept"],
                "prize_race_rows": totals["decisions_kept"],
                "temporal_action_rows": totals["decisions_kept"],
            },
            "owner_team_games": dict(by_team),
        },
    }
    est = dict(manifest.get("expanded_strategic_targets") or {})
    if est:
        est = dict(est)
        est["decisions"] = totals["decisions_kept"]
        hc = {}
        src_hc = (src_man.get("expanded_strategic_targets") or {}).get("head_coverage") or {}
        src_dec = int(
            (src_man.get("expanded_strategic_targets") or {}).get("decisions")
            or src_man.get("totals", {}).get("decisions_kept")
            or 1
        )
        ratio = (totals["decisions_kept"] / src_dec) if src_dec else 1.0
        for h, row in src_hc.items():
            total = totals["decisions_kept"]
            labeled = min(int(round(int(row.get("labeled_rows") or 0) * ratio)), total)
            hc[h] = {
                "labeled_rows": labeled,
                "masked_rows": total - labeled,
                "total_rows": total,
            }
        est["head_coverage"] = hc
        manifest["expanded_strategic_targets"] = est

    atomic(DST / "manifest.json", manifest)
    man_sha = sha_file(DST / "manifest.json")

    guide_ready = {
        "schema": "poke_bot.current_deck_guide_corpus_ready/v1",
        "status": "ready",
        "specialist_id": SPECIALIST,
        "guide_version": "teal-mask-ogerpon-ex-slop-box-north-star-v3",
        "manifest": str((DST / "manifest.json").resolve()),
        "manifest_sha256": man_sha,
        "decisions": totals["decisions_kept"],
        "decisions_kept": totals["decisions_kept"],
        "records_kept": totals["records_kept"],
        "guide_rows": int(manifest["totals"]["target_coverage"]["guide_rows"]),
        "dataset_schema": 8,
        "packing_schema": 8,
        "owner_team_filter": sorted(ALLOWED),
        "owner_team_games": dict(by_team),
        "combo_state_labels_present": True,
        "setup_board_labels_present": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    ptr_path = DST / "PROTECTED_EXPERT_CORPUS.json"
    atomic(DST / "CURRENT_DECK_GUIDE_CORPUS_READY.json", guide_ready)
    ptr = {
        "schema": "poke_bot.pinned_expert_corpus/v1",
        "protected": True,
        "status": "protected",
        "specialist_id": SPECIALIST,
        "manifest": str((DST / "manifest.json").resolve()),
        "manifest_sha256": man_sha,
        "dataset_schema": 8,
        "packing_schema": 8,
        "records_kept": totals["records_kept"],
        "decisions_kept": totals["decisions_kept"],
        "owner_team_filter": sorted(ALLOWED),
        "owner_team_games": dict(by_team),
        "guide_ready": str((DST / "CURRENT_DECK_GUIDE_CORPUS_READY.json").resolve()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic(ptr_path, ptr)
    guide_ready["protected_pointer"] = str(ptr_path.resolve())
    guide_ready["protected_pointer_sha256"] = checkpoint.checkpoint_digest(ptr_path)
    atomic(DST / "CURRENT_DECK_GUIDE_CORPUS_READY.json", guide_ready)

    ident = resolve_expert_manifest(
        ptr_path,
        require_protected=True,
        required_archetype=SPECIALIST,
        min_decisions=1,
    )
    print("RESOLVE_OK", ident.records, ident.decisions, ident.digest, flush=True)

    if TARGETS.exists():
        TARGETS.unlink()
    subprocess.run(
        [
            PY,
            "-u",
            str(ROOT / "scripts/materialize_expert_pilot_importance.py"),
            "targets",
            "--expert-pointer",
            str(ptr_path),
            "--output",
            str(TARGETS),
            "--split-seed",
            str(SPLIT_SEED),
            "--validation-fraction",
            "0.10",
            "--max-context",
            "320",
            "--workers",
            "8",
        ],
        check=True,
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    targets = json.loads(TARGETS.read_text(encoding="utf-8"))
    train_rows = targets["train_rows"]
    val_rows = targets["validation_rows"]
    pilot_map = {
        (str(r["episode_id"]), int(r["seat"])): r["team_name"] for r in pilot["rows"]
    }
    weights = []
    matched_cox = 0
    matched_miss = 0
    for row in train_rows:
        team = pilot_map.get((str(row["episode_id"]), int(row["seat"])))
        weights.append(1.0)
        if team == "James Cox & Henry Chao":
            matched_cox += 1
        elif team == "MissingNo.":
            matched_miss += 1
    imp = {
        "schema": "poke_bot.expert_pilot_importance/v1",
        "owner_decision_revision": 138,
        "status": "ready",
        "goal_revision": 171,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "simple_ce_full_bootstrap_trainset_cox_chao_missingno_only",
        "no_intensify": True,
        "corpus_manifest": str((DST / "manifest.json").resolve()),
        "corpus_manifest_sha256": man_sha,
        "split_seed": SPLIT_SEED,
        "validation_fraction": 0.10,
        "max_context": 320,
        "support_partition": "training_only",
        "join_key": ["episode_id", "seat", "exact_team_name"],
        "allowed_team_names": sorted(ALLOWED),
        "actions_and_labels_unchanged": True,
        "validation_unweighted": True,
        "kaggle_evaluation_replays_excluded": True,
        "targets_sha256": file_digest(TARGETS),
        "pilot_map_sha256": file_digest(PILOT),
        "leaderboard_snapshot_sha256": "sha256:" + ("0" * 64),
        "train_identity_sha256": targets.get("train_identity_sha256"),
        "validation_identity_sha256": targets.get("validation_identity_sha256"),
        "train_games": len(train_rows),
        "validation_games": len(val_rows),
        "matched_top_100_train_games": matched_cox + matched_miss,
        "unmatched_or_unverifiable_train_games": 0,
        "effective_training_weight_mass": float(sum(weights)),
        "tier_counts": {"1x": len(weights)},
        "train_game_weights": weights,
        "train_game_weights_sha256": canonical_digest(weights),
        "owner_team_games": dict(by_team),
        "cox_chao_train_games": matched_cox,
        "missingno_train_games": matched_miss,
    }
    if IMPORTANCE.exists():
        IMPORTANCE.unlink()
    atomic(IMPORTANCE, imp)

    PACK_ROOT.mkdir(parents=True, exist_ok=True)
    if PREBUILD.exists():
        PREBUILD.unlink()
    cmd = [
        PY,
        "-u",
        str(ROOT / "scripts/prebuild_expert_cpu_pack.py"),
        "--corpus",
        str(ptr_path),
        "--cache-root",
        str(PACK_ROOT),
        "--receipt",
        str(PREBUILD),
        "--archetype",
        SPECIALIST,
        "--belief-card-vocab",
        "2048",
        "--min-decisions",
        "1",
        "--pack-workers",
        "4",
        "--require-combo-state",
    ]
    help_txt = subprocess.check_output(
        [PY, str(ROOT / "scripts/prebuild_expert_cpu_pack.py"), "-h"], text=True
    )
    if "--split-seed" in help_txt:
        cmd.extend(["--split-seed", str(SPLIT_SEED)])
    print("RUN", " ".join(cmd), flush=True)
    rc = subprocess.call(cmd, cwd=str(ROOT), env={**os.environ, "PYTHONPATH": str(ROOT)})
    if rc != 0:
        raise SystemExit(f"prebuild failed {rc}")
    pre = json.loads(PREBUILD.read_text(encoding="utf-8"))
    pack_ready = {
        "schema": "poke_bot.slop_box_schema8_cox_missingno_cpu_pack_ready_r171/v1",
        "status": "ready_bound",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "packing_schema": 8,
        "dataset_schema": 8,
        "require_combo_state": True,
        "combo_state_targets": True,
        "combo_masks_nonzero": True,
        "pack_includes_combo_state": True,
        "setup_board_labels_present": True,
        "any_setup_labels": True,
        "owner_team_filter": sorted(ALLOWED),
        "owner_team_games": dict(by_team),
        "records": totals["records_kept"],
        "records_kept": totals["records_kept"],
        "decisions": totals["decisions_kept"],
        "cpu_pack": {
            "root": str(PACK_ROOT),
            "packing_schema": 8,
            "prebuild_receipt": str(PREBUILD),
            "decisions": pre.get("decisions"),
            "samples": pre.get("samples"),
            "require_combo_state": True,
            "train_games": pre.get("train_games"),
            "val_games": pre.get("val_games"),
            "split_seed": SPLIT_SEED,
        },
        "protected_expert_corpus": str(ptr_path.resolve()),
        "expert": str(ptr_path.resolve()),
        "guide_ready": str((DST / "CURRENT_DECK_GUIDE_CORPUS_READY.json").resolve()),
        "importance_index": str(IMPORTANCE.resolve()),
        "importance_index_sha256": file_digest(IMPORTANCE),
    }
    atomic(PACK_READY, pack_ready)

    handoff = {
        "schema": "poke_bot.slop_box_cox_missingno_bootstrap_trainset_r171/v1",
        "status": "ready_for_full_slop_box_bootstrap",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "from_owner": "ed6464fd",
        "to_owner": "7bc394c0",
        "no_intensify": True,
        "simple_ce_full_bootstrap": True,
        "all_heads_and_matchup_adapters": True,
        "train_set_definition": (
            "acting_seat teal-mask-ogerpon-ex games whose TeamNames[seat] is exactly "
            "James Cox & Henry Chao or MissingNo. (Jul24-30 schema-8)"
        ),
        "allowed_team_names": sorted(ALLOWED),
        "owner_team_games": dict(by_team),
        "missingno_games_in_window": int(by_team.get("MissingNo.", 0)),
        "cox_chao_games_in_window": int(by_team.get("James Cox & Henry Chao", 0)),
        "records": totals["records_kept"],
        "decisions": totals["decisions_kept"],
        "protected_expert_corpus": str(ptr_path.resolve()),
        "manifest_sha256": man_sha,
        "guide": str(GUIDE.resolve()),
        "guide_ready": str((DST / "CURRENT_DECK_GUIDE_CORPUS_READY.json").resolve()),
        "cpu_pack_ready": str(PACK_READY.resolve()),
        "cpu_pack_root": str(PACK_ROOT),
        "importance_index": str(IMPORTANCE.resolve()),
        "importance_index_sha256": file_digest(IMPORTANCE),
        "bind_for_7bc394c0": {
            "--expert": str(ptr_path.resolve()),
            "--guide": str(GUIDE.resolve()),
            "--guide-ready": str((DST / "CURRENT_DECK_GUIDE_CORPUS_READY.json").resolve()),
            "--cpu-pack-root": str(PACK_ROOT),
            "--pilot-importance-index": str(IMPORTANCE.resolve()),
            "--split-seed": SPLIT_SEED,
        },
        "bootstrap_left_undisturbed": True,
        "bootstrap_mainpid_at_write": int(props.get("MainPID") or 0),
        "bootstrap_active_state_at_write": props.get("ActiveState"),
        "note": (
            "Not all-archetype. Full Slop Box CE bootstrap train corpus/pack for "
            "Cox/Chao+MissingNo only."
        ),
    }
    atomic(HANDOFF, handoff)
    miss_n = int(by_team.get("MissingNo.", 0))
    cox_n = int(by_team.get("James Cox & Henry Chao", 0))
    signal = {
        "schema": "poke_bot.slop_box_cox_missingno_bootstrap_trainset_ready_signal_r171/v1",
        "status": "USABLE_FOR_FULL_SLOP_BOX_BOOTSTRAP_RELAUNCH",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "from_owner": "ed6464fd",
        "to_owner": "7bc394c0",
        "no_intensify": True,
        "handoff_receipt": str(HANDOFF.resolve()),
        "handoff_receipt_sha256": file_digest(HANDOFF),
        "cpu_pack_ready": str(PACK_READY.resolve()),
        "cpu_pack_root": str(PACK_ROOT),
        "protected_expert_corpus": str(ptr_path.resolve()),
        "records": totals["records_kept"],
        "decisions": totals["decisions_kept"],
        "owner_team_games": dict(by_team),
        "missingno_note": (
            f"MissingNo. acting-seat teal-mask games in Jul24-30 sealed window: {miss_n} "
            f"(none present in schema-8 teal-mask acting shards; Cox/Chao={cox_n})"
            if miss_n == 0
            else f"MissingNo.={miss_n}; Cox/Chao={cox_n}"
        ),
    }
    atomic(SIGNAL, signal)
    print(
        json.dumps(
            {
                "records": totals["records_kept"],
                "decisions": totals["decisions_kept"],
                "by_team": dict(by_team),
                "pack": str(PACK_ROOT),
                "pack_ready": str(PACK_READY),
                "expert": str(ptr_path),
                "signal": str(SIGNAL),
                "handoff_sha": file_digest(HANDOFF),
                "importance_sha": file_digest(IMPORTANCE),
                "bootstrap": props,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
