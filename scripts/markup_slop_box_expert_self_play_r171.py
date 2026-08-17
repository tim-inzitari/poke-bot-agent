#!/usr/bin/env python3
"""Self-play-equivalent matchup-ticket markup for Slop Box expert corpus (r171)."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from poke_bot.feature_shards import (
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    iter_feature_shard,
)
from poke_bot.matchup_adapter_activation import (
    TRAINING_TICKET_SCHEMA,
    adapter_training_ticket,
)
from poke_bot.matchup_adapters import UNKNOWN_ROUTE, route_for_archetype
from poke_bot.pure_rl.model_registry import sha256 as file_sha256

ROOT = Path("/home/pokebot/poke-bot-agent")
OUT = Path(
    "/home/pokebot/poke-bot-agent/outputs/bootstrap/"
    "slop-box-expert-self-play-equivalent-markup-r171"
)
EXPERT = ROOT / (
    "data/bootstrap/expert-slop-box-teal-mask-full41-r170/teal-mask-ogerpon-ex"
)
TRAJ = ROOT / "outputs/bootstrap/slop-box-h10-rtp/expert_trajectory_shard.jsonl"
GATE = Path(
    "/home/pokebot/poke-bot-agent-deployments/final-format-marnie-postupload-r136/"
    "runtime/final_format_slop_box_gate_r172_crustle_h10_s.json"
)
SPECIALIST = "teal-mask-ogerpon-ex"


def canonical_digest(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def package_digest_for_expert(
    *, archive_sha: str, episode_id: str, seat: int, opp: str
) -> str:
    raw = f"{archive_sha}|{episode_id}|{seat}|{opp}".encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def main() -> None:
    gate = json.loads(GATE.read_text(encoding="utf-8"))
    gate_id = str(gate.get("active_gate_id") or gate.get("id") or "")
    gate_digest = file_sha256(GATE)
    man = json.loads((EXPERT / "manifest.json").read_text(encoding="utf-8"))
    expert_manifest_digest = file_sha256(EXPERT / "manifest.json")
    feat_out = OUT / "features"
    feat_out.mkdir(parents=True, exist_ok=True)

    route_games: Counter[str] = Counter()
    route_decs: Counter[str] = Counter()
    ticketed_games = 0
    ticketed_decs = 0
    skipped_unknown = 0
    skipped_unknown_decs = 0
    shard_rows: list[dict[str, object]] = []

    for shard in man["shards"]:
        src = EXPERT / shard["path"]
        with src.open("rb") as fh:
            header = pickle.load(fh)
        archive_sha = str(header.get("source_archive_sha256") or "")
        out_path = feat_out / Path(shard["path"]).name.replace(
            ".features", ".ticketed.features"
        )
        n_seq = 0
        n_dec = 0
        with out_path.open("wb") as out_fh:
            new_header = dict(header)
            new_header["self_play_equivalent_markup"] = {
                "schema": "poke_bot.slop_box_expert_self_play_equivalent_markup/v1",
                "specialist_id": SPECIALIST,
                "source_features": str(src),
                "gate_digest": gate_digest,
            }
            pickle.dump(new_header, out_fh, protocol=pickle.HIGHEST_PROTOCOL)
            for seq in iter_feature_shard(src):
                opp = str(seq.opp_archetype or "").strip().casefold()
                route = route_for_archetype(opp)
                if route == UNKNOWN_ROUTE or not opp:
                    skipped_unknown += 1
                    skipped_unknown_decs += len(seq.decisions or [])
                    continue
                pkg = package_digest_for_expert(
                    archive_sha=archive_sha,
                    episode_id=str(seq.episode_id),
                    seat=int(seq.seat),
                    opp=opp,
                )
                opponent_id = f"expert:{opp}:{seq.episode_id}:{int(seq.seat)}"
                seq.target_provenance = {
                    "collect": "expert_self_play_equivalent_markup",
                    "self_play": False,
                    "opponent_training_group": "strong_public_practice",
                    "opponent_id": opponent_id,
                    "opponent_archetype_id": opp,
                    "opponent_content_digest": pkg,
                    "active_gate_id": gate_id,
                    "expert_ladder_markup": True,
                    "source_archive_sha256": archive_sha,
                    "trusted": True,
                }
                seq.matchup_adapter_training_ticket = {
                    "schema": TRAINING_TICKET_SCHEMA,
                    "opponent_id": opponent_id,
                    "package_digest": pkg,
                    "archetype_id": opp,
                    "route": int(route),
                    "corpus_manifest_digest": expert_manifest_digest,
                    "gate_contract_digest": gate_digest,
                    "episode_id": str(seq.episode_id),
                    "seat": int(seq.seat),
                    "acting_archetype_id": SPECIALIST,
                }
                for decision in seq.decisions:
                    decision.matchup_adapter_oracle_route = int(route)
                    decision.matchup_adapter_public_route = UNKNOWN_ROUTE
                adapter_training_ticket(seq)
                pickle.dump(seq, out_fh, protocol=pickle.HIGHEST_PROTOCOL)
                n_seq += 1
                n_dec += len(seq.decisions or [])
                ticketed_games += 1
                ticketed_decs += len(seq.decisions or [])
                route_games[opp] += 1
                route_decs[opp] += len(seq.decisions or [])
            footer = {
                "records": n_seq,
                "format": SHARD_FORMAT,
                "format_version": SHARD_FORMAT_VERSION,
            }
            pickle.dump(footer, out_fh, protocol=pickle.HIGHEST_PROTOCOL)
        dig = file_sha256(out_path)
        shard_rows.append(
            {
                "path": out_path.name,
                "source": str(src),
                "records": n_seq,
                "decisions": n_dec,
                "sha256": dig,
            }
        )
        print(f"wrote {out_path.name} games={n_seq} dec={n_dec}", flush=True)

    corpus_digest = canonical_digest(
        {
            "schema": "poke_bot.live_matchup_adapter_corpus/v1",
            "gate_digest": gate_digest,
            "shards": shard_rows,
            "ticketed_games": ticketed_games,
        }
    )

    traj_out = OUT / "expert_trajectory_self_play_ticketed.jsonl"
    traj_games = 0
    traj_decs = 0
    traj_ticketed = 0
    with TRAJ.open(encoding="utf-8") as inp, traj_out.open(
        "w", encoding="utf-8"
    ) as out:
        for line in inp:
            o = json.loads(line)
            traj_games += 1
            nd = len(o.get("decisions") or [])
            traj_decs += nd
            tp = dict(o.get("target_provenance") or {})
            if not (tp.get("self_play") and tp.get("collect") == "self_play"):
                out.write(json.dumps(o, separators=(",", ":")) + "\n")
                continue
            opp = str(
                tp.get("opponent_archetype_id") or o.get("opp_archetype") or SPECIALIST
            ).casefold()
            route = route_for_archetype(opp)
            pkg = str(tp.get("opponent_checkpoint_digest") or "").strip().lower()
            if route == UNKNOWN_ROUTE or not pkg.startswith("sha256:"):
                out.write(json.dumps(o, separators=(",", ":")) + "\n")
                continue
            opponent_id = str(tp.get("opponent_id") or f"self:{SPECIALIST}")
            ticket = {
                "schema": TRAINING_TICKET_SCHEMA,
                "opponent_id": opponent_id,
                "package_digest": pkg,
                "archetype_id": opp,
                "route": int(route),
                "corpus_manifest_digest": corpus_digest,
                "gate_contract_digest": gate_digest,
                "episode_id": str(o.get("episode_id") or o.get("game_id") or ""),
                "seat": int(o.get("seat") if o.get("seat") is not None else 0),
                "acting_archetype_id": SPECIALIST,
            }
            o["matchup_adapter_training_ticket"] = ticket
            for d in o.get("decisions") or []:
                if isinstance(d, dict):
                    d["matchup_adapter_oracle_route"] = int(route)
                    d["matchup_adapter_public_route"] = int(UNKNOWN_ROUTE)
            out.write(json.dumps(o, separators=(",", ":")) + "\n")
            traj_ticketed += 1
            if traj_games % 5000 == 0:
                print(
                    f"traj progress games={traj_games} ticketed={traj_ticketed}",
                    flush=True,
                )

    manifest = {
        "format": "pokebot-bootstrap-feature-manifest",
        "format_version": 1,
        "specialist_id": SPECIALIST,
        "schema": "poke_bot.slop_box_expert_self_play_equivalent_markup/v1",
        "shards": shard_rows,
        "totals": {
            "ticketed_games": ticketed_games,
            "ticketed_decisions": ticketed_decs,
            "skipped_unknown_games": skipped_unknown,
            "skipped_unknown_decisions": skipped_unknown_decs,
            "route_games": dict(route_games),
            "route_decisions": dict(route_decs),
        },
        "source_expert": str(EXPERT),
        "gate": str(GATE),
        "gate_digest": gate_digest,
        "corpus_digest": corpus_digest,
        "expert_manifest_digest": expert_manifest_digest,
    }
    (feat_out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (feat_out / "PROTECTED_EXPERT_CORPUS.json").write_text(
        json.dumps(
            {
                "schema": "poke_bot.pinned_expert_corpus/v1",
                "protected": True,
                "manifest": "manifest.json",
                "manifest_sha256": file_sha256(feat_out / "manifest.json"),
                "selection": {
                    "field": "GameSequence.archetype",
                    "operator": "exact_casefold",
                    "value": SPECIALIST,
                    "seat_semantics": "acting_seat_only",
                },
                "totals": {
                    "records_kept": ticketed_games,
                    "decisions_kept": ticketed_decs,
                },
                "markup": "self_play_equivalent_matchup_tickets_r171",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    receipt = {
        "schema": "poke_bot.slop_box_expert_self_play_equivalent_markup_r171/v1",
        "status": "markup_complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "agent": "7bc394c0",
        "specialist_id": SPECIALIST,
        "expert_feature_markup": {
            "output_dir": str(feat_out),
            "ticketed_games": ticketed_games,
            "ticketed_decisions": ticketed_decs,
            "skipped_unknown_games": skipped_unknown,
            "route_games": dict(route_games),
            "manifest_sha256": file_sha256(feat_out / "manifest.json"),
            "corpus_digest": corpus_digest,
        },
        "trajectory_self_play_markup": {
            "source": str(TRAJ),
            "output": str(traj_out),
            "games": traj_games,
            "decisions": traj_decs,
            "ticketed_games": traj_ticketed,
            "sha256": file_sha256(traj_out),
        },
        "gate_digest": gate_digest,
        "combo_state_labels": "pending_sibling_f93685e2",
        "schema8_select_context": "pending_sibling_f163b865",
        "ready_for_fresh_bootstrap": True,
    }
    out_receipt = (
        ROOT / "outputs/state/slop-box-expert-self-play-equivalent-markup-r171.json"
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    out_receipt.write_text(text, encoding="utf-8")
    (OUT / "markup.receipt.json").write_text(text, encoding="utf-8")
    print(
        "MARKUP_COMPLETE",
        json.dumps(
            {
                "ticketed_games": ticketed_games,
                "ticketed_decisions": ticketed_decs,
                "traj_ticketed": traj_ticketed,
                "traj_games": traj_games,
                "receipt_sha256": file_sha256(out_receipt),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
