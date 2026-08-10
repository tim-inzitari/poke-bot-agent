#!/usr/bin/env python3
"""Mark up Slop Box expert features as self-play-equivalent adapter tickets.

Owner r171: run the expert acting-seat corpus through the same matchup-ticket
/ Format-6 oracle-route labeling path RL uses for dormant adapter fitting.
Does not train adapters and does not touch live Slop Box weights.

Parallelism:
  --jobs N  ProcessPool over archives + feature shards (CPU/parse bound).
  Each archive worker also uses a ThreadPool for zip member I/O.

Output:
  - ticketed temporal-expert feature shards (GameSequence + tickets)
  - ticket index JSONL (compact proof / handoff)
  - immutable markup receipt for sibling wipe+fresh bootstrap
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import tempfile
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from poke_bot.feature_shards import (
    COMPACT_MODE_TEMPORAL_EXPERT,
    SHARD_FORMAT,
    SHARD_FORMAT_VERSION,
    iter_feature_shard,
)
from poke_bot.matchup_adapter_activation import TRAINING_TICKET_SCHEMA
from poke_bot.matchup_adapters import (
    EXPERT_IDS,
    LOGICAL_EXPERT_ALIASES_V5,
    UNKNOWN_ROUTE,
)
from poke_bot.matchup_adapters_v6 import (
    load_slot_registry,
    registry_digest,
    route_for_archetype as v6_route_for_archetype,
)
from poke_bot.pure_rl.matchup_adapter_corpus import full_deck_digest, sha256_file
from poke_bot.replay_import import _agent_names, episode_id_of, extract_setup_decks


SPECIALIST = "teal-mask-ogerpon-ex"
MARKUP_SCHEMA = "poke_bot.slop_box_expert_matchup_selfplay_equiv_markup/v1"
TICKET_INDEX_SCHEMA = "poke_bot.slop_box_expert_matchup_ticket_index_row/v1"
IDENTITY_KIND = "official-ladder-archive-agent-and-full-deck/v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_digest(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _bytes_digest(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return sha256_file(path)


def _normalize_opponent(archetype_id: str, aliases: Mapping[str, str]) -> str:
    raw = str(archetype_id or "").strip().casefold()
    return str(aliases.get(raw, raw))


def _opponent_identity(agent_name: str, deck_digest: str) -> tuple[str, str]:
    from poke_bot.matchup_adapter_activation import normalize_matchup_identity

    agent = normalize_matchup_identity(agent_name) or "unknown-agent"
    content = _canonical_digest(
        {
            "kind": IDENTITY_KIND,
            "agent": agent,
            "deck_digest": deck_digest,
        }
    )
    return f"expert-ladder:{content[7:31]}", content


def _feature_header(path: Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        header = pickle.load(stream)
    if not isinstance(header, dict):
        raise ValueError(f"invalid feature header: {path}")
    return header


def _oracle_row_from_member(
    *,
    episode_id: str,
    raw: bytes,
    payload: dict[str, Any],
    wanted_keys: set[tuple[str, int]],
    archive_digest: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    if str(episode_id_of(payload)) != str(episode_id):
        return {}
    decks = extract_setup_decks(payload)
    agents = _agent_names(payload)
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for seat in (0, 1):
        key = (str(episode_id), int(seat))
        if key not in wanted_keys:
            continue
        opponent = 1 - seat
        opponent_deck = decks[opponent]
        if opponent_deck is None or len(opponent_deck) != 60:
            continue
        deck_digest = full_deck_digest(opponent_deck)
        opponent_id, content_digest = _opponent_identity(agents[opponent], deck_digest)
        rows[key] = {
            "episode_id": str(episode_id),
            "seat": int(seat),
            "opponent_id": opponent_id,
            "opponent_content_digest": content_digest,
            "opponent_deck_digest": deck_digest,
            "source_archive_digest": archive_digest,
            "source_member_digest": _bytes_digest(raw),
            "agent_name": str(agents[opponent]),
        }
    return rows


def _archive_oracle_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker: join one day archive to feature episode keys.

    Uses one shared ZipFile per archive process. Re-opening the same SMB zip
    from many threads thrashes TrueNAS; serialize member reads under a lock
    (or run single-threaded) while still parallelizing across day archives.
    """

    import threading

    archive_path = str(task["archive_path"])
    archive_name = str(task["archive_name"])
    archive_digest = str(task["archive_digest"])
    checkpoint_path = Path(str(task["checkpoint_path"]))
    thread_workers = max(1, int(task["thread_workers"]))
    wanted_keys = {(str(e), int(s)) for e, s in task["wanted_keys"]}
    if checkpoint_path.is_file():
        payload = pickle.loads(checkpoint_path.read_bytes())
        rows = dict(payload.get("rows") or {})
        return {
            "archive_name": archive_name,
            "archive_digest": archive_digest,
            "rows": rows,
            "oracle_rows": len(rows),
            "wanted_keys": len(wanted_keys),
            "resumed": True,
        }

    wanted_episodes = sorted({episode_id for episode_id, _seat in wanted_keys})
    print(
        json.dumps(
            {
                "stage": "archive_oracle_join",
                "archive": archive_name,
                "wanted_keys": len(wanted_keys),
                "wanted_episodes": len(wanted_episodes),
                "thread_workers": thread_workers,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with zipfile.ZipFile(archive_path, "r") as stream:
        names = set(stream.namelist())
        name_by_episode = {}
        for episode_id in wanted_episodes:
            member = f"{episode_id}.json"
            if member in names:
                name_by_episode[episode_id] = member
                continue
            matches = [
                name
                for name in names
                if name.endswith(f"/{member}") or name.endswith(member)
            ]
            if matches:
                name_by_episode[episode_id] = matches[0]
        zip_lock = threading.Lock()

        def _read_episode(episode_id: str) -> dict[tuple[str, int], dict[str, Any]]:
            member = name_by_episode.get(episode_id)
            if member is None:
                return {}
            with zip_lock:
                raw = stream.read(member)
            try:
                payload = json.loads(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                return {}
            return _oracle_row_from_member(
                episode_id=episode_id,
                raw=raw,
                payload=payload,
                wanted_keys=wanted_keys,
                archive_digest=archive_digest,
            )

        # Keep a small thread pool for JSON/deck parse overlap, but all zip
        # reads share one handle under a lock (SMB-safe).
        if thread_workers <= 1 or len(wanted_episodes) <= 8:
            for episode_id in wanted_episodes:
                rows.update(_read_episode(episode_id))
        else:
            with ThreadPoolExecutor(max_workers=thread_workers) as pool:
                futures = [
                    pool.submit(_read_episode, episode_id)
                    for episode_id in wanted_episodes
                ]
                done = 0
                for future in as_completed(futures):
                    rows.update(future.result())
                    done += 1
                    if done % 50 == 0 or done == len(futures):
                        print(
                            json.dumps(
                                {
                                    "stage": "archive_oracle_join_progress",
                                    "archive": archive_name,
                                    "done_episodes": done,
                                    "total_episodes": len(futures),
                                    "oracle_rows": len(rows),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".tmp")
    temporary.write_bytes(
        pickle.dumps(
            {
                "archive_name": archive_name,
                "archive_digest": archive_digest,
                "rows": rows,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )
    os.replace(temporary, checkpoint_path)
    return {
        "archive_name": archive_name,
        "archive_digest": archive_digest,
        "rows": rows,
        "oracle_rows": len(rows),
        "wanted_keys": len(wanted_keys),
        "resumed": False,
    }


def _plan_shard_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker: collect episode/seat keys for one feature shard."""

    feature_path = Path(str(task["feature_path"]))
    header = _feature_header(feature_path)
    archive_name = str(header.get("source_archive") or "")
    if not archive_name:
        raise ValueError(f"feature shard lacks source_archive: {feature_path}")
    keys: list[tuple[str, int]] = []
    games = 0
    decisions = 0
    for sequence in iter_feature_shard(feature_path):
        games += 1
        decisions += len(sequence.decisions or [])
        keys.append((str(sequence.episode_id), int(sequence.seat)))
    return {
        "feature_path": str(feature_path),
        "feature_digest": sha256_file(feature_path),
        "source_row": dict(task["source_row"]),
        "header": header,
        "archive_name": archive_name,
        "keys": keys,
        "games": games,
        "decisions": decisions,
    }


def _markup_shard_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Process-pool worker: ticket one feature shard to an output file."""

    from poke_bot.matchup_adapters_v6 import load_slot_registry

    feature_path = Path(str(task["feature_path"]))
    out_path = Path(str(task["out_path"]))
    index_path = Path(str(task["index_path"]))
    done_path = Path(str(task["done_path"]))
    if done_path.is_file() and out_path.is_file() and index_path.is_file():
        return pickle.loads(done_path.read_bytes())

    registry = load_slot_registry(Path(str(task["roster_path"])))
    aliases = dict(LOGICAL_EXPERT_ALIASES_V5)
    aliases.update(dict(registry.get("logical_aliases") or {}))
    oracle_by_key = {
        (str(e), int(s)): dict(row)
        for (e, s), row in task["oracle_rows"]
    }
    header = dict(task["header"])
    source_digest = str(task["source_digest"])
    gate_contract_digest = str(task["gate_contract_digest"])
    gate_file_digest = str(task["gate_file_digest"])
    roster_digest = str(task["roster_digest"])
    feature_digest = str(task["feature_digest"])
    archive_digest = str(task["archive_digest"])
    out_name = str(task["out_name"])

    excluded: Counter[str] = Counter()
    route_sequences: Counter[str] = Counter()
    route_decisions: Counter[str] = Counter()
    package_registry: dict[str, dict[str, str]] = {}
    membership_rows: list[tuple[str, int, str, int]] = []
    index_lines: list[str] = []
    kept = 0
    kept_decisions = 0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_out = out_path.with_suffix(out_path.suffix + ".partial")
    with temporary_out.open("wb") as out_stream:
        out_header = {
            **header,
            "format": SHARD_FORMAT,
            "format_version": SHARD_FORMAT_VERSION,
            "compact_mode": COMPACT_MODE_TEMPORAL_EXPERT,
            "matchup_adapter_marked_selfplay_equiv": True,
            "matchup_adapter_format": "poke-bot-matchup-adapter-bank-v6",
            "slot_registry_digest": roster_digest,
            "markup_schema": MARKUP_SCHEMA,
            "offline_oracle_only": True,
            "runtime_routes_enabled": False,
            "source_feature_digest": feature_digest,
            "gate_contract_digest": gate_contract_digest,
            "gate_contract_file_digest": gate_file_digest,
            "acting_archetype": SPECIALIST,
        }
        pickle.dump(out_header, out_stream, protocol=pickle.HIGHEST_PROTOCOL)
        for sequence in iter_feature_shard(feature_path):
            key = (str(sequence.episode_id), int(sequence.seat))
            if str(sequence.archetype or "").casefold() != SPECIALIST:
                excluded["wrong_acting_archetype"] += 1
                continue
            oracle = oracle_by_key.get(key)
            if oracle is None:
                excluded["missing_archive_oracle"] += 1
                continue
            canonical_opp = _normalize_opponent(
                str(sequence.opp_archetype or ""), aliases
            )
            route = int(v6_route_for_archetype(canonical_opp, registry=registry))
            if route == UNKNOWN_ROUTE:
                excluded["unsupported_or_unroutable"] += 1
                continue
            if route >= len(EXPERT_IDS) and canonical_opp not in {
                SPECIALIST,
                "slowking",
            }:
                excluded["unsupported_or_unroutable"] += 1
                continue

            sequence.opp_archetype = canonical_opp
            provenance = dict(sequence.target_provenance or {})
            provenance.update(
                {
                    "pure_rl": False,
                    "self_play": False,
                    "self_play_equivalent_markup": True,
                    "collect": "expert_ladder_selfplay_equiv",
                    "opponent_training_group": "expert_ladder",
                    "opponent_id": oracle["opponent_id"],
                    "opponent_archetype_id": canonical_opp,
                    "opponent_content_digest": oracle["opponent_content_digest"],
                    "opponent_deck_digest": oracle["opponent_deck_digest"],
                    "source_archive_digest": oracle["source_archive_digest"],
                    "source_member_digest": oracle["source_member_digest"],
                    "formal_eval": False,
                    "trusted": True,
                    "slop_box_expert_matchup_markup_r171": True,
                }
            )
            sequence.target_provenance = provenance
            ticket = {
                "schema": TRAINING_TICKET_SCHEMA,
                "opponent_id": oracle["opponent_id"],
                "package_digest": oracle["opponent_content_digest"],
                "archetype_id": canonical_opp,
                "route": route,
                "corpus_manifest_digest": source_digest,
                "gate_contract_digest": gate_contract_digest,
                "episode_id": str(sequence.episode_id),
                "seat": int(sequence.seat),
                "acting_archetype_id": SPECIALIST,
                "identity_kind": IDENTITY_KIND,
                "opponent_deck_digest": oracle["opponent_deck_digest"],
                "source_archive_digest": oracle["source_archive_digest"],
                "source_member_digest": oracle["source_member_digest"],
                "source_feature_digest": feature_digest,
                "slot_registry_digest": roster_digest,
                "formal_eval": False,
                "runtime_route_authorized": False,
                "self_play_equivalent_markup": True,
            }
            sequence.matchup_adapter_training_ticket = ticket
            for decision in sequence.decisions:
                decision.matchup_adapter_oracle_route = route
                decision.matchup_adapter_public_route = UNKNOWN_ROUTE
            pickle.dump(sequence, out_stream, protocol=pickle.HIGHEST_PROTOCOL)
            kept += 1
            n_dec = len(sequence.decisions or [])
            kept_decisions += n_dec
            route_sequences[canonical_opp] += 1
            route_decisions[canonical_opp] += n_dec
            membership_rows.append(
                (str(sequence.episode_id), int(sequence.seat), canonical_opp, route)
            )
            package_registry[oracle["opponent_id"]] = {
                "content_digest": oracle["opponent_content_digest"],
                "agent_name": oracle["agent_name"],
            }
            index_lines.append(
                json.dumps(
                    {
                        "schema": TICKET_INDEX_SCHEMA,
                        "episode_id": str(sequence.episode_id),
                        "seat": int(sequence.seat),
                        "acting_archetype_id": SPECIALIST,
                        "archetype_id": canonical_opp,
                        "route": route,
                        "opponent_id": oracle["opponent_id"],
                        "package_digest": oracle["opponent_content_digest"],
                        "decisions": n_dec,
                        "source_feature": out_name,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
        footer = {
            "format": SHARD_FORMAT + "-footer",
            "format_version": SHARD_FORMAT_VERSION,
            "stats": {
                "records_total": kept,
                "records_kept": kept,
                "records_dropped": 0,
                "decisions_kept": kept_decisions,
            },
        }
        pickle.dump(footer, out_stream, protocol=pickle.HIGHEST_PROTOCOL)
        out_stream.flush()
        os.fsync(out_stream.fileno())
    os.replace(temporary_out, out_path)
    index_path.write_text(
        ("\n".join(index_lines) + ("\n" if index_lines else "")), encoding="utf-8"
    )
    result = {
        "out_name": out_name,
        "path": f"features/{out_name}",
        "sha256": sha256_file(out_path),
        "bytes": out_path.stat().st_size,
        "source_feature_digest": feature_digest,
        "source_archive": str(task["archive_name"]),
        "source_archive_sha256": archive_digest,
        "stats": {"records_kept": kept, "decisions_kept": kept_decisions},
        "excluded": dict(excluded),
        "route_sequences": dict(route_sequences),
        "route_decisions": dict(route_decisions),
        "package_registry": package_registry,
        "membership_rows": membership_rows,
        "index_path": str(index_path),
    }
    done_path.write_bytes(pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL))
    return result


def markup_corpus(
    *,
    feature_manifest: Path,
    archive_dir: Path,
    output_dir: Path,
    roster_path: Path,
    gate_contract_path: Path,
    jobs: int,
    thread_workers: int,
    work_dir: Optional[Path] = None,
    val_frac: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    feature_manifest = Path(feature_manifest).expanduser().resolve()
    archive_dir = Path(archive_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    roster_path = Path(roster_path).expanduser().resolve()
    gate_contract_path = Path(gate_contract_path).expanduser().resolve()
    jobs = max(1, int(jobs))
    thread_workers = max(1, int(thread_workers))

    if output_dir.exists():
        raise FileExistsError(output_dir)

    manifest = json.loads(feature_manifest.read_text(encoding="utf-8"))
    if manifest.get("format") != "pokebot-bootstrap-feature-manifest":
        raise ValueError("source is not a bootstrap feature manifest")
    selection = dict(manifest.get("selection") or {})
    if str(selection.get("value") or "").casefold() != SPECIALIST:
        raise ValueError(f"feature manifest is not {SPECIALIST}-acting")

    registry = load_slot_registry(roster_path)
    roster_digest = registry_digest(registry)
    gate_payload = json.loads(gate_contract_path.read_text(encoding="utf-8"))
    gate_file_digest = sha256_file(gate_contract_path)
    gate_contract_digest = _canonical_digest(gate_payload)
    source_digest = sha256_file(feature_manifest)
    parent = feature_manifest.parent

    if work_dir is None:
        work_dir = output_dir.parent / f".{output_dir.name}.work"
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    oracle_ckpt_dir = work_dir / "oracle"
    shard_ckpt_dir = work_dir / "shards"
    oracle_ckpt_dir.mkdir(parents=True, exist_ok=True)
    shard_ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(
        json.dumps(
            {
                "stage": "parallel_markup_begin",
                "jobs": jobs,
                "thread_workers": thread_workers,
                "host_cpus": os.cpu_count(),
                "work_dir": str(work_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )

    plan_tasks = [
        {
            "feature_path": str((parent / str(row["path"])).resolve()),
            "source_row": dict(row),
        }
        for row in list(manifest.get("shards") or [])
    ]
    shard_plans: list[dict[str, Any]] = []
    source_games = 0
    source_decisions = 0
    with ProcessPoolExecutor(max_workers=min(jobs, max(1, len(plan_tasks)))) as pool:
        futures = [pool.submit(_plan_shard_worker, task) for task in plan_tasks]
        for future in as_completed(futures):
            plan = future.result()
            shard_plans.append(plan)
            source_games += int(plan["games"])
            source_decisions += int(plan["decisions"])
            print(
                json.dumps(
                    {
                        "stage": "plan_shard_done",
                        "feature": Path(plan["feature_path"]).name,
                        "games": plan["games"],
                        "decisions": plan["decisions"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    shard_plans.sort(key=lambda row: str(row["source_row"]["path"]))

    wanted_by_archive: dict[str, set[tuple[str, int]]] = {}
    header_digest_by_archive: dict[str, str] = {}
    for plan in shard_plans:
        name = str(plan["archive_name"])
        digest = str(plan["header"].get("source_archive_sha256") or "").strip().lower()
        if not digest.startswith("sha256:") or len(digest) != 71:
            raise ValueError(f"feature shard lacks archive digest: {plan['feature_path']}")
        prior = header_digest_by_archive.get(name)
        if prior is not None and prior != digest:
            raise ValueError(f"conflicting archive digests for {name}")
        header_digest_by_archive[name] = digest
        wanted_by_archive.setdefault(name, set()).update(
            (str(e), int(s)) for e, s in plan["keys"]
        )

    archive_tasks = []
    for archive_name, keys in sorted(wanted_by_archive.items()):
        archive = archive_dir / archive_name
        if not archive.is_file():
            raise FileNotFoundError(archive)
        archive_tasks.append(
            {
                "archive_path": str(archive),
                "archive_name": archive_name,
                "archive_digest": header_digest_by_archive[archive_name],
                "wanted_keys": sorted(keys),
                "checkpoint_path": str(
                    oracle_ckpt_dir / f"{archive_name}.oracle.pkl"
                ),
                "thread_workers": thread_workers,
            }
        )

    oracle_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    archive_digests: dict[str, str] = {}
    with ProcessPoolExecutor(
        max_workers=min(jobs, max(1, len(archive_tasks)))
    ) as pool:
        futures = {
            pool.submit(_archive_oracle_worker, task): task["archive_name"]
            for task in archive_tasks
        }
        for future in as_completed(futures):
            result = future.result()
            archive_digests[result["archive_name"]] = result["archive_digest"]
            normalized: dict[tuple[str, int], dict[str, Any]] = {}
            for key, row in dict(result["rows"] or {}).items():
                if isinstance(key, (list, tuple)) and len(key) == 2:
                    normalized[(str(key[0]), int(key[1]))] = dict(row)
            oracle_by_key.update(normalized)
            print(
                json.dumps(
                    {
                        "stage": "archive_oracle_join_done",
                        "archive": result["archive_name"],
                        "oracle_rows": result["oracle_rows"],
                        "wanted_keys": result["wanted_keys"],
                        "resumed": result["resumed"],
                        "jobs": jobs,
                        "thread_workers": thread_workers,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial.", dir=output_dir.parent)
    )
    features_out_dir = temporary / "features"
    features_out_dir.mkdir(parents=True, exist_ok=True)
    index_parts_dir = work_dir / "index_parts"
    index_parts_dir.mkdir(parents=True, exist_ok=True)

    markup_tasks = []
    for plan in shard_plans:
        out_name = str(plan["source_row"]["path"])
        markup_tasks.append(
            {
                "feature_path": plan["feature_path"],
                "out_path": str(features_out_dir / out_name),
                "index_path": str(index_parts_dir / f"{out_name}.jsonl"),
                "done_path": str(shard_ckpt_dir / f"{out_name}.done.pkl"),
                "header": plan["header"],
                "feature_digest": plan["feature_digest"],
                "archive_name": plan["archive_name"],
                "archive_digest": archive_digests[plan["archive_name"]],
                "out_name": out_name,
                "roster_path": str(roster_path),
                "roster_digest": roster_digest,
                "source_digest": source_digest,
                "gate_contract_digest": gate_contract_digest,
                "gate_file_digest": gate_file_digest,
                "oracle_rows": [
                    ((str(e), int(s)), oracle_by_key[(str(e), int(s))])
                    for e, s in plan["keys"]
                    if (str(e), int(s)) in oracle_by_key
                ],
            }
        )

    output_shards: list[dict[str, Any]] = []
    route_sequences: Counter[str] = Counter()
    route_decisions: Counter[str] = Counter()
    excluded: Counter[str] = Counter()
    package_registry: dict[str, dict[str, str]] = {}
    membership_rows: list[tuple[str, int, str, int]] = []
    selected_games = 0
    selected_decisions = 0
    index_paths: list[Path] = []

    try:
        with ProcessPoolExecutor(
            max_workers=min(jobs, max(1, len(markup_tasks)))
        ) as pool:
            futures = [pool.submit(_markup_shard_worker, task) for task in markup_tasks]
            for future in as_completed(futures):
                result = future.result()
                output_shards.append(
                    {
                        "path": result["path"],
                        "sha256": result["sha256"],
                        "bytes": result["bytes"],
                        "source_feature_digest": result["source_feature_digest"],
                        "source_archive": result["source_archive"],
                        "source_archive_sha256": result["source_archive_sha256"],
                        "stats": result["stats"],
                    }
                )
                selected_games += int(result["stats"]["records_kept"])
                selected_decisions += int(result["stats"]["decisions_kept"])
                excluded.update(result.get("excluded") or {})
                route_sequences.update(result.get("route_sequences") or {})
                route_decisions.update(result.get("route_decisions") or {})
                package_registry.update(result.get("package_registry") or {})
                membership_rows.extend(
                    [
                        (str(e), int(s), str(a), int(r))
                        for e, s, a, r in result.get("membership_rows") or []
                    ]
                )
                index_paths.append(Path(result["index_path"]))
                print(
                    json.dumps(
                        {
                            "stage": "markup_shard_done",
                            "feature": result["out_name"],
                            "records_kept": result["stats"]["records_kept"],
                            "decisions_kept": result["stats"]["decisions_kept"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        output_shards.sort(key=lambda row: row["path"])
        ticket_index_path = temporary / "ticket_index.jsonl"
        with ticket_index_path.open("w", encoding="utf-8") as index_stream:
            for path in sorted(index_paths):
                if path.is_file() and path.stat().st_size:
                    index_stream.write(path.read_text(encoding="utf-8"))
                    if not index_stream.tell() or True:
                        pass

        membership_digest = _canonical_digest(
            {
                "schema": "poke_bot.slop_box_expert_matchup_membership/v1",
                "seed": int(seed),
                "val_frac": float(val_frac),
                "rows": sorted(membership_rows),
            }
        )
        package_registry_path = temporary / "opponent-registry.json"
        package_registry_digest = _write_json(
            package_registry_path,
            {
                "schema": "poke_bot.matchup_adapter_package_registry/v1",
                "identity_kind": IDENTITY_KIND,
                "packages": [
                    {
                        "opponent_id": opponent_id,
                        "content_digest": row["content_digest"],
                        "aliases": [],
                        "agent_name": row["agent_name"],
                    }
                    for opponent_id, row in sorted(package_registry.items())
                ],
            },
        )
        ticket_index_digest = sha256_file(ticket_index_path)
        markup_manifest = {
            "schema": MARKUP_SCHEMA,
            "status": "ready",
            "created_at_utc": _utc_now(),
            "specialist_id": SPECIALIST,
            "parallelism": {
                "process_jobs": jobs,
                "thread_workers_per_archive": thread_workers,
                "host_cpus": os.cpu_count(),
            },
            "purpose": (
                "self-play-equivalent expert matchup markup for forced "
                "Format-6 adapter bootstrap after owner wipe"
            ),
            "offline_oracle_only": True,
            "runtime_routes_enabled": False,
            "do_not_fit_on_pre_wipe_weights": True,
            "handoff_owner": "7bc394c0",
            "source_feature_manifest": str(feature_manifest),
            "source_feature_manifest_digest": source_digest,
            "archive_dir": str(archive_dir),
            "slot_registry": str(roster_path),
            "slot_registry_digest": roster_digest,
            "gate_contract": str(gate_contract_path),
            "gate_contract_digest": gate_contract_digest,
            "gate_contract_file_digest": gate_file_digest,
            "package_registry": "opponent-registry.json",
            "package_registry_digest": package_registry_digest,
            "ticket_index": "ticket_index.jsonl",
            "ticket_index_digest": ticket_index_digest,
            "membership_digest": membership_digest,
            "matchup_adapter_format": "poke-bot-matchup-adapter-bank-v6",
            "v5_prefix_expert_ids": list(EXPERT_IDS),
            "totals": {
                "source_games": source_games,
                "source_decisions": source_decisions,
                "marked_games": selected_games,
                "marked_decisions": selected_decisions,
                "excluded": dict(excluded),
                "route_sequences": dict(sorted(route_sequences.items())),
                "route_decisions": dict(sorted(route_decisions.items())),
                "package_count": len(package_registry),
            },
            "shards": output_shards,
            "consumer_contract": {
                "requires_format6_registry": True,
                "ticket_schema": TRAINING_TICKET_SCHEMA,
                "decision_fields": [
                    "matchup_adapter_oracle_route",
                    "matchup_adapter_public_route",
                ],
                "sequence_fields": [
                    "matchup_adapter_training_ticket",
                    "target_provenance.self_play_equivalent_markup",
                ],
                "fit_entrypoints": [
                    "poke_bot.train._train_dormant_matchup_adapter_phase",
                    "scripts/train_matchup_adapters.py (Format-6-aware successor)",
                ],
                "note": (
                    "Sibling 7bc394c0 owns wipe of current Slop Box learned "
                    "weights and fresh bootstrap that forces adapter fit from "
                    "this marked corpus. Do not long-fit adapters on pre-wipe ckpt."
                ),
            },
        }
        _write_json(temporary / "manifest.json", markup_manifest)
        receipt = {
            "schema": "poke_bot.slop_box_h10_rtp_expert_matchup_markup_r171/v1",
            "status": "ready_for_wipe_and_fresh_adapter_bootstrap",
            "created_at_utc": _utc_now(),
            "specialist_id": SPECIALIST,
            "markup_manifest": str(output_dir / "manifest.json"),
            "marked_games": selected_games,
            "marked_decisions": selected_decisions,
            "source_games": source_games,
            "source_decisions": source_decisions,
            "excluded": dict(excluded),
            "route_sequences": dict(sorted(route_sequences.items())),
            "route_decisions": dict(sorted(route_decisions.items())),
            "source_feature_manifest_digest": source_digest,
            "slot_registry_digest": roster_digest,
            "gate_contract_digest": gate_contract_digest,
            "package_registry_digest": package_registry_digest,
            "ticket_index_digest": ticket_index_digest,
            "membership_digest": membership_digest,
            "parallelism": {
                "process_jobs": jobs,
                "thread_workers_per_archive": thread_workers,
                "host_cpus": os.cpu_count(),
            },
            "do_not_fit_on_pre_wipe_weights": True,
            "handoff_owner": "7bc394c0",
            "parent_weights_policy": (
                "delete current Slop Box learned weights; hot-start fresh "
                "bootstrap then forced adapter fit from this marked corpus"
            ),
        }
        _write_json(temporary / "MARKUP_READY.json", receipt)
        os.replace(temporary, output_dir)
        return receipt
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    cpu = int(os.cpu_count() or 8)
    # Physical-core-ish default on 7950X (16c/32t): leave headroom for remat/pack.
    default_jobs = max(4, min(14, cpu - 4))
    default_threads = max(4, min(8, max(2, cpu // default_jobs)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--roster",
        type=Path,
        default=Path("state/matchup_adapter_roster.json"),
    )
    parser.add_argument("--gate-contract", type=Path, required=True)
    parser.add_argument("--receipt-out", type=Path, required=True)
    parser.add_argument(
        "--jobs",
        type=int,
        default=default_jobs,
        help="ProcessPool worker count for archives/shards",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Alias for --jobs",
    )
    parser.add_argument(
        "--thread-workers",
        type=int,
        default=default_threads,
        help="ThreadPool workers per archive for zip member I/O",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Durable checkpoint dir for resume across relaunches",
    )
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    jobs = int(args.workers if args.workers is not None else args.jobs)

    receipt = markup_corpus(
        feature_manifest=args.feature_manifest,
        archive_dir=args.archive_dir,
        output_dir=args.output_dir,
        roster_path=args.roster,
        gate_contract_path=args.gate_contract,
        jobs=jobs,
        thread_workers=int(args.thread_workers),
        work_dir=args.work_dir,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    receipt_digest = _write_json(args.receipt_out, receipt)
    receipt["receipt_path"] = str(Path(args.receipt_out).resolve())
    receipt["receipt_digest"] = receipt_digest
    _write_json(args.receipt_out, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
