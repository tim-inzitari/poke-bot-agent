#!/usr/bin/env python3
"""Derive and queue two NO-RTP Alakazam guide-logit A/B submissions."""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

SCHEMA = "poke_bot.alakazam_guide_logit_ab_r206/v1"
GUIDE_SCHEMA = "poke_bot.submission_guide_decision_policy/v1"
QUEUE_SCHEMA = "poke_bot.kaggle_submission_queue/v1"
BASE_BUNDLE_SHA256 = (
    "sha256:dfa8bfccf9ee41d2205c7e30d817489391bb6295fa1ed1eff78c36fd8a8b7145"
)
MODEL_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
DECK_SHA256 = "sha256:1705f0f4db0c54b32f297fc9292a417b0c3abc9fdb6edf6a5370af6a635efe65"
DECK_CARDS_SHA256 = (
    "sha256:660c1274aac19d88c40fd2bb52187f53dc639d944506760e386f2686b91cc247"
)
REPRESENTATIVES_SHA256 = (
    "sha256:4439874690fbaaeb72f9d224f92c37556e5c2ef6818192871220946975cfdc0f"
)
MATCHUP_TREE_SHA256 = (
    "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
)
SEARCH_CONFIG_SHA256 = (
    "sha256:7ce431662904d97727d6838bcd60d9f54426d7922058f9aa018614378fbca819"
)
BELIEF_DECKS_SHA256 = (
    "sha256:b8a7f709426652fe85c18b6f5c9cdb757dd99abef6fd04d62537805306e29af0"
)
AGENT_PATCH_NEEDLE = """            else:
                idx = max(range(len(candidates)), key=lambda i: policy[i])
            selected = list(candidates[idx])
            factorized_stages.append(
                {
                    \"action_combos\": [list(c) for c in candidates],
                    \"policy\": policy,
                    \"selected_index\": int(idx),
                }
            )
"""
AGENT_PATCH_REPLACEMENT = """            else:
                idx = max(range(len(candidates)), key=lambda i: policy[i])
            model_selected_index = int(idx)
            from .submission_guide_policy import select_index as _guide_select_index

            idx, guide_decision = _guide_select_index(
                observation=obs_dict,
                candidates=candidates,
                model_policy=policy,
                model_index=model_selected_index,
                deck=self.deck,
            )
            selected = list(candidates[idx])
            factorized_stages.append(
                {
                    \"action_combos\": [list(c) for c in candidates],
                    \"policy\": policy,
                    \"model_selected_index\": model_selected_index,
                    \"selected_index\": int(idx),
                    \"guide_decision\": guide_decision,
                }
            )
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    temporary.write_text(body, encoding="utf-8")
    os.replace(temporary, path)


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        name = member.name.removeprefix("./")
        if name in {"", "."} and member.isdir():
            continue
        target = (root / name).resolve()
        if (
            not name
            or name.startswith("/")
            or root not in target.parents
            or member.issym()
            or member.islnk()
        ):
            raise RuntimeError(f"unsafe base bundle member: {member.name}")
    archive.extractall(destination)


def _write_deterministic_bundle(source: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    with (
        temporary.open("wb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for path in sorted(source.rglob("*"), key=lambda item: item.as_posix()):
            relative = path.relative_to(source).as_posix()
            info = archive.gettarinfo(str(path), arcname=f"./{relative}")
            info.uid = 0
            info.gid = 0
            info.uname = "root"
            info.gname = "root"
            info.mtime = 0
            if path.is_file():
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
            else:
                archive.addfile(info)
    os.replace(temporary, output)


def derive_bundle(
    *,
    base_bundle: Path,
    helper_source: Path,
    output: Path,
    weight: float,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="pokebot-guide-r206-") as raw_temp:
        root = Path(raw_temp)
        with tarfile.open(base_bundle, "r:gz") as archive:
            _safe_extract(archive, root)
        agent_path = root / "poke_bot" / "agent.py"
        source = agent_path.read_text(encoding="utf-8")
        if source.count(AGENT_PATCH_NEEDLE) != 1:
            raise RuntimeError("exact r195 PolicyAgent patch boundary changed")
        agent_path.write_text(
            source.replace(AGENT_PATCH_NEEDLE, AGENT_PATCH_REPLACEMENT),
            encoding="utf-8",
        )
        helper_target = root / "poke_bot" / "submission_guide_policy.py"
        shutil.copy2(helper_source, helper_target)
        guide_policy = {
            "schema": GUIDE_SCHEMA,
            "mode": "guide_logit_bonus",
            "guide_id": "alakazam",
            "guide_version": "powerful-hand-v1",
            "guide_logit_weight": weight,
            "guide_score_normalization": "per_stage_min_max_0_to_1",
            "unavailable_or_tied_behavior": "exact_model_fallback",
            "model_parameters_mutated": False,
            "rtp_enabled": False,
            "owner_decision_revision": 206,
        }
        (root / "guide_policy.json").write_text(
            json.dumps(guide_policy, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        profile_path = root / "runtime_profile.json"
        profile = read_object(profile_path)
        if (
            profile.get("schema") != "poke_bot.submission_runtime_profile/v1"
            or profile.get("recursive_turn_planner") != "disabled"
            or profile.get("rtp_sidecar_packaged") is not False
        ):
            raise RuntimeError("base package is not the exact NO-RTP profile")
        profile["guide_decision_policy"] = guide_policy
        profile["display"] = f"NO RTP GUIDE LOGIT {weight:.2f}"
        profile_path.write_text(
            json.dumps(profile, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_deterministic_bundle(root, output)

    with tarfile.open(output, "r:gz") as archive:
        members = {
            member.name.removeprefix("./"): member
            for member in archive.getmembers()
            if member.isfile()
        }

        def member_digest(name: str) -> str:
            stream = archive.extractfile(members[name])
            if stream is None:
                raise RuntimeError(f"unreadable bundle member: {name}")
            return "sha256:" + hashlib.sha256(stream.read()).hexdigest()

        if any(name.endswith("rtp_shadow_planner.pt") for name in members):
            raise RuntimeError("NO-RTP guide package contains an RTP sidecar")
        checks = {
            "model": member_digest("model.pt") == MODEL_SHA256,
            "deck": member_digest("deck.csv") == DECK_SHA256,
            "matchup_tree": member_digest("matchup_tree.json") == MATCHUP_TREE_SHA256,
            "search_config": member_digest("search_config.json")
            == SEARCH_CONFIG_SHA256,
            "belief_decks": member_digest("belief_decks.json") == BELIEF_DECKS_SHA256,
            "helper": "poke_bot/submission_guide_policy.py" in members,
            "guide_policy": "guide_policy.json" in members,
        }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("derived guide bundle audit failed: " + ",".join(failed))
    base_attestation_path = Path(str(base_bundle) + ".go-first-verified.json")
    base_attestation = read_object(base_attestation_path)
    if (
        base_attestation.get("schema")
        != "poke_bot.submission_turn_order_attestation/v1"
        or base_attestation.get("turn_order_preference") != "first_if_allowed"
        or base_attestation.get("go_first_if_offered") is not True
        or base_attestation.get("go_second_if_offered") is not False
        or not {
            "integer_enum",
            "string_enum_reversed_options",
            "live_engine_prompt",
        }.issubset(set(base_attestation.get("verified_cases") or []))
    ):
        raise RuntimeError("base turn-order attestation changed")
    attestation = dict(base_attestation)
    attestation["file_sha256"] = sha256(output)
    attestation["submission_message_required_literal"] = "NO RTP GUIDE"
    attestation["derived_from_file_sha256"] = BASE_BUNDLE_SHA256
    attestation["guide_logit_weight"] = weight
    attestation["owner_decision_revision"] = 206
    attestation_path = Path(str(output) + ".go-first-verified.json")
    atomic_json(attestation_path, attestation)
    return {
        "path": str(output),
        "sha256": sha256(output),
        "bytes": output.stat().st_size,
        "guide_policy": guide_policy,
        "checks": checks,
        "turn_order_attestation": str(attestation_path),
        "turn_order_attestation_sha256": sha256(attestation_path),
    }


def queue_variants(
    *, queue_path: Path, variants: list[dict[str, Any]], output_root: Path
) -> list[dict[str, Any]]:
    lock_path = queue_path.with_suffix(queue_path.suffix + ".lock")
    queued_at = dt.datetime.now(dt.timezone.utc).isoformat()
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        payload = read_object(queue_path)
        if payload.get("schema") != QUEUE_SCHEMA:
            raise RuntimeError("Kaggle submission queue schema changed")
        entries = [dict(row) for row in payload.get("queue") or []]
        selected: list[dict[str, Any]] = []
        for slot, variant in enumerate(variants, start=1):
            weight = float(variant["guide_policy"]["guide_logit_weight"])
            target_dir = output_root / f"copy-{slot}"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "submission.tar.gz"
            source = Path(str(variant["path"]))
            if not target.exists():
                shutil.copy2(source, target)
            if sha256(target) != variant["sha256"]:
                raise RuntimeError("immutable guide submission copy changed")
            source_attestation = Path(str(source) + ".go-first-verified.json")
            target_attestation = Path(str(target) + ".go-first-verified.json")
            if not target_attestation.exists():
                shutil.copy2(source_attestation, target_attestation)
            if sha256(target_attestation) != sha256(source_attestation):
                raise RuntimeError("guide submission turn-order attestation changed")
            label = (
                "alakazam guide logit A/B iter 21 "
                f"copy {slot}/2 first 261d367e131e NO RTP GUIDE {weight:.2f}"
            )
            identity = (
                "alakazam-guide-logit-ab-r206",
                slot,
                MODEL_SHA256,
            )
            existing = next(
                (
                    row
                    for row in entries
                    if (
                        str(row.get("gate_id") or ""),
                        int(row.get("copy_number") or -1),
                        str(row.get("checkpoint_checksum") or ""),
                    )
                    == identity
                ),
                None,
            )
            expected = {
                "specialist_id": "alakazam",
                "copy_number": slot,
                "turn_order_preference": "first_if_allowed",
                "label": label,
                "checkpoint_checksum": MODEL_SHA256,
                "model_checksum": MODEL_SHA256,
                "deck_file_checksum": DECK_SHA256,
                "deck_cards_checksum": DECK_CARDS_SHA256,
                "representatives_checksum": REPRESENTATIVES_SHA256,
                "matchup_tree_checksum": MATCHUP_TREE_SHA256,
                "search_config_checksum": SEARCH_CONFIG_SHA256,
                "belief_decks_checksum": BELIEF_DECKS_SHA256,
                "gate_id": "alakazam-guide-logit-ab-r206",
                "iteration": 21,
                "queued_at": queued_at,
                "queue_status": "pending",
                "competition": "pokemon-tcg-ai-battle",
                "file": str(target),
                "file_sha256": str(variant["sha256"]),
                "rtp_mode": "disabled",
                "guide_decision_mode": "guide_logit_bonus",
                "guide_logit_weight": weight,
                "owner_decision_revision": 206,
                "owner_authorization_source": "GOAL.md#/decision-ledger/revision-206",
                "retry_count": 0,
                "submitted_at": None,
                "submission_id": None,
                "returned_score": None,
                "failure_reason": None,
            }
            if existing is not None:
                immutable = set(expected).difference(
                    {
                        "queued_at",
                        "queue_status",
                        "submitted_at",
                        "submission_id",
                        "returned_score",
                        "failure_reason",
                        "retry_count",
                    }
                )
                if any(existing.get(key) != expected.get(key) for key in immutable):
                    raise RuntimeError("existing r206 queue identity changed")
                selected.append(existing)
            else:
                entries.append(expected)
                selected.append(expected)
        payload["queue"] = entries
        payload["updated_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_json(queue_path, payload)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--helper-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    for name in ("base_bundle", "helper_source", "output_root", "queue", "receipt"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if sha256(args.base_bundle) != BASE_BUNDLE_SHA256:
        raise RuntimeError("exact r195 NO-RTP base bundle digest changed")
    args.output_root.mkdir(parents=True, exist_ok=True)
    variants = [
        derive_bundle(
            base_bundle=args.base_bundle,
            helper_source=args.helper_source,
            output=args.output_root / f"build-guide-{weight:.2f}" / "submission.tar.gz",
            weight=weight,
        )
        for weight in (0.05, 0.10)
    ]
    queued = queue_variants(
        queue_path=args.queue,
        variants=variants,
        output_root=args.output_root,
    )
    receipt = {
        "schema": SCHEMA,
        "status": "queued",
        "owner_decision_revision": 206,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_bundle": str(args.base_bundle),
        "base_bundle_sha256": BASE_BUNDLE_SHA256,
        "model_sha256": MODEL_SHA256,
        "rtp_enabled": False,
        "variants": variants,
        "queue_entries": queued,
    }
    atomic_json(args.receipt, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
