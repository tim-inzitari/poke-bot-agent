#!/usr/bin/env python3
"""Stage abandoned Crustle H10 iter4 as Slop Box S-tier public/holdout opponent.

Owner design (GOAL revision 172): eligible non-active H10 specialists are tier S
weight 2.0 in Slop Box strong-public / formal holdout. Prefer the latest committed
RL checkpoint with a valid submission bundle (iter_00004), not incomplete iter5.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any

from poke_bot.baselines_runtime import baseline_content_digest
from poke_bot.pure_rl.model_registry import sha256


CHECKPOINT_DIGEST = (
    "sha256:7efd8d4113e736d28576bdbfa1c9d1c3f3a7cf1a31a0b3cfadd1e7f82cf08955"
)
BUNDLE_DIGEST = (
    "sha256:3a380d6bd723866911d2e99e9239c679baedfdbb3ba21f27a8f2d522f7738a90"
)
COMMIT_DIGEST = (
    "sha256:8817ad8592df90ff75eca919a1c8a8c0e838f1944d828ec6a2d885e2c3b38ba4"
)
OPPONENT_ID = "specialist-crustle-final-format-h10-7efd8d4113e7"
BASELINE_DIR = "crustle-final-format-h10-7efd8d4113e7"
SPECIALIST_ID = "crustle"
NEW_GATE_ID = (
    "specialist-strong-public-roster-sw80-at-iter5-v1+h10-s-other-a-r111"
    "+marnie-h10-r163+lucario-a-r168+crustle-h10-s-r172"
)
OWNER_REVISION = 172

DEFAULT_BUNDLE = Path(
    "/home/inzi/poke-bot-agent/outputs/submissions/"
    "final-format-crustle-milestones-r167/iter-00004/build/submission.tar.gz"
)
DEFAULT_MILESTONE = Path(
    "/home/inzi/poke-bot-agent/outputs/state/crustle-iter4-kaggle-milestone-r167.json"
)
DEFAULT_COMMIT = Path(
    "/home/inzi/poke-bot-agent/outputs/pure_rl/"
    "final_format_crustle_r113_h10_i_v6_8k/commits/iter_00004.json"
)
DEFAULT_RUNTIME_ROOT = Path(
    "/home/inzi/poke-bot-agent-deployments/final-format-marnie-postupload-r136"
)
DEFAULT_SOURCE_GATE = DEFAULT_RUNTIME_ROOT / (
    "runtime/final_format_crustle_gate_r168_lucario_a.json"
)
DEFAULT_SOURCE_FROZEN = DEFAULT_RUNTIME_ROOT / (
    "ops/frozen_specialist_registry_crustle_r168_lucario_a.json"
)
DEFAULT_SLOP_REGISTRY = Path(
    "/home/inzi/poke-bot-agent/outputs/final_format_slop_box_h10_rtp/runtime/"
    "specialist_runtime_registry_h10_r171.json"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON object required: {path}")
    return value


def _canonical_digest(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _extract_bundle(bundle: Path, destination: Path) -> str:
    if not bundle.is_file() or sha256(bundle) != BUNDLE_DIGEST:
        raise RuntimeError("Crustle H10 iter4 bundle missing or digest mismatch")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{BASELINE_DIR}.", dir=str(destination.parent))
    )
    try:
        with tarfile.open(bundle, "r:gz") as archive:
            for member in archive.getmembers():
                member_path = PurePosixPath(member.name)
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or member.issym()
                    or member.islnk()
                    or member.isdev()
                ):
                    raise RuntimeError("unsafe member in Crustle H10 bundle")
            archive.extractall(temporary)
        for name in ("main.py", "model.pt", "deck.csv", "matchup_tree.json"):
            if not (temporary / name).is_file():
                raise RuntimeError(f"Crustle H10 package missing {name}")
        if sha256(temporary / "model.pt") != CHECKPOINT_DIGEST:
            raise RuntimeError("Crustle H10 package has the wrong checkpoint")
        content_digest = baseline_content_digest(temporary)
        if destination.exists():
            if baseline_content_digest(destination) != content_digest:
                raise RuntimeError("existing Crustle H10 package has different bytes")
            shutil.rmtree(temporary)
        else:
            os.replace(temporary, destination)
        return content_digest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _install_package_copy(source: Path, destination: Path, content_digest: str) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = baseline_content_digest(destination)
        return {
            "path": str(destination),
            "exists": True,
            "content_digest": actual,
            "matches_canonical": actual == content_digest,
        }
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=str(destination.parent))
    )
    try:
        shutil.copytree(source, temporary, dirs_exist_ok=True)
        if baseline_content_digest(temporary) != content_digest:
            raise RuntimeError(f"install digest mismatch for {destination}")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return {
        "path": str(destination),
        "exists": True,
        "content_digest": content_digest,
        "matches_canonical": True,
    }


def _update_manifest(path: Path) -> None:
    if not path.is_file():
        return
    manifest = _read_json(path)
    agents = [
        copy.deepcopy(row)
        for row in (manifest.get("agents") or [])
        if str(row.get("id") or "") != OPPONENT_ID
    ]
    agents.append(
        {
            "id": OPPONENT_ID,
            "name": "Frozen final-format H10 Crustle (iter4 holdout; abandoned mid-iter5)",
            "group": "specialists",
            "dir": BASELINE_DIR,
            "source": (
                "checksum-bound Crustle H10 iter_00004 milestone bundle "
                + BUNDLE_DIGEST
            ),
        }
    )
    ids = [str(row.get("id") or "") for row in agents]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"baseline manifest duplicate ids: {path}")
    manifest["agents"] = agents
    notes = dict(manifest.get("field_notes") or {})
    notes["total"] = len(agents)
    manifest["field_notes"] = notes
    _atomic_json(path, manifest)


def _build_frozen(
    *,
    source: dict[str, Any],
    package: Path,
    content_digest: str,
    milestone: Path,
) -> dict[str, Any]:
    if source.get("schema") != "poke_bot.frozen_specialist_registry/v1":
        raise RuntimeError("source frozen registry schema invalid")
    rows = [
        copy.deepcopy(row)
        for row in (source.get("specialists") or [])
        if str(row.get("opponent_id") or "") != OPPONENT_ID
    ]
    if any(str(row.get("specialist_id") or "") == SPECIALIST_ID for row in rows):
        # Abandoned Crustle was never registered as a frozen opponent row.
        # Keep any unexpected historical crustle specialist rows and add H10.
        pass
    if len(rows) != 15:
        raise RuntimeError(f"expected 15 parent frozen rows, got {len(rows)}")
    # Mirror Alakazam/Marnie H10 frozen row shape from parent Alakazam row.
    template = next(
        row
        for row in rows
        if row.get("opponent_id")
        == "specialist-alakazam-final-format-h10-02c014ad7c33"
    )
    new_row = {
        **copy.deepcopy(template),
        "archetype_id": SPECIALIST_ID,
        "archetype_label": (
            "Frozen final-format H10 Crustle iter4 (abandoned; holdout only)"
        ),
        "baseline_dir": BASELINE_DIR,
        "baseline_group": "specialists",
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "content_digest": content_digest,
        "deck_file_checksum": sha256(package / "deck.csv"),
        "final_format_h10_refresh": True,
        "frozen": True,
        "kaggle_submission_eligible": False,
        "matchup_runtime_enabled": True,
        "matchup_runtime_inference_only": True,
        "matchup_tree_checksum": sha256(package / "matchup_tree.json"),
        "opponent_id": OPPONENT_ID,
        "public_mix_eligible": True,
        "refresh_completion_receipt": str(milestone),
        "refresh_completion_receipt_sha256": sha256(milestone),
        "registered_at_utc": _now(),
        "research_eligible": False,
        "roster_version": 6,
        "source": (
            "owner-ordered Slop Box S-tier holdout from committed Crustle "
            "iter_00004 milestone; incomplete iter5 quarantine unused"
        ),
        "source_passing_checkpoint_digest": CHECKPOINT_DIGEST,
        "specialist_id": SPECIALIST_ID,
        "crustle_abandon_preserved": True,
        "crustle_iter5_quarantine_unused": True,
        "owner_decision_revision": OWNER_REVISION,
    }
    # Remove Alakazam-only supersession key if copied.
    new_row.pop("supersedes_for_marnie_opponent_id", None)
    new_row.pop("supersedes_for_crustle_opponent_id", None)
    rows.append(new_row)
    if len(rows) != 16:
        raise RuntimeError("frozen roster must expand 15→16")
    result = copy.deepcopy(source)
    result["specialists"] = rows
    result["version"] = max(16, int(result.get("version") or 0) + 1)
    result["updated_at_utc"] = _now()
    result["owner_decision_revision"] = OWNER_REVISION
    result["scope"] = "final-format-slop-box-h10-rtp-crustle-s-r172"
    result["crustle_h10_s_tier_holdout"] = {
        "opponent_id": OPPONENT_ID,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "bundle_digest": BUNDLE_DIGEST,
        "content_digest": content_digest,
        "tier": "S",
        "weight": 2.0,
    }
    return result


def _build_gate(
    *,
    source: dict[str, Any],
    content_digest: str,
) -> dict[str, Any]:
    gate = copy.deepcopy(source)
    next_gate = dict(gate.get("next_gate") or {})
    roster = [copy.deepcopy(row) for row in (next_gate.get("roster") or [])]
    if len(roster) != 19:
        raise RuntimeError(f"expected parent r168 roster size 19, got {len(roster)}")
    if any(row.get("opponent_id") == OPPONENT_ID for row in roster):
        raise RuntimeError("Crustle H10 opponent already present in parent gate")
    roster.append(
        {
            "archetype_id": SPECIALIST_ID,
            "archetype_label": (
                "Frozen final-format H10 Crustle iter4 (abandoned; holdout only)"
            ),
            "content_digest": content_digest,
            "frozen_checkpoint_digest": CHECKPOINT_DIGEST,
            "frozen_specialist": True,
            "opponent_id": OPPONENT_ID,
            "owner_decision_revision": OWNER_REVISION,
            "source": (
                "owner-ordered Slop Box S-tier holdout from committed Crustle "
                "iter_00004 milestone"
            ),
            "tier": "S",
            "weight": 2.0,
        }
    )
    evaluation = dict(next_gate.get("evaluation") or {})
    evaluation["games_total"] = 250 * len(roster)
    evaluation["games_per_opponent"] = 250
    evaluation["seat0_games_per_opponent"] = 125
    evaluation["seat1_games_per_opponent"] = 125
    next_gate["evaluation"] = evaluation
    next_gate["roster"] = roster
    next_gate["id"] = NEW_GATE_ID
    next_gate["label"] = (
        "Strong public/frozen A roster + non-active H10 S including Crustle "
        "(revision 172)"
    )
    gate["next_gate"] = next_gate
    gate["active_gate_id"] = NEW_GATE_ID
    gate["owner_decision_revision"] = OWNER_REVISION
    semantics = dict(gate.get("active_gate_semantics") or {})
    semantics["base_premium_agents"] = 4
    semantics["frozen_specialist_agents"] = 16
    semantics["gate_roster_size"] = 20
    semantics["gate_games_total"] = 5000
    semantics["games_per_opponent"] = 250
    semantics["opponent_tier_policy"] = {
        "eligible_non_active_h10_specialist": {"tier": "S", "weight": 2.0},
        "other_frozen_specialist": {"tier": "A", "weight": 1.0},
        "remaining_public_opponent": {"tier": "A", "weight": 1.0},
    }
    semantics["crustle_h10_s_tier_owner_revision"] = OWNER_REVISION
    semantics["invariant"] = (
        "Checksum roster expands r168's 19 opponents with abandoned Crustle H10 "
        f"{OPPONENT_ID} as tier S/2.0 (checkpoint {CHECKPOINT_DIGEST}); "
        "20 opponents at 250 greedy games each with 125/125 seats (5000 formal). "
        "Eligible non-active H10 specialists remain S/2.0; other frozen/public A/1.0. "
        "Incomplete iter5 quarantine unused; no Crustle deletes."
    )
    gate["active_gate_semantics"] = semantics
    gate["derivation"] = {
        **dict(gate.get("derivation") or {}),
        "owner_decision_revision": OWNER_REVISION,
        "source_gate_id": str(source.get("active_gate_id") or ""),
        "added_opponent_id": OPPONENT_ID,
        "parent_roster_size": 19,
        "new_roster_size": 20,
        "formal_games_total": 5000,
        "incomplete_iter5_quarantine_unused": True,
    }
    return gate


def _rebind_slop_registry(
    registry_path: Path,
    *,
    gate_rel: str,
    frozen_rel: str,
) -> dict[str, Any] | None:
    if not registry_path.is_file():
        return None
    registry = _read_json(registry_path)
    registry["active_gate_contract"] = gate_rel
    registry["frozen_specialist_registry"] = frozen_rel
    registry["terminal_active_gate_id"] = NEW_GATE_ID
    registry["owner_decision_revision"] = OWNER_REVISION
    registry["opponent_tier_policy"] = {
        "owner_decision_revision": OWNER_REVISION,
        "scope": ["public_mix", "formal_premium_holdout"],
        "eligible_non_active_h10_specialist": {"tier": "S", "weight": 2.0},
        "other_frozen_specialist": {"tier": "A", "weight": 1.0},
        "remaining_public_opponent": {"tier": "A", "weight": 1.0},
        "crustle_h10_s_tier_holdout": {
            "opponent_id": OPPONENT_ID,
            "tier": "S",
            "weight": 2.0,
            "checkpoint_digest": CHECKPOINT_DIGEST,
        },
    }
    _atomic_json(registry_path, registry)
    return registry


def _ssh_base() -> list[str]:
    key = Path.home() / ".ssh/id_ed25519_poke_lan"
    base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
    if key.is_file():
        base.extend(["-o", "IdentitiesOnly=yes", "-i", str(key)])
    return base


def _rsync_ssh() -> str:
    key = Path.home() / ".ssh/id_ed25519_poke_lan"
    if key.is_file():
        return (
            "ssh -o BatchMode=yes -o ConnectTimeout=15 "
            f"-o IdentitiesOnly=yes -i {key}"
        )
    return "ssh -o BatchMode=yes -o ConnectTimeout=15"


def _remote_model_digest(host: str, remote_model: str) -> str:
    out = subprocess.run(
        [*_ssh_base(), host, f"sha256sum {remote_model}"],
        check=True,
        capture_output=True,
        text=True,
    )
    digest = out.stdout.strip().split()[0]
    return "sha256:" + digest


def _remote_install(local_package: Path, content_digest: str) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    targets = [
        (
            "bert",
            "/Users/tsinzitari/workspace/poke-bot-agent/baselines/specialists/"
            + BASELINE_DIR,
        ),
        (
            "bert",
            "/Users/tsinzitari/workspace/poke-bot-agent-h10-r79-stage/baselines/"
            "specialists/"
            + BASELINE_DIR,
        ),
        (
            "elmo",
            f"/mnt/Main/Elmo/poke-bot-agent/baselines/specialists/{BASELINE_DIR}",
        ),
        (
            "elmo",
            f"/mnt/Main/main/poke-bot-agent/baselines/specialists/{BASELINE_DIR}",
        ),
    ]
    for host, remote in targets:
        try:
            subprocess.run(
                [*_ssh_base(), host, f"mkdir -p {remote}"],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--delete",
                    "-e",
                    _rsync_ssh(),
                    str(local_package) + "/",
                    f"{host}:{remote}/",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            model_digest = _remote_model_digest(host, f"{remote}/model.pt")
            main_ok = (
                subprocess.run(
                    [*_ssh_base(), host, f"test -f {remote}/main.py"],
                    check=False,
                    capture_output=True,
                    text=True,
                ).returncode
                == 0
            )
            checks.append(
                {
                    "host": host,
                    "path": remote,
                    "model_sha256": model_digest,
                    "model_matches": model_digest == CHECKPOINT_DIGEST,
                    "main_py_present": main_ok,
                    "canonical_content_digest": content_digest,
                    "matches_canonical": model_digest == CHECKPOINT_DIGEST and main_ok,
                }
            )
        except Exception as exc:  # noqa: BLE001 - record remote failure, do not abort
            checks.append(
                {
                    "host": host,
                    "path": remote,
                    "content_digest": None,
                    "matches_canonical": False,
                    "error": str(exc)[:400],
                }
            )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--milestone", type=Path, default=DEFAULT_MILESTONE)
    parser.add_argument("--commit", type=Path, default=DEFAULT_COMMIT)
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--source-gate", type=Path, default=DEFAULT_SOURCE_GATE)
    parser.add_argument("--source-frozen", type=Path, default=DEFAULT_SOURCE_FROZEN)
    parser.add_argument("--slop-registry", type=Path, default=DEFAULT_SLOP_REGISTRY)
    parser.add_argument(
        "--receipt",
        type=Path,
        default=Path(
            "/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-crustle-h10-s-tier-holdout-r172.json"
        ),
    )
    parser.add_argument(
        "--repo-receipt",
        type=Path,
        default=Path("state/slop-box-crustle-h10-s-tier-holdout-r172.json"),
    )
    parser.add_argument("--skip-remote-install", action="store_true")
    args = parser.parse_args()

    runtime_root = args.runtime_root.resolve()
    bundle = args.bundle.resolve()
    milestone = args.milestone.resolve()
    commit_path = args.commit.resolve()
    source_gate_path = args.source_gate.resolve()
    source_frozen_path = args.source_frozen.resolve()

    milestone_payload = _read_json(milestone)
    if (
        milestone_payload.get("checkpoint_sha256") != CHECKPOINT_DIGEST
        or milestone_payload.get("bundle_sha256") != BUNDLE_DIGEST
        or milestone_payload.get("commit_sha256") != COMMIT_DIGEST
    ):
        raise RuntimeError("Crustle iter4 milestone identity mismatch")
    if sha256(commit_path) != COMMIT_DIGEST:
        raise RuntimeError("Crustle iter4 commit digest mismatch")
    if sha256(source_gate_path) != (
        "sha256:611c67e6d4db1ae4995c307ec65180f49b4b2a9299c57c9eedcd66c7e8f1580a"
    ):
        # Allow if parent was already rebound; still require 19-opponent r168 shape.
        parent = _read_json(source_gate_path)
        if len((parent.get("next_gate") or {}).get("roster") or []) != 19:
            raise RuntimeError("source gate is not the expected 19-opponent r168 parent")

    package = runtime_root / "baselines" / "specialists" / BASELINE_DIR
    content_digest = _extract_bundle(bundle, package)

    local_installs = [
        _install_package_copy(
            package,
            Path("/home/inzi/poke-bot-agent/baselines/specialists") / BASELINE_DIR,
            content_digest,
        ),
        _install_package_copy(
            package,
            Path(
                "/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v9/"
                "baselines/specialists"
            )
            / BASELINE_DIR,
            content_digest,
        ),
        {
            "path": str(package),
            "exists": True,
            "content_digest": content_digest,
            "matches_canonical": True,
        },
    ]
    for manifest in (
        runtime_root / "baselines" / "manifest.json",
        Path("/home/inzi/poke-bot-agent/baselines/manifest.json"),
        Path(
            "/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v9/"
            "baselines/manifest.json"
        ),
    ):
        _update_manifest(manifest)

    frozen = _build_frozen(
        source=_read_json(source_frozen_path),
        package=package,
        content_digest=content_digest,
        milestone=milestone,
    )
    gate = _build_gate(source=_read_json(source_gate_path), content_digest=content_digest)

    gate_rel = "runtime/final_format_slop_box_gate_r172_crustle_h10_s.json"
    frozen_rel = "ops/frozen_specialist_registry_slop_box_r172_crustle_h10_s.json"
    gate_out = runtime_root / gate_rel
    frozen_out = runtime_root / frozen_rel
    _atomic_json(gate_out, gate)
    _atomic_json(frozen_out, frozen)

    # Validate gate/frozen contract locally using the same invariants.
    roster = gate["next_gate"]["roster"]
    frozen_rows = [
        row
        for row in frozen["specialists"]
        if row.get("frozen") is True and row.get("public_mix_eligible") is True
    ]
    if len(roster) != 20 or len(frozen_rows) != 16:
        raise RuntimeError(
            f"roster/frozen size mismatch: roster={len(roster)} frozen={len(frozen_rows)}"
        )
    if gate["next_gate"]["evaluation"]["games_total"] != 5000:
        raise RuntimeError("formal games_total must be 5000")
    crustle_gate = next(row for row in roster if row["opponent_id"] == OPPONENT_ID)
    crustle_frozen = next(
        row for row in frozen_rows if row["opponent_id"] == OPPONENT_ID
    )
    if (crustle_gate.get("tier"), float(crustle_gate.get("weight"))) != ("S", 2.0):
        raise RuntimeError("Crustle gate tier is not S/2.0")
    if crustle_frozen.get("final_format_h10_refresh") is not True:
        raise RuntimeError("Crustle frozen row must mark final_format_h10_refresh")
    if crustle_frozen.get("checkpoint_digest") != CHECKPOINT_DIGEST:
        raise RuntimeError("Crustle frozen checkpoint mismatch")

    rebound = _rebind_slop_registry(
        args.slop_registry.resolve(),
        gate_rel=gate_rel,
        frozen_rel=frozen_rel,
    )

    remote_checks: list[dict[str, Any]] = []
    if not args.skip_remote_install:
        remote_checks = _remote_install(package, content_digest)

    s_tiers = [
        (row.get("opponent_id"), row.get("tier"), row.get("weight"))
        for row in roster
        if row.get("tier") == "S"
    ]
    receipt = {
        "schema": "poke_bot.slop_box_crustle_h10_s_tier_holdout_r172/v1",
        "status": "staged_and_bound_for_slop_box_rl",
        "owner_goal_revision": OWNER_REVISION,
        "created_at_utc": _now(),
        "opponent_id": OPPONENT_ID,
        "specialist_id": SPECIALIST_ID,
        "tier": "S",
        "weight": 2.0,
        "checkpoint_digest": CHECKPOINT_DIGEST,
        "bundle_digest": BUNDLE_DIGEST,
        "commit_sha256": COMMIT_DIGEST,
        "baseline_dir": BASELINE_DIR,
        "baseline_content_digest": content_digest,
        "package_path": str(package),
        "source_package_choice": {
            "selected": "committed_rl_iter_00004_milestone_bundle",
            "rejected_incomplete_iter5_quarantine": True,
            "bootstrap_available_but_not_selected": (
                "sha256:73443f9544471b25c13f5833847471f4d311b5640af3b2c6d6d3bef6ba2fdf95"
            ),
            "reason": (
                "prefer latest committed RL checkpoint with valid checksum-bound "
                "submission bundle over bootstrap; never use incomplete iter5"
            ),
        },
        "parent_gate": str(source_gate_path),
        "parent_gate_sha256": sha256(source_gate_path),
        "parent_frozen": str(source_frozen_path),
        "parent_frozen_sha256": sha256(source_frozen_path),
        "staged_gate": str(gate_out),
        "staged_gate_sha256": sha256(gate_out),
        "staged_frozen_registry": str(frozen_out),
        "staged_frozen_registry_sha256": sha256(frozen_out),
        "gate_id": NEW_GATE_ID,
        "roster": {
            "previous_size": 19,
            "new_size": 20,
            "games_per_opponent": 250,
            "seat0_games_per_opponent": 125,
            "seat1_games_per_opponent": 125,
            "previous_gate_games_total": 4750,
            "new_gate_games_total": 5000,
            "s_tier_opponents": s_tiers,
        },
        "slop_box_runtime_registry": (
            str(args.slop_registry.resolve()) if rebound is not None else None
        ),
        "slop_box_runtime_registry_sha256": (
            sha256(args.slop_registry.resolve()) if rebound is not None else None
        ),
        "slop_box_registry_rebound": rebound is not None,
        "live_for_next_slop_box_collect": rebound is not None,
        "package_install_checks": local_installs + remote_checks,
        "constraints_honored": [
            "no_crustle_game_or_quarantine_deletion",
            "no_incomplete_iter5_package",
            "no_chao_hard_ce_disruption",
            "systemd_launchd_only_no_interactive_kills",
            "coordinate_with_ceiling_register_before_rl_collect",
        ],
        "coordination_note": (
            "1f90a2c3 ceiling register failed on guide loss weight; Crustle S-tier "
            "gate/frozen rebound into specialist_runtime_registry_h10_r171.json so "
            "the next successful register/RL start inherits the 20-opponent roster."
        ),
    }
    receipt["receipt_payload_digest"] = _canonical_digest(
        {k: v for k, v in receipt.items() if k != "receipt_payload_digest"}
    )
    _atomic_json(args.receipt.resolve(), receipt)
    repo_receipt = args.repo_receipt
    if not repo_receipt.is_absolute():
        repo_receipt = (Path.cwd() / repo_receipt).resolve()
    try:
        _atomic_json(repo_receipt, receipt)
    except OSError:
        pass
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
