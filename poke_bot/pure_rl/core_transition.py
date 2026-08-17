"""Exact-heldout policy for freezing the reusable deck-agnostic core."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model_registry import sha256


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _validate_exact(
    *,
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    required_games: int,
    verify_bytes: bool,
) -> dict[str, Any]:
    audit = evidence.get("audit")
    digest = str(candidate.get("digest") or evidence.get("checkpoint_digest") or "")
    checkpoint = Path(
        str(candidate.get("path") or evidence.get("checkpoint") or "")
    ).expanduser()
    games = int(evidence.get("games", 0))
    if not isinstance(audit, dict):
        raise ValueError("heldout audit missing")
    if (
        games != int(required_games)
        or int(audit.get("valid_games", 0)) != games
        or audit.get("passed") is not True
        or audit.get("exact_distribution") is not True
        or audit.get("exact_weights") is not True
        or audit.get("greedy_required") is not True
        or str(audit.get("checkpoint_digest") or "") != digest
        or str(evidence.get("checkpoint_digest") or digest) != digest
    ):
        raise ValueError("heldout evidence is not an exact audited wave")
    if verify_bytes:
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file() or sha256(checkpoint) != digest:
            raise ValueError("heldout checkpoint bytes do not match their identity")
    return {
        "iteration": int(evidence.get("iteration", -1)),
        "checkpoint": str(checkpoint),
        "checkpoint_digest": digest,
        "games": games,
        "win_rate": float(evidence.get("win_rate", evidence.get("wr", -1.0))),
        "confidence_lower": float(evidence.get("confidence_lower", -1.0)),
        "confidence_upper": float(evidence.get("confidence_upper", -1.0)),
        "audit": json.loads(json.dumps(audit)),
        "per_opponent": json.loads(json.dumps(evidence.get("per_opponent") or {})),
    }


def inherited_anchor(
    run_dir: Path, *, required_games: int = 1000, verify_bytes: bool = True
) -> dict[str, Any]:
    """Return the exact inherited champion present when the watcher is armed."""
    run_dir = Path(run_dir)
    handoff = _read_json(run_dir / "lineage_handoff.json")
    inherited = handoff.get("inherited_official_heldout")
    if isinstance(inherited, dict):
        candidate = {
            "path": inherited.get("checkpoint"),
            "digest": inherited.get("checkpoint_digest"),
        }
        evidence = {
            **inherited,
            "checkpoint_digest": inherited.get("checkpoint_digest"),
            "win_rate": inherited.get("wr"),
            "iteration": inherited.get("lineage_iteration", -1),
        }
        return _validate_exact(
            candidate=candidate,
            evidence=evidence,
            required_games=required_games,
            verify_bytes=verify_bytes,
        )
    loop = _read_json(run_dir / "loop_state.json")
    candidate = dict(loop.get("heldout_champion") or {})
    evidence = dict(loop.get("heldout_champion_evidence") or {})
    if not candidate or not evidence:
        raise ValueError("lineage has no exact heldout anchor")
    return _validate_exact(
        candidate=candidate,
        evidence=evidence,
        required_games=required_games,
        verify_bytes=verify_bytes,
    )


def committed_observations(
    run_dir: Path,
    *,
    start_iteration: int,
    required_games: int = 1000,
    verify_bytes: bool = False,
) -> list[dict[str, Any]]:
    """Read only exact evals with a matching immutable iteration commit."""
    run_dir = Path(run_dir)
    observations: list[dict[str, Any]] = []
    for eval_path in sorted((run_dir / "eval").glob("iter_*.json")):
        payload = _read_json(eval_path)
        iteration = int(payload.get("iteration", -1))
        if iteration < int(start_iteration):
            continue
        commit_path = run_dir / "commits" / f"iter_{iteration:05d}.json"
        commit = _read_json(commit_path)
        if int(commit.get("last_completed_iteration", -1)) < iteration:
            continue
        candidate = dict(payload.get("heldout_candidate") or {})
        gate = dict(payload.get("raw_heldout_gate") or {})
        audit = dict(payload.get("heldout_audit") or {})
        evidence = {
            "iteration": iteration,
            "checkpoint_digest": candidate.get("digest"),
            "games": gate.get("games"),
            "win_rate": gate.get("win_rate"),
            "confidence_lower": gate.get("confidence_lower"),
            "confidence_upper": gate.get("confidence_upper"),
            "per_opponent": gate.get("per_opponent"),
            "audit": audit,
        }
        try:
            observation = _validate_exact(
                candidate=candidate,
                evidence=evidence,
                required_games=required_games,
                verify_bytes=verify_bytes,
            )
        except ValueError:
            # A partial/broken formal wave is not evidence and cannot advance
            # either the threshold or the patience counter.
            continue
        observation["eval_path"] = str(eval_path)
        observation["commit_path"] = str(commit_path)
        observations.append(observation)
    observations.sort(key=lambda row: int(row["iteration"]))
    return observations


def transition_decision(
    run_dir: Path,
    *,
    anchor: dict[str, Any],
    start_iteration: int,
    threshold_wr: float = 0.40,
    plateau_patience: int = 10,
    minimum_improvement: float = 0.0,
    required_games: int = 1000,
    verify_best_bytes: bool = True,
) -> dict[str, Any]:
    """Trigger at target WR or after N exact iterations without a new best."""
    if int(plateau_patience) <= 0:
        raise ValueError("plateau patience must be positive")
    best = json.loads(json.dumps(anchor))
    best_wr = float(best["win_rate"])
    streak = 0
    trigger: str | None = None
    observations = committed_observations(
        run_dir,
        start_iteration=int(start_iteration),
        required_games=int(required_games),
        verify_bytes=False,
    )
    for observation in observations:
        wr = float(observation["win_rate"])
        if wr > best_wr + float(minimum_improvement):
            best = observation
            best_wr = wr
            streak = 0
        else:
            streak += 1
        if wr >= float(threshold_wr):
            trigger = "target_win_rate_reached"
            # With equal-sized exact waves, the threshold observation is at
            # least as strong as every lower-WR prior observation.
            if wr >= best_wr:
                best = observation
                best_wr = wr
            break
        if streak >= int(plateau_patience):
            trigger = "no_new_best_for_patience"
            break
    if trigger and verify_best_bytes:
        checkpoint = Path(str(best["checkpoint"])).expanduser().resolve()
        if not checkpoint.is_file() or sha256(checkpoint) != best["checkpoint_digest"]:
            raise ValueError("selected best checkpoint bytes failed verification")
        best["checkpoint"] = str(checkpoint)
    return {
        "triggered": trigger is not None,
        "reason": trigger or "watching",
        "threshold_wr": float(threshold_wr),
        "plateau_patience": int(plateau_patience),
        "minimum_improvement": float(minimum_improvement),
        "start_iteration": int(start_iteration),
        "exact_iterations_observed": len(observations),
        "non_improving_streak": streak,
        "latest_iteration": (
            int(observations[-1]["iteration"]) if observations else None
        ),
        "anchor": json.loads(json.dumps(anchor)),
        "best": best,
        "observations": observations,
    }
