#!/usr/bin/env python3
"""Publish a low-overhead live win rate for the current public-mix shard.

The training shard already records one retained trajectory per public-baseline
game with the acting policy's terminal value.  This watcher tails that immutable
stream and writes a tiny dashboard sidecar.  It is deliberately separate from
the heldout gate: public-mix actions are sampled training behavior, not greedy
evaluation evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any


RUN_NAME_RE = re.compile(r"(?:^|\s)--run-name(?:=|\s+)([^\s]+)")
ITERATION_RE = re.compile(r"pure_rl\s+(\S+)\s+iter=(\d+):")
COUNT_RE = re.compile(r"\s(\d+)/(\d+)\s+\[")
VALUE_RE = re.compile(rb'"value":(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)')
OPPONENT_RE = re.compile(rb'"opp_archetype":"([^"\\]+)"')
TARGET_OPPONENT_RE = re.compile(rb'"opponent_id":"([^"\\]+)"')
SEAT_RE = re.compile(rb'"seat":([01])')
CHECKPOINT_RE = re.compile(rb'"behavior_checkpoint_digest":"([^"\\]+)"')
PUBLIC_MARKER = b'"collect":"public_mix"'
STRONG_PUBLIC_PRACTICE_MARKER = b'"collect":"strong_public_practice"'
RESEARCH_CONTROL_MARKER = b'"collect":"research_controls"'
DECISIONS_MARKER = b',"decisions":['
SCHEMA = "poke_bot.public_mix_live_winrate/v6"
STRONG_PUBLIC_PRACTICE_SCHEMA = (
    "poke_bot.strong_public_practice_live_winrate/v1"
)
RESEARCH_SCHEMA = "poke_bot.research_controls_live_winrate/v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _active_run_name(service: str) -> str | None:
    try:
        pid = subprocess.run(
            ["systemctl", "--user", "show", service, "-p", "MainPID", "--value"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).stdout.strip()
        if not pid.isdigit() or int(pid) <= 0:
            return None
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", "replace"
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = RUN_NAME_RE.search(command)
    return match.group(1).strip("'\"") if match else None


def _current_stage(root: Path, run_name: str, loop: dict[str, Any]) -> tuple[str, int]:
    status = root / "outputs" / "logs" / f"{run_name}.progress.status"
    try:
        line = status.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        line = ""
    match = ITERATION_RE.search(line)
    if match:
        stage = match.group(1)
        counts = COUNT_RE.search(line)
        if (
            stage in ("collect:research_controls", "collect:public_mix")
            and counts is not None
            and int(counts.group(1)) >= int(counts.group(2))
        ):
            stage = f"{stage}:complete"
        return stage, int(match.group(2))
    return "waiting", int(loop.get("next_iteration") or 0)


def _new_state(
    run_name: str,
    iteration: int,
    shard: Path,
    *,
    last_completed_strong_public_practice: dict[str, Any] | None = None,
    last_completed_research_controls: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "run": run_name,
        "iteration": int(iteration),
        "shard": str(shard),
        "inode": None,
        "offset": 0,
        "games": 0,
        "wins": 0.0,
        "draws": 0,
        "losses": 0,
        "per_opponent": {},
        "checkpoint_digests": {},
        "_strong_public_practice_current": _new_counter(),
        "_strong_public_practice_last": dict(
            last_completed_strong_public_practice or {}
        ),
        "_research_controls_current": _new_counter(),
        "_research_controls_last": dict(
            last_completed_research_controls or {}
        ),
    }


def _new_counter() -> dict[str, Any]:
    return {
        "games": 0,
        "wins": 0.0,
        "draws": 0,
        "losses": 0,
        "per_opponent": {},
        "checkpoint_digests": {},
    }


def _completed_research_counter(
    state: dict[str, Any], *, include_current: bool
) -> dict[str, Any] | None:
    """Return the newest non-empty research counter with its source identity.

    The public sidecar is both the dashboard payload and this watcher's restart
    checkpoint.  Keep the raw current counter separate from the display-only
    ``research_controls`` fallback so a restart in the next iteration cannot
    reinterpret the previous iteration's displayed result as current data.
    """

    displayed = state.get("research_controls")
    displayed_is_prior = (
        isinstance(displayed, dict)
        and displayed.get("iteration") is not None
        and displayed.get("iteration") != state.get("iteration")
    )
    candidates = (
        *(
            (state.get("_research_controls_current"),)
            if include_current
            else ()
        ),
        state.get("_research_controls_last"),
        # v3/early-v4 migration only.  A same-iteration display can be a
        # partial counter, so it is eligible only at a newer iteration boundary.
        displayed if include_current or displayed_is_prior else None,
    )
    for raw in candidates:
        if not isinstance(raw, dict) or int(raw.get("games") or 0) <= 0:
            continue
        return {
            **raw,
            "run": raw.get("run") or state.get("run"),
            "iteration": (
                raw.get("iteration")
                if raw.get("iteration") is not None
                else state.get("iteration")
            ),
            "shard": raw.get("shard") or state.get("shard"),
        }
    return None


def _completed_strong_public_practice_counter(
    state: dict[str, Any], *, include_current: bool
) -> dict[str, Any] | None:
    """Return the newest sampled gate-practice counter and its identity."""

    displayed = state.get("strong_public_practice")
    displayed_is_prior = (
        isinstance(displayed, dict)
        and displayed.get("iteration") is not None
        and displayed.get("iteration") != state.get("iteration")
    )
    candidates = (
        *((state.get("_strong_public_practice_current"),) if include_current else ()),
        state.get("_strong_public_practice_last"),
        displayed if include_current or displayed_is_prior else None,
    )
    for raw in candidates:
        if not isinstance(raw, dict) or int(raw.get("games") or 0) <= 0:
            continue
        return {
            **raw,
            "run": raw.get("run") or state.get("run"),
            "iteration": (
                raw.get("iteration")
                if raw.get("iteration") is not None
                else state.get("iteration")
            ),
            "shard": raw.get("shard") or state.get("shard"),
        }
    return None


def _state_for_iteration(
    state: dict[str, Any],
    *,
    run_name: str,
    iteration: int,
    shard: Path,
) -> dict[str, Any]:
    """Roll the watcher checkpoint to a new shard without mixing counters."""

    if (
        state.get("run") == run_name
        and state.get("iteration") == int(iteration)
        and state.get("shard") == str(shard)
    ):
        return state
    previous_iteration = state.get("iteration")
    strictly_newer_iteration = (
        state.get("run") == run_name
        and isinstance(previous_iteration, int)
        and int(iteration) > previous_iteration
    )
    return _new_state(
        run_name,
        iteration,
        shard,
        last_completed_strong_public_practice=(
            _completed_strong_public_practice_counter(
                state,
                include_current=strictly_newer_iteration,
            )
        ),
        last_completed_research_controls=_completed_research_counter(
            state,
            include_current=strictly_newer_iteration,
        ),
    )


def _record_result(
    counter: dict[str, Any],
    *,
    value: float,
    opponent_id: str,
    seat: int | None,
    checkpoint_digest: str,
) -> None:
    checkpoint_digests = counter.setdefault("checkpoint_digests", {})
    checkpoint_digests[checkpoint_digest] = (
        int(checkpoint_digests.get(checkpoint_digest) or 0) + 1
    )
    per_opponent = counter.setdefault("per_opponent", {})
    opponent = per_opponent.setdefault(
        opponent_id,
        {
            "games": 0,
            "wins": 0.0,
            "draws": 0,
            "losses": 0,
            "seat0": 0,
            "seat1": 0,
        },
    )
    counter["games"] = int(counter.get("games") or 0) + 1
    opponent["games"] = int(opponent.get("games") or 0) + 1
    if seat in (0, 1):
        key = f"seat{seat}"
        opponent[key] = int(opponent.get(key) or 0) + 1
    if value > 0:
        counter["wins"] = float(counter.get("wins") or 0.0) + 1.0
        opponent["wins"] = float(opponent.get("wins") or 0.0) + 1.0
    elif value < 0:
        counter["losses"] = int(counter.get("losses") or 0) + 1
        opponent["losses"] = int(opponent.get("losses") or 0) + 1
    else:
        counter["wins"] = float(counter.get("wins") or 0.0) + 0.5
        counter["draws"] = int(counter.get("draws") or 0) + 1
        opponent["wins"] = float(opponent.get("wins") or 0.0) + 0.5
        opponent["draws"] = int(opponent.get("draws") or 0) + 1


def _consume_available(shard: Path, state: dict[str, Any]) -> int:
    try:
        stat = shard.stat()
    except OSError:
        return 0
    inode = int(stat.st_ino)
    current_research = state.get("_research_controls_current")
    last_research = state.get("_research_controls_last")
    current_practice = state.get("_strong_public_practice_current")
    last_practice = state.get("_strong_public_practice_last")
    if not isinstance(current_practice, dict):
        displayed = state.get("strong_public_practice")
        displayed_iteration = (
            displayed.get("iteration") if isinstance(displayed, dict) else None
        )
        if isinstance(displayed, dict) and displayed_iteration == state.get("iteration"):
            current_practice = {
                key: displayed.get(key, default)
                for key, default in _new_counter().items()
            }
        else:
            current_practice = _new_counter()
            if (
                not isinstance(last_practice, dict)
                and isinstance(displayed, dict)
                and int(displayed.get("games") or 0) > 0
            ):
                last_practice = dict(displayed)
        state["_strong_public_practice_current"] = current_practice
    if not isinstance(last_practice, dict):
        last_practice = {}
        state["_strong_public_practice_last"] = last_practice
    if not isinstance(current_research, dict):
        displayed = state.get("research_controls")
        displayed_iteration = (
            displayed.get("iteration") if isinstance(displayed, dict) else None
        )
        if (
            isinstance(displayed, dict)
            and displayed_iteration == state.get("iteration")
        ):
            current_research = {
                key: displayed.get(key, default)
                for key, default in _new_counter().items()
            }
        else:
            current_research = _new_counter()
            if (
                not isinstance(last_research, dict)
                and isinstance(displayed, dict)
                and int(displayed.get("games") or 0) > 0
            ):
                last_research = dict(displayed)
        state["_research_controls_current"] = current_research
    if not isinstance(last_research, dict):
        last_research = {}
        state["_research_controls_last"] = last_research
    if (
        state.get("schema") != SCHEMA
        or not isinstance(state.get("per_opponent"), dict)
        or not isinstance(state.get("checkpoint_digests"), dict)
        or state.get("inode") != inode
        or int(state.get("offset") or 0) > stat.st_size
    ):
        # An inode change or truncation is recovery within the same iteration,
        # not evidence that its partial research phase completed.  Rebuild the
        # current counter from the replacement shard and retain only the last
        # result already marked complete at an iteration rollover.
        state.update(
            schema=SCHEMA,
            inode=inode,
            offset=0,
            games=0,
            wins=0.0,
            draws=0,
            losses=0,
            per_opponent={},
            checkpoint_digests={},
            _strong_public_practice_current=_new_counter(),
            _research_controls_current=_new_counter(),
        )
    practice_counter = state.get("_strong_public_practice_current")
    if not isinstance(practice_counter, dict):
        practice_counter = _new_counter()
        state["_strong_public_practice_current"] = practice_counter
    research_counter = state.get("_research_controls_current")
    if not isinstance(research_counter, dict):
        research_counter = _new_counter()
        state["_research_controls_current"] = research_counter

    consumed = 0
    with shard.open("rb", buffering=1024 * 1024) as handle:
        handle.seek(int(state.get("offset") or 0))
        while True:
            line_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                handle.seek(line_start)
                break
            state["offset"] = handle.tell()
            # All fields needed here precede the very large decision array.
            prefix = line.split(DECISIONS_MARKER, 1)[0]
            if RESEARCH_CONTROL_MARKER in prefix:
                counter = research_counter
            elif STRONG_PUBLIC_PRACTICE_MARKER in prefix:
                counter = practice_counter
            elif PUBLIC_MARKER in prefix:
                counter = state
            else:
                continue
            match = VALUE_RE.search(prefix)
            if match is None:
                continue
            value = float(match.group(1))
            # ``opp_archetype`` intentionally groups decks and can collapse
            # multiple pinned agents (for example several Lucario packages).
            # Prefer the immutable scheduled opponent identity recorded in
            # target provenance; retain the archetype only as a legacy fallback.
            opponent_match = TARGET_OPPONENT_RE.search(prefix) or OPPONENT_RE.search(
                prefix
            )
            opponent_id = (
                opponent_match.group(1).decode("utf-8", "replace")
                if opponent_match is not None
                else "unknown"
            )
            seat_match = SEAT_RE.search(prefix)
            seat = int(seat_match.group(1)) if seat_match is not None else None
            checkpoint_match = CHECKPOINT_RE.search(prefix)
            checkpoint_digest = (
                checkpoint_match.group(1).decode("utf-8", "replace")
                if checkpoint_match is not None
                else "unknown"
            )
            _record_result(
                counter,
                value=value,
                opponent_id=opponent_id,
                seat=seat,
                checkpoint_digest=checkpoint_digest,
            )
            consumed += 1
    return consumed


def _counter_payload(counter: dict[str, Any]) -> dict[str, Any]:
    games = int(counter.get("games") or 0)
    wins = float(counter.get("wins") or 0.0)
    checkpoint_digests = counter.get("checkpoint_digests") or {}
    known_digests = [
        str(digest)
        for digest in checkpoint_digests
        if str(digest) and str(digest) != "unknown"
    ]
    per_opponent = []
    for opponent_id, raw in sorted((counter.get("per_opponent") or {}).items()):
        if not isinstance(raw, dict):
            continue
        opponent_games = int(raw.get("games") or 0)
        opponent_wins = float(raw.get("wins") or 0.0)
        per_opponent.append(
            {
                "opponent_id": str(opponent_id),
                "games": opponent_games,
                "wins": opponent_wins,
                "draws": int(raw.get("draws") or 0),
                "losses": int(raw.get("losses") or 0),
                "seat0": int(raw.get("seat0") or 0),
                "seat1": int(raw.get("seat1") or 0),
                "win_rate": (
                    opponent_wins / opponent_games if opponent_games else None
                ),
            }
        )
    return {
        "games": games,
        "wins": wins,
        "draws": int(counter.get("draws") or 0),
        "losses": int(counter.get("losses") or 0),
        "per_opponent": dict(counter.get("per_opponent") or {}),
        "checkpoint_digests": dict(checkpoint_digests),
        "matchups": per_opponent,
        "checkpoint_digest": known_digests[0] if len(known_digests) == 1 else None,
        "checkpoint_mixed": len(known_digests) > 1,
        "available": games > 0,
        "win_rate": wins / games if games else None,
    }


def _payload(state: dict[str, Any], *, stage: str, active: bool) -> dict[str, Any]:
    public = _counter_payload(state)
    current_practice = _counter_payload(
        state.get("_strong_public_practice_current")
        if isinstance(state.get("_strong_public_practice_current"), dict)
        else {}
    )
    last_practice_raw = state.get("_strong_public_practice_last")
    last_practice = _counter_payload(
        last_practice_raw if isinstance(last_practice_raw, dict) else {}
    )
    current_practice_result = bool(current_practice["available"])
    practice = current_practice if current_practice_result else last_practice
    practice_iteration = (
        state.get("iteration")
        if current_practice_result
        else (
            last_practice_raw.get("iteration")
            if isinstance(last_practice_raw, dict)
            else None
        )
    )
    practice_active = bool(
        stage == "collect:public_mix" and current_practice_result
    )
    practice_stage = (
        stage
        if practice_active
        else "collect:strong_public_practice:complete"
        if practice["available"]
        else "waiting"
    )
    current_research = _counter_payload(
        state.get("_research_controls_current")
        if isinstance(state.get("_research_controls_current"), dict)
        else {}
    )
    last_research_raw = state.get("_research_controls_last")
    last_research = _counter_payload(
        last_research_raw if isinstance(last_research_raw, dict) else {}
    )
    current_result = bool(current_research["available"])
    research = current_research if current_result else last_research
    research_active = bool(stage == "collect:research_controls" and current_result)
    research_iteration = (
        state.get("iteration")
        if current_result
        else (
            last_research_raw.get("iteration")
            if isinstance(last_research_raw, dict)
            else None
        )
    )
    research_stage = (
        stage
        if current_result and stage.startswith("collect:research_controls")
        else "collect:research_controls:complete"
        if research["available"]
        else "waiting"
    )
    top_state = {
        key: value
        for key, value in state.items()
        if key not in ("research_controls", "strong_public_practice")
    }
    return {
        **top_state,
        **public,
        "schema": SCHEMA,
        "active": bool(active),
        "stage": stage,
        "updated_at": time.time(),
        "definition": (
            "retained public-mix training trajectories; draw=0.5; sampled "
            "behavior policy; non-gate"
        ),
        "strong_public_practice": {
            **practice,
            "schema": STRONG_PUBLIC_PRACTICE_SCHEMA,
            "run": (
                state.get("run")
                if current_practice_result
                else last_practice_raw.get("run")
                if isinstance(last_practice_raw, dict)
                else state.get("run")
            ),
            "iteration": practice_iteration,
            "shard": (
                state.get("shard")
                if current_practice_result
                else last_practice_raw.get("shard")
                if isinstance(last_practice_raw, dict)
                else None
            ),
            "active": practice_active,
            "stage": practice_stage,
            "updated_at": time.time(),
            "definition": (
                "sampled training-only practice against the active eight-agent "
                "gate roster; replay eligible; never formal gate evidence"
            ),
        },
        "research_controls": {
            **research,
            "schema": RESEARCH_SCHEMA,
            "run": (
                state.get("run")
                if current_result
                else last_research_raw.get("run")
                if isinstance(last_research_raw, dict)
                else state.get("run")
            ),
            "iteration": research_iteration,
            "shard": (
                state.get("shard")
                if current_result
                else last_research_raw.get("shard")
                if isinstance(last_research_raw, dict)
                else None
            ),
            "active": research_active,
            "stage": research_stage,
            "updated_at": time.time(),
            "definition": (
                "legacy in-shard research-control telemetry; new lineages publish "
                "a separate additive greedy non-training result artifact"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/inzi/poke-bot-agent"))
    parser.add_argument("--service", default="pokebot-pure-rl-alakazam.service")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    parser.add_argument(
        "--status",
        type=Path,
        default=Path("/home/inzi/poke-bot-agent/outputs/state/public_mix_live_wr.json"),
    )
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    state = _read_json(args.status)
    while True:
        run_name = _active_run_name(args.service)
        if not run_name:
            if state:
                _atomic_json(args.status, _payload(state, stage="inactive", active=False))
            if args.once:
                return
            time.sleep(max(0.2, args.poll_seconds))
            continue

        run_dir = args.root / "outputs" / "pure_rl" / run_name
        loop = _read_json(run_dir / "loop_state.json")
        stage, iteration = _current_stage(args.root, run_name, loop)
        shard = run_dir / "shards" / f"iter_{iteration:05d}.jsonl"
        state = _state_for_iteration(
            state,
            run_name=run_name,
            iteration=iteration,
            shard=shard,
        )
        _consume_available(shard, state)
        _atomic_json(
            args.status,
            _payload(state, stage=stage, active=stage == "collect:public_mix"),
        )
        if args.once:
            return
        time.sleep(max(0.2, args.poll_seconds))


if __name__ == "__main__":
    main()
