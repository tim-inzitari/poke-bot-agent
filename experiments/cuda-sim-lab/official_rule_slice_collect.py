#!/usr/bin/env python3
"""Capture ground truth for the smallest independently testable official rule slice.

The slice is the engine's public selection gate (`checkPlayerSelect`) plus its
terminal predicate.  It includes exact error precedence for duplicate choices,
out-of-range choices, and min/max cardinality, and verifies that rejected
steps preserve the public selection/terminal state.  It does not claim to
implement accepted card-effect transitions.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from poke_bot.engine_rebuild.interfaces import ResetSpec
from poke_bot.engine_rebuild.libcg_multi_env import LibcgMultiEnv


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def public_slice(obs: dict[str, Any]) -> dict[str, int]:
    select = obs.get("select") or {}
    current = obs.get("current") or {}
    return {
        "select_type": int(select.get("type", -1)),
        "select_context": int(select.get("context", -1)),
        "select_min": int(select.get("minCount", 0)),
        "select_max": int(select.get("maxCount", 0)),
        "option_count": len(select.get("option") or []),
        "result": int(current.get("result", -1)),
    }


def valid_action(obs: dict[str, Any]) -> list[int]:
    select = obs.get("select") or {}
    options = select.get("option") or []
    minimum = int(select.get("minCount", 0))
    maximum = int(select.get("maxCount", 0))
    count = max(minimum, min(maximum, len(options)))
    return list(range(count))


def candidate_actions(obs: dict[str, Any]) -> list[tuple[str, list[int]]]:
    select = obs.get("select") or {}
    option_count = len(select.get("option") or [])
    minimum = int(select.get("minCount", 0))
    maximum = int(select.get("maxCount", 0))
    candidates: list[tuple[str, list[int]]] = []
    if option_count:
        candidates.extend([
            ("duplicate", [0, 0]),
            ("duplicate_precedes_range", [option_count, option_count]),
            ("negative_index", [-1]),
            ("upper_out_of_range", [option_count]),
        ])
    if minimum > 0:
        candidates.append(("below_min", []))
    if option_count > maximum and maximum + 1 <= 60:
        candidates.append(("above_max", list(range(maximum + 1))))
    elif option_count:
        # Official duplicate detection is intentionally skipped above 60
        # choices, so 61 in-range zeros reaches the cardinality error.
        candidates.append(("above_60_cardinality", [0] * 61))
    candidates.append(("valid", valid_action(obs)))
    # Stable de-duplication avoids redundant [] candidates.
    seen: set[tuple[int, ...]] = set()
    out: list[tuple[str, list[int]]] = []
    for label, action in candidates:
        key = tuple(action)
        if key not in seen:
            seen.add(key)
            out.append((label, action))
    return out


def call_select(lib: Any, ptr: int, action: list[int]) -> int:
    values = (ctypes.c_int * len(action))(*action)
    return int(lib.Select(ptr, values, len(action)))


def source_audit(source_dir: Path) -> dict[str, Any]:
    files = sorted(list(source_dir.glob("*.h")) + list(source_dir.glob("*.cpp")))
    texts = [path.read_text(errors="replace") for path in files]
    joined = "\n".join(texts)
    patterns = {
        "std_vector": "std::vector",
        "std_unordered_map": "std::unordered_map",
        "std_string": "std::string",
        "std_u8string": "std::u8string",
        "std_function": "std::function",
        "mt19937": "std::mt19937",
        "random_device": "std::random_device",
        "throw_or_exception": "Exception(",
        "function_pointer_storage": "void*",
    }
    return {
        "source_files": len(files),
        "source_lines": sum(text.count("\n") + 1 for text in texts),
        "construct_counts": {name: joined.count(token) for name, token in patterns.items()},
        "source_tree_sha256": hashlib.sha256(
            b"".join(path.name.encode() + b"\0" + path.read_bytes() for path in files)
        ).hexdigest(),
        "direct_cuda_compile_ready": False,
        "evidence_based_boundaries": [
            "authoritative State uses dynamically sized STL containers",
            "card effects dispatch through host function-pointer stacks",
            "RNG uses host std::mt19937/random_device paths",
            "public ABI serializes JSON/base64 for every observation",
            "batch fork amortizes the Python/native boundary but retains CPU transitions and per-lane serialization",
        ],
        "required_full_port_work": [
            "fixed-capacity structure-of-arrays state representation",
            "device-compatible deterministic per-lane RNG",
            "card-effect opcode/device dispatch replacing host function pointers",
            "device-resident legal-action construction and policy inference",
            "official seeded step/legal/terminal trace parity across all cards and games",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--official-lib", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=800)
    args = parser.parse_args()
    deck = [int(value) for value in args.deck.read_text().split()]
    if len(deck) != 60:
        raise ValueError(f"deck must contain 60 cards, got {len(deck)}")
    official_sha = hashlib.sha256(args.official_lib.read_bytes()).hexdigest()
    report: dict[str, Any] = {
        "schema": "poke_bot.official_rule_slice_fixtures/v1",
        "scope": "official selection legality + terminal predicate + rejected-step public-slice invariance",
        "accepted_transition_coverage": False,
        "official_lib_sha256": official_sha,
        "started_at": time.time(),
        "fixtures": [],
        "source_audit": source_audit(args.source_dir),
    }
    env = LibcgMultiEnv(1)
    errors: Counter[int] = Counter()
    contexts: Counter[int] = Counter()
    terminal_rows = 0
    rejected_slice_mismatches = 0
    try:
        for game in range(args.games):
            batch = env.reset([ResetSpec(deck, deck, seed=game)])
            obs = batch.envs[0].obs
            for step in range(args.max_steps):
                pre = public_slice(obs)
                contexts[pre["select_context"]] += 1
                if pre["result"] != -1:
                    report["fixtures"].append({
                        "game": game,
                        "step": step,
                        "kind": "terminal",
                        "state": pre,
                        "action": [],
                        "official_error": 0,
                        "official_terminal": True,
                        "rejected_public_slice_unchanged": None,
                    })
                    terminal_rows += 1
                    break
                ptr = int(env._ptrs[0] or 0)  # isolated fixture process
                lib = env._lib
                valid: list[int] | None = None
                for kind, action in candidate_actions(obs):
                    if kind == "valid":
                        valid = action
                        continue
                    before_obs = env._get_obs(ptr)
                    before_slice = public_slice(before_obs)
                    official_error = call_select(lib, ptr, action)
                    after_obs = env._get_obs(ptr)
                    after_slice = public_slice(after_obs)
                    unchanged = before_slice == after_slice
                    if not unchanged:
                        rejected_slice_mismatches += 1
                    errors[official_error] += 1
                    report["fixtures"].append({
                        "game": game,
                        "step": step,
                        "kind": kind,
                        "state": before_slice,
                        "action": action,
                        "official_error": official_error,
                        "official_terminal": before_slice["result"] != -1,
                        "rejected_public_slice_unchanged": unchanged,
                        "public_obs_unchanged": canonical(before_obs) == canonical(after_obs),
                    })
                if valid is None:
                    raise RuntimeError("no valid action generated")
                official_error = call_select(lib, ptr, valid)
                if official_error:
                    raise RuntimeError(
                        f"official valid selection failed game={game} step={step} err={official_error}"
                    )
                errors[official_error] += 1
                next_obs = env._get_obs(ptr)
                env._obs[0] = next_obs
                env._done[0] = env._is_done(next_obs)
                report["fixtures"].append({
                    "game": game,
                    "step": step,
                    "kind": "valid",
                    "state": pre,
                    "action": valid,
                    "official_error": official_error,
                    "official_terminal": False,
                    "rejected_public_slice_unchanged": None,
                })
                obs = next_obs
            else:
                raise RuntimeError(f"game {game} exceeded {args.max_steps} steps")
    finally:
        env.close()
    report.update({
        "status": "complete",
        "completed_at": time.time(),
        "fixture_count": len(report["fixtures"]),
        "games_completed": terminal_rows,
        "official_error_counts": dict(sorted(errors.items())),
        "select_context_counts": dict(sorted(contexts.items())),
        "rejected_public_slice_mismatches": rejected_slice_mismatches,
    })
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.json_out.with_suffix(args.json_out.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temp.replace(args.json_out)
    print(json.dumps({key: report[key] for key in (
        "status", "fixture_count", "games_completed", "official_error_counts",
        "rejected_public_slice_mismatches", "official_lib_sha256",
    )}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
