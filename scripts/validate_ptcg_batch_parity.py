#!/usr/bin/env python3
"""Exact seeded transition parity: individual ABI versus StepBatch ABI."""

from __future__ import annotations

import argparse
import base64
from collections import Counter
import ctypes
import hashlib
import json
from pathlib import Path
from typing import Optional

from poke_bot.engine_rebuild.interfaces import ResetSpec
from poke_bot.engine_rebuild.libcg_batch import (
    BatchedLibcgMultiEnv,
    SerialData,
    load_batch_library,
)


def _load_deck(path: Path) -> list[int]:
    cards = [int(line) for line in path.read_text().splitlines() if line.strip()]
    if len(cards) != 60:
        raise ValueError(f"deck must contain 60 cards, got {len(cards)}")
    return cards


def _decode(serial: SerialData) -> dict:
    if not serial.json:
        raise RuntimeError("native engine returned empty JSON")
    obs = json.loads(serial.json.decode("utf-8"))
    if serial.data and serial.count > 0:
        obs["search_begin_input"] = ctypes.string_at(
            serial.data, serial.count
        ).decode("ascii")
    return obs


def _action(obs: dict) -> Optional[list[int]]:
    select = (obs or {}).get("select") or {}
    options = select.get("option") or []
    if not options:
        return None
    minimum = int(select.get("minCount") or 0)
    maximum = int(select.get("maxCount") or 1)
    count = max(minimum, min(maximum, len(options)))
    return list(range(count)) if count > 0 else []


def _done(obs: dict) -> bool:
    result = ((obs or {}).get("current") or {}).get("result")
    return result is not None and int(result) != -1


def _canonical(obs: dict) -> bytes:
    semantic = {key: value for key, value in obs.items() if key != "search_begin_input"}
    return json.dumps(semantic, sort_keys=True, separators=(",", ":")).encode()


def _expand_engine_base64(value: str) -> bytes:
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

    def index(char: str) -> int:
        return alphabet.index(char) if char in alphabet else 0

    expanded: list[str] = []
    i = 0
    while i < len(value):
        char = value[i]
        if char == "A":
            i += 1
            count = index(value[i])
            expanded.append("A" * count)
        elif char == "-":
            count = index(value[i + 1]) + 64 * index(value[i + 2])
            expanded.append("A" * count)
            i += 2
        elif char == "*":
            count = (
                index(value[i + 1])
                + 64 * index(value[i + 2])
                + 64 * 64 * index(value[i + 3])
            )
            expanded.append("A" * count)
            i += 3
        else:
            expanded.append(char)
        i += 1
    return base64.b64decode("".join(expanded))


def _first_diff(expected: object, actual: object, path: str = "$") -> str:
    if type(expected) is not type(actual):
        return f"{path}: type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)  # type: ignore[arg-type]
        if expected_keys != actual_keys:
            return (
                f"{path}: keys only_expected={sorted(expected_keys - actual_keys)} "
                f"only_actual={sorted(actual_keys - expected_keys)}"
            )
        for key in sorted(expected_keys):
            diff = _first_diff(
                expected[key], actual[key], f"{path}.{key}"  # type: ignore[index]
            )
            if diff:
                return diff
        return ""
    if isinstance(expected, list):
        if len(expected) != len(actual):  # type: ignore[arg-type]
            return f"{path}: length {len(expected)} != {len(actual)}"  # type: ignore[arg-type]
        for i, (left, right) in enumerate(zip(expected, actual)):  # type: ignore[arg-type]
            diff = _first_diff(left, right, f"{path}[{i}]")
            if diff:
                return diff
        return ""
    if expected != actual:
        return f"{path}: {expected!r} != {actual!r}"
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lib", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=0x51A7E000)
    args = parser.parse_args()
    if args.num_envs < 1 or args.max_steps < 1:
        parser.error("num-envs and max-steps must be positive")

    deck = _load_deck(args.deck)
    lib = load_batch_library(args.lib)
    reference_ptrs: list[int] = []
    reference_obs: list[dict] = []
    digest = hashlib.sha256()
    serialized_mismatches = 0
    serialized_diff_offsets: Counter[int] = Counter()
    batch = BatchedLibcgMultiEnv(args.num_envs, lib=lib)
    try:
        specs = [
            ResetSpec(deck, deck, seed=(args.seed + i) & 0xFFFFFFFF)
            for i in range(args.num_envs)
        ]
        batch_obs = batch.reset(specs)

        for spec in specs:
            cards = (ctypes.c_int * 120)(*(spec.deck0 + spec.deck1))
            start = lib.BattleStartSeeded(cards, spec.seed & 0xFFFFFFFF)
            ptr = int(start.battlePtr or 0)
            if not ptr:
                raise RuntimeError(
                    f"individual seeded start failed player={start.errorPlayer} "
                    f"type={start.errorType}"
                )
            reference_ptrs.append(ptr)
            reference_obs.append(_decode(lib.GetBattleData(ptr)))

        for i in range(args.num_envs):
            expected = _canonical(reference_obs[i])
            actual = _canonical(batch_obs.envs[i].obs)
            if expected != actual:
                raise AssertionError(f"initial state mismatch env={i}")
            digest.update(actual)

        decisions = 0
        for step in range(args.max_steps):
            if all(_done(obs) for obs in reference_obs):
                report = {
                    "ok": True,
                    "num_envs": args.num_envs,
                    "steps": step,
                    "decisions": decisions,
                    "serialized_mismatches": serialized_mismatches,
                    "serialized_diff_offsets": serialized_diff_offsets.most_common(20),
                    "transition_sha256": digest.hexdigest(),
                }
                print(json.dumps(report, sort_keys=True))
                return 0

            actions: list[Optional[list[int]]] = [None] * args.num_envs
            for i, obs in enumerate(reference_obs):
                if _done(obs):
                    continue
                action = _action(obs)
                if action is None:
                    raise RuntimeError(f"env={i} live without legal action at step={step}")
                actions[i] = action
                values = (ctypes.c_int * len(action))(*action)
                err = int(lib.Select(reference_ptrs[i], values, len(action)))
                if err:
                    raise RuntimeError(
                        f"individual Select failed env={i} step={step} err={err}"
                    )
                reference_obs[i] = _decode(lib.GetBattleData(reference_ptrs[i]))
                decisions += 1

            batch_obs = batch.step_batch(actions)
            for i, action in enumerate(actions):
                expected = _canonical(reference_obs[i])
                actual = _canonical(batch_obs.envs[i].obs)
                if expected != actual:
                    detail = _first_diff(reference_obs[i], batch_obs.envs[i].obs)
                    raise AssertionError(
                        f"transition mismatch env={i} step={step} action={action}; "
                        f"{detail}; expected_sha={hashlib.sha256(expected).hexdigest()} "
                        f"actual_sha={hashlib.sha256(actual).hexdigest()}"
                    )
                expected_serial = reference_obs[i].get("search_begin_input")
                actual_serial = batch_obs.envs[i].obs.get("search_begin_input")
                if expected_serial != actual_serial:
                    serialized_mismatches += 1
                    expected_raw = _expand_engine_base64(str(expected_serial))
                    actual_raw = _expand_engine_base64(str(actual_serial))
                    for offset, (left, right) in enumerate(
                        zip(expected_raw, actual_raw)
                    ):
                        if left != right:
                            serialized_diff_offsets[offset] += 1
                    if len(expected_raw) != len(actual_raw):
                        serialized_diff_offsets[-1] += 1
                digest.update(json.dumps(action, separators=(",", ":")).encode())
                digest.update(actual)
        raise RuntimeError(f"games did not finish within {args.max_steps} steps")
    finally:
        batch.close()
        for ptr in reference_ptrs:
            lib.BattleFinish(ptr)


if __name__ == "__main__":
    raise SystemExit(main())
