#!/usr/bin/env python3
"""Finite CUDA gate for official clean Basic Pokemon Bench Plays."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import triton
import triton.language as tl


OFFICIAL_LIB_SHA256 = "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
POKEMON_CARD_ID = 22
ENERGY_CARD_ID = 6
OPTION_PLAY = 7
OPTION_ATTACH = 8
OPTION_END = 14
AREA_HAND = 2
AREA_ACTIVE = 4
AREA_BENCH = 5
MAX_HAND = 64
MAX_OPTIONS = 384
CARD_WIDTH = 3
POKEMON_WIDTH = 7
OPTION_WIDTH = 7

SCALAR_KEYS = (
    "turn",
    "turnActionCount",
    "yourIndex",
    "firstPlayer",
    "supporterPlayed",
    "stadiumPlayed",
    "energyAttached",
    "retreated",
    "result",
    "handCount",
    "benchCount",
    "benchMax",
    "terminal",
)
SCALAR_WIDTH = len(SCALAR_KEYS)
SCALAR_INDEX = {key: index for index, key in enumerate(SCALAR_KEYS)}
LEGAL_META_KEYS = ("select_type", "select_context", "select_min", "select_max", "option_count")
OPTION_KEYS = (
    "type",
    "area",
    "index",
    "playerIndex",
    "inPlayArea",
    "inPlayIndex",
    "attackId",
)


@triton.jit
def clean_basic_bench_play_kernel(
    input_scalar,
    input_hand,
    selected,
    output_scalar,
    output_hand,
    output_pokemon,
    output_legal_meta,
    output_options,
    n: tl.constexpr,
    scalar_width: tl.constexpr,
    max_hand: tl.constexpr,
    max_options: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.arange(0, block)
    lane_valid = row < n
    scalar_mask = lane_valid & (offset < scalar_width)
    scalar = tl.load(
        input_scalar + row * scalar_width + offset,
        mask=scalar_mask,
        other=-1,
    ).to(tl.int32)
    scalar = tl.where(offset == 1, scalar + 1, scalar)  # turnActionCount
    scalar = tl.where(offset == 9, scalar - 1, scalar)  # handCount
    scalar = tl.where(offset == 10, scalar + 1, scalar)  # benchCount
    tl.store(output_scalar + row * scalar_width + offset, scalar, mask=scalar_mask)

    source_index = tl.load(selected + row, mask=lane_valid, other=0).to(tl.int32)
    old_hand_count = tl.load(
        input_scalar + row * scalar_width + 9,
        mask=lane_valid,
        other=0,
    ).to(tl.int32)
    old_bench_count = tl.load(
        input_scalar + row * scalar_width + 10,
        mask=lane_valid,
        other=0,
    ).to(tl.int32)
    bench_max = tl.load(
        input_scalar + row * scalar_width + 11,
        mask=lane_valid,
        other=0,
    ).to(tl.int32)
    energy_attached = tl.load(
        input_scalar + row * scalar_width + 6,
        mask=lane_valid,
        other=1,
    ).to(tl.int32)
    new_hand_count = old_hand_count - 1
    new_bench_count = old_bench_count + 1
    source_slot = offset + (offset >= source_index).to(tl.int32)
    hand_mask = lane_valid & (offset < max_hand)
    source_valid = hand_mask & (offset < new_hand_count)
    new_id = tl.load(
        input_hand + (row * max_hand + source_slot) * 3,
        mask=source_valid,
        other=-1,
    ).to(tl.int32)
    new_serial = tl.load(
        input_hand + (row * max_hand + source_slot) * 3 + 1,
        mask=source_valid,
        other=-1,
    ).to(tl.int32)
    new_player = tl.load(
        input_hand + (row * max_hand + source_slot) * 3 + 2,
        mask=source_valid,
        other=-1,
    ).to(tl.int32)
    tl.store(output_hand + (row * max_hand + offset) * 3, new_id, mask=hand_mask)
    tl.store(output_hand + (row * max_hand + offset) * 3 + 1, new_serial, mask=hand_mask)
    tl.store(output_hand + (row * max_hand + offset) * 3 + 2, new_player, mask=hand_mask)

    moved_id = tl.load(
        input_hand + (row * max_hand + source_index) * 3,
        mask=lane_valid,
        other=-1,
    ).to(tl.int32)
    moved_serial = tl.load(
        input_hand + (row * max_hand + source_index) * 3 + 1,
        mask=lane_valid,
        other=-1,
    ).to(tl.int32)
    moved_player = tl.load(
        input_hand + (row * max_hand + source_index) * 3 + 2,
        mask=lane_valid,
        other=-1,
    ).to(tl.int32)
    pokemon_mask = lane_valid & (offset < 7)
    pokemon_value = tl.where(
        offset == 0,
        moved_id,
        tl.where(
            offset == 1,
            moved_serial,
            tl.where(
                offset == 2,
                moved_player,
                tl.where(
                    (offset == 3) | (offset == 4),
                    90,
                    tl.where(offset == 5, 1, 0),
                ),
            ),
        ),
    )
    tl.store(output_pokemon + row * 7 + offset, pokemon_value, mask=pokemon_mask)

    # Construct the complete next legal surface on device in official order.
    hand_lane = lane_valid & (offset < new_hand_count)
    can_play = hand_lane & (new_id == 22) & (new_bench_count < bench_max)
    can_attach = hand_lane & (new_id == 6) & (energy_attached == 0)
    in_play_count = 1 + new_bench_count
    contribution = tl.where(can_play, 1, tl.where(can_attach, in_play_count, 0)).to(tl.int32)
    prefix = tl.cumsum(contribution, axis=0)
    start = prefix - contribution
    option_total = tl.sum(contribution, axis=0)

    # A Play contributes one option at its prefix position.
    for field in range(0, 7):
        play_value = tl.where(field == 0, 7, tl.where(field == 2, offset, -1))
        tl.store(
            output_options + (row * max_options + start) * 7 + field,
            play_value,
            mask=can_play,
        )

    # Each Energy contributes Active first, then every Bench target.
    for target_index in range(0, 6):
        target_valid = can_attach & (target_index < in_play_count)
        target_area = tl.where(target_index == 0, 4, 5)
        target_area_index = tl.where(target_index == 0, 0, target_index - 1)
        slot = start + target_index
        for field in range(0, 7):
            attach_value = tl.where(
                field == 0,
                8,
                tl.where(
                    field == 1,
                    2,
                    tl.where(
                        field == 2,
                        offset,
                        tl.where(
                            field == 4,
                            target_area,
                            tl.where(field == 5, target_area_index, -1),
                        ),
                    ),
                ),
            )
            tl.store(
                output_options + (row * max_options + slot) * 7 + field,
                attach_value,
                mask=target_valid,
            )

    # End is always the final legal option.
    end_lane = lane_valid & (offset == 0)
    for field in range(0, 7):
        end_value = tl.where(field == 0, 14, -1)
        tl.store(
            output_options + (row * max_options + option_total) * 7 + field + offset,
            end_value,
            mask=end_lane,
        )

    meta_mask = lane_valid & (offset < 5)
    meta_value = tl.where(
        offset == 0,
        0,
        tl.where(
            offset == 1,
            0,
            tl.where((offset == 2) | (offset == 3), 1, option_total + 1),
        ),
    )
    tl.store(output_legal_meta + row * 5 + offset, meta_value, mask=meta_mask)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def publish(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def scalar_row(current: dict[str, Any]) -> list[int]:
    actor = int(current["yourIndex"])
    player = current["players"][actor]
    return [
        int(current["turn"]),
        int(current["turnActionCount"]),
        actor,
        int(current["firstPlayer"]),
        int(bool(current["supporterPlayed"])),
        int(bool(current["stadiumPlayed"])),
        int(bool(current["energyAttached"])),
        int(bool(current["retreated"])),
        int(current["result"]),
        int(player["handCount"]),
        len(player.get("bench") or []),
        int(player["benchMax"]),
        int(int(current["result"]) != -1),
    ]


def hand_row(current: dict[str, Any]) -> list[list[int]]:
    actor = int(current["yourIndex"])
    hand = current["players"][actor]["hand"]
    if hand is None or len(hand) > MAX_HAND:
        raise ValueError("fixture selecting hand is hidden or over capacity")
    values = [
        [int(card["id"]), int(card["serial"]), int(card["playerIndex"])]
        for card in hand
    ]
    return values + [[-1, -1, -1] for _ in range(MAX_HAND - len(values))]


def pokemon_row(current: dict[str, Any], bench_index: int) -> list[int]:
    actor = int(current["yourIndex"])
    pokemon = current["players"][actor]["bench"][bench_index]
    if pokemon.get("energies") or pokemon.get("energyCards") or pokemon.get("tools") or pokemon.get("preEvolution"):
        raise ValueError("new clean Bench Pokemon unexpectedly has attached state")
    return [
        int(pokemon["id"]),
        int(pokemon["serial"]),
        int(pokemon["playerIndex"]),
        int(pokemon["hp"]),
        int(pokemon["maxHp"]),
        int(bool(pokemon["appearThisTurn"])),
        0,
    ]


def encode_option(option: dict[str, Any]) -> list[int]:
    return [
        int(option.get("type", -1)),
        int(option.get("area", -1)),
        int(option.get("index", -1)),
        int(option.get("playerIndex", -1)),
        int(option.get("inPlayArea", -1)),
        int(option.get("inPlayIndex", -1)),
        int(option.get("attackId", -1)),
    ]


def legal_rows(select: dict[str, Any]) -> tuple[list[int], list[list[int]]]:
    options = select.get("option") or []
    if len(options) > MAX_OPTIONS:
        raise ValueError("legal option surface exceeds fixed CUDA capacity")
    meta = [
        int(select.get("type", -1)),
        int(select.get("context", -1)),
        int(select.get("minCount", -1)),
        int(select.get("maxCount", -1)),
        len(options),
    ]
    encoded = [encode_option(option) for option in options]
    encoded += [[-1] * OPTION_WIDTH for _ in range(MAX_OPTIONS - len(encoded))]
    return meta, encoded


def pack(
    fixtures: list[dict[str, Any]], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    targets_meta_options = [legal_rows(row["after"]["select"]) for row in fixtures]
    inputs = {
        "scalar": torch.tensor(
            [scalar_row(row["before"]["current"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
        "hand": torch.tensor(
            [hand_row(row["before"]["current"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
        "selected": torch.tensor(
            [int(row["selected_option"]["index"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
    }
    targets = {
        "scalar": torch.tensor(
            [scalar_row(row["after"]["current"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
        "hand": torch.tensor(
            [hand_row(row["after"]["current"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
        "pokemon": torch.tensor(
            [
                pokemon_row(
                    row["after"]["current"],
                    len(row["before"]["current"]["players"][row["before"]["current"]["yourIndex"]]["bench"]),
                )
                for row in fixtures
            ],
            dtype=torch.int32,
            device=device,
        ),
        "legal_meta": torch.tensor(
            [value[0] for value in targets_meta_options], dtype=torch.int32, device=device
        ),
        "options": torch.tensor(
            [value[1] for value in targets_meta_options], dtype=torch.int32, device=device
        ),
    }
    return inputs, targets


def allocate_outputs(n: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "scalar": torch.empty((n, SCALAR_WIDTH), dtype=torch.int32, device=device),
        "hand": torch.empty((n, MAX_HAND, CARD_WIDTH), dtype=torch.int32, device=device),
        "pokemon": torch.empty((n, POKEMON_WIDTH), dtype=torch.int32, device=device),
        "legal_meta": torch.empty((n, len(LEGAL_META_KEYS)), dtype=torch.int32, device=device),
        "options": torch.full((n, MAX_OPTIONS, OPTION_WIDTH), -1, dtype=torch.int32, device=device),
    }


def launch(inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
    n = int(inputs["scalar"].shape[0])
    clean_basic_bench_play_kernel[(n,)](
        inputs["scalar"],
        inputs["hand"],
        inputs["selected"],
        outputs["scalar"],
        outputs["hand"],
        outputs["pokemon"],
        outputs["legal_meta"],
        outputs["options"],
        n=n,
        scalar_width=SCALAR_WIDTH,
        max_hand=MAX_HAND,
        max_options=MAX_OPTIONS,
        block=64,
        num_warps=2,
    )


def reconstruct_current(
    before: dict[str, Any], scalar: list[int], hand: list[list[int]], pokemon: list[int]
) -> dict[str, Any]:
    current = copy.deepcopy(before)
    values = dict(zip(SCALAR_KEYS, scalar))
    for key in ("turn", "turnActionCount", "yourIndex", "firstPlayer", "result"):
        current[key] = int(values[key])
    for key in ("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated"):
        current[key] = bool(values[key])
    actor = int(current["yourIndex"])
    player = current["players"][actor]
    hand_count = int(values["handCount"])
    player["handCount"] = hand_count
    player["hand"] = [
        {"id": int(card[0]), "serial": int(card[1]), "playerIndex": int(card[2])}
        for card in hand[:hand_count]
    ]
    player["bench"].append({
        "id": int(pokemon[0]),
        "serial": int(pokemon[1]),
        "playerIndex": int(pokemon[2]),
        "hp": int(pokemon[3]),
        "maxHp": int(pokemon[4]),
        "appearThisTurn": bool(pokemon[5]),
        "energies": [],
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    })
    return current


def reconstruct_select(meta: list[int], encoded: list[list[int]]) -> dict[str, Any]:
    options: list[dict[str, int]] = []
    for values in encoded[: int(meta[4])]:
        option: dict[str, int] = {"type": int(values[0])}
        for key, value in zip(OPTION_KEYS[1:], values[1:]):
            if int(value) != -1:
                option[key] = int(value)
        options.append(option)
    return {
        "type": int(meta[0]),
        "context": int(meta[1]),
        "minCount": int(meta[2]),
        "maxCount": int(meta[3]),
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": options,
        "deck": None,
        "contextCard": None,
        "effect": None,
    }


def diff_paths(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if type(expected) is not type(actual):
        return [prefix or "$type"]
    if isinstance(expected, dict):
        paths: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in expected or key not in actual:
                paths.append(child)
            else:
                paths.extend(diff_paths(expected[key], actual[key], child))
        return paths
    if isinstance(expected, list):
        paths = []
        if len(expected) != len(actual):
            paths.append(f"{prefix}.length")
        for index, (left, right) in enumerate(zip(expected, actual)):
            paths.extend(diff_paths(left, right, f"{prefix}[{index}]"))
        return paths
    return [] if expected == actual else [prefix]


def repeat_inputs(inputs: dict[str, torch.Tensor], lanes: int) -> dict[str, torch.Tensor]:
    n = int(inputs["scalar"].shape[0])
    repeats = (lanes + n - 1) // n
    return {
        key: value.repeat((repeats,) + (1,) * (value.ndim - 1))[:lanes].contiguous()
        for key, value in inputs.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--benchmark-lanes", type=int, default=16384)
    parser.add_argument("--benchmark-iters", type=int, default=25)
    parser.add_argument("--expected-libcg-sha256", default=OFFICIAL_LIB_SHA256)
    args = parser.parse_args()
    if args.benchmark_lanes < 1 or args.benchmark_iters < 1:
        raise ValueError("finite benchmark lanes/iters must be >= 1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(device)
    if "3080 Ti" not in gpu:
        raise SystemExit(f"refusing non-3080 Ti GPU: {gpu}")
    fixture_report = json.loads(args.fixtures.read_text())
    if fixture_report.get("status") != "complete" or not fixture_report.get("fixtures"):
        raise RuntimeError("official fixture report is incomplete or empty")
    if fixture_report.get("official_lib_sha256") != args.expected_libcg_sha256:
        raise RuntimeError("fixture oracle digest does not match required official libcg")
    fixtures = fixture_report["fixtures"]
    report: dict[str, Any] = {
        "schema": "poke_bot.official_basic_bench_play_cuda/v1",
        "status": "running",
        "started_at": time.time(),
        "finite_run": True,
        "gpu": gpu,
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "official_lib_sha256": fixture_report["official_lib_sha256"],
        "official_fixture_report_sha256": hashlib.sha256(args.fixtures.read_bytes()).hexdigest(),
        "official_fixture_count": len(fixtures),
        "scope": fixture_report["scope"],
        "coverage": fixture_report["coverage"],
        "excluded": fixture_report["excluded"],
        "implemented_state": {
            "scalar_fields": list(SCALAR_KEYS),
            "hand_capacity": MAX_HAND,
            "legal_option_capacity": MAX_OPTIONS,
            "legal_option_fields": list(OPTION_KEYS),
            "full_public_reconstruction": True,
            "full_legal_reconstruction": True,
        },
        "full_seeded_game_parity": False,
        "full_card_effect_parity": False,
        "full_engine_transition_coverage": False,
        "production_eligible": False,
    }
    publish(args.json_out, report)
    try:
        inputs, targets = pack(fixtures, device)
        outputs = allocate_outputs(len(fixtures), device)
        launch(inputs, outputs)
        torch.cuda.synchronize(device)
        field_mismatches: dict[str, int] = {}
        for index, key in enumerate(SCALAR_KEYS):
            field_mismatches[f"state.{key}"] = int(
                (outputs["scalar"][:, index] != targets["scalar"][:, index]).sum().item()
            )
        for index, key in enumerate(("id", "serial", "playerIndex")):
            field_mismatches[f"state.hand.{key}"] = int(
                (outputs["hand"][:, :, index] != targets["hand"][:, :, index]).sum().item()
            )
        for index, key in enumerate(("id", "serial", "playerIndex", "hp", "maxHp", "appearThisTurn", "energy_count")):
            field_mismatches[f"state.new_bench.{key}"] = int(
                (outputs["pokemon"][:, index] != targets["pokemon"][:, index]).sum().item()
            )
        for index, key in enumerate(LEGAL_META_KEYS):
            field_mismatches[f"legal.{key}"] = int(
                (outputs["legal_meta"][:, index] != targets["legal_meta"][:, index]).sum().item()
            )
        for index, key in enumerate(OPTION_KEYS):
            field_mismatches[f"legal.options.{key}"] = int(
                (outputs["options"][:, :, index] != targets["options"][:, :, index]).sum().item()
            )
        cpu = {key: value.cpu().tolist() for key, value in outputs.items()}
        public_paths: Counter[str] = Counter()
        legal_paths: Counter[str] = Counter()
        public_mismatches = 0
        legal_mismatches = 0
        terminal_mismatches = 0
        first_mismatches: list[dict[str, Any]] = []
        for index, row in enumerate(fixtures):
            reconstructed_current = reconstruct_current(
                row["before"]["current"],
                cpu["scalar"][index],
                cpu["hand"][index],
                cpu["pokemon"][index],
            )
            current_diff = diff_paths(row["after"]["current"], reconstructed_current)
            if current_diff:
                public_mismatches += 1
                public_paths.update(current_diff)
            reconstructed_select = reconstruct_select(
                cpu["legal_meta"][index], cpu["options"][index]
            )
            select_diff = diff_paths(row["after"]["select"], reconstructed_select)
            if select_diff:
                legal_mismatches += 1
                legal_paths.update(select_diff)
            expected_terminal = bool(row["after"]["terminal"])
            actual_terminal = bool(cpu["scalar"][index][SCALAR_INDEX["terminal"]])
            if expected_terminal != actual_terminal:
                terminal_mismatches += 1
            if (current_diff or select_diff or expected_terminal != actual_terminal) and len(first_mismatches) < 8:
                first_mismatches.append({
                    "fixture": index,
                    "public_paths": current_diff[:16],
                    "legal_paths": select_diff[:16],
                    "terminal_expected": expected_terminal,
                    "terminal_actual": actual_terminal,
                })
        oracle_failures = sum(
            not all(bool(value) for value in row.get("oracle_checks", {}).values())
            for row in fixtures
        )
        nonzero_fields = {key: value for key, value in field_mismatches.items() if value}
        categories = {
            "step": max(public_mismatches, legal_mismatches, terminal_mismatches),
            "legal": legal_mismatches,
            "terminal": terminal_mismatches,
            "public_state": public_mismatches,
            "oracle_fixture_semantics": oracle_failures,
        }
        exact = not nonzero_fields and not any(categories.values())
        report["parity"] = {
            "exact": exact,
            "step_exact": categories["step"] == 0,
            "legal_exact": categories["legal"] == 0,
            "terminal_exact": categories["terminal"] == 0,
            "public_state_exact": categories["public_state"] == 0,
            "category_fixture_mismatches": categories,
            "field_element_mismatches": field_mismatches,
            "nonzero_field_element_mismatches": nonzero_fields,
            "public_path_mismatches": dict(public_paths.most_common()),
            "legal_path_mismatches": dict(legal_paths.most_common()),
            "first_mismatches": first_mismatches,
            "result_d2h_bytes": sum(value.numel() * value.element_size() for value in outputs.values()),
        }
        if not exact:
            report["status"] = "failed"
            raise RuntimeError(f"official Basic Bench Play parity failed: {categories}; {nonzero_fields}")

        lanes = max(len(fixtures), int(args.benchmark_lanes))
        bench_inputs = repeat_inputs(inputs, lanes)
        bench_outputs = allocate_outputs(lanes, device)
        for _ in range(5):
            launch(bench_inputs, bench_outputs)
        torch.cuda.synchronize(device)
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        begin.record()
        for _ in range(args.benchmark_iters):
            launch(bench_inputs, bench_outputs)
        end.record()
        torch.cuda.synchronize(device)
        wall_s = time.perf_counter() - wall_started
        kernel_s = begin.elapsed_time(end) / 1000.0
        transitions = lanes * args.benchmark_iters
        report["finite_benchmark"] = {
            "lanes": lanes,
            "iterations": args.benchmark_iters,
            "transitions": transitions,
            "kernel_s": kernel_s,
            "wall_s": wall_s,
            "kernel_transitions_per_s": transitions / max(kernel_s, 1e-12),
            "whole_loop_transitions_per_s": transitions / max(wall_s, 1e-12),
            "measured_bulk_h2d_bytes": 0,
            "measured_bulk_d2h_bytes": 0,
            "device_resident_state_bytes": sum(value.numel() * value.element_size() for value in bench_inputs.values()),
            "device_resident_output_bytes": sum(value.numel() * value.element_size() for value in bench_outputs.values()),
        }
        report["status"] = "complete"
        report["completed_at"] = time.time()
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["failed_at"] = time.time()
        raise
    finally:
        publish(args.json_out, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
