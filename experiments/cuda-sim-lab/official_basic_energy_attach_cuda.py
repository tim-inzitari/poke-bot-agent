#!/usr/bin/env python3
"""Finite CUDA parity gate for a clean official Basic Energy attachment.

This is one bounded transition slice, not a Pokemon engine claim.  It consumes
recorded transitions from the unmodified official libcg binary, executes the
same hand-to-Bench Basic Energy mutation on the RTX 3080 Ti, reconstructs the
complete public ``current`` and ``select`` objects from CUDA outputs, and
compares those objects exactly with the official observations.
"""

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
ENERGY_CARD_ID = 6
ENERGY_TYPE = 6
OPTION_END = 14
AREA_BENCH = 5
MAX_HAND = 64
MAX_TARGET_ENERGY = 64
CARD_WIDTH = 3  # id, serial, playerIndex
ENERGY_WIDTH = 4  # Energy type, card id, serial, playerIndex
LEGAL_WIDTH = 12

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
    "target_id",
    "target_serial",
    "target_playerIndex",
    "target_hp",
    "target_maxHp",
    "target_appearThisTurn",
    "target_energy_count",
    "terminal",
)
SCALAR_WIDTH = len(SCALAR_KEYS)
SCALAR_INDEX = {key: index for index, key in enumerate(SCALAR_KEYS)}
LEGAL_KEYS = (
    "select_type",
    "select_context",
    "select_min",
    "select_max",
    "option_count",
    "option0_type",
    "option0_area",
    "option0_index",
    "option0_playerIndex",
    "option0_inPlayArea",
    "option0_inPlayIndex",
    "option0_attackId",
)


@triton.jit
def clean_basic_energy_attach_kernel(
    input_scalar,
    input_hand,
    input_energy,
    selected,
    output_scalar,
    output_hand,
    output_energy,
    output_legal,
    n: tl.constexpr,
    scalar_width: tl.constexpr,
    max_hand: tl.constexpr,
    max_energy: tl.constexpr,
    block: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.arange(0, block)
    lane_valid = row < n

    # Scalar state is copied device-to-device, then the four official clean
    # Attach mutations are applied.
    scalar_mask = lane_valid & (offset < scalar_width)
    scalar = tl.load(
        input_scalar + row * scalar_width + offset,
        mask=scalar_mask,
        other=-1,
    ).to(tl.int32)
    scalar = tl.where(offset == 1, scalar + 1, scalar)  # turnActionCount
    scalar = tl.where(offset == 6, 1, scalar)  # energyAttached
    scalar = tl.where(offset == 9, scalar - 1, scalar)  # handCount
    scalar = tl.where(offset == 16, scalar + 1, scalar)  # target energy count
    tl.store(
        output_scalar + row * scalar_width + offset,
        scalar,
        mask=scalar_mask,
    )

    source_index = tl.load(selected + row * 3, mask=lane_valid, other=0).to(tl.int32)
    old_hand_count = tl.load(
        input_scalar + row * scalar_width + 9,
        mask=lane_valid,
        other=0,
    ).to(tl.int32)
    new_hand_count = old_hand_count - 1
    source_slot = offset + (offset >= source_index).to(tl.int32)
    hand_mask = lane_valid & (offset < max_hand)
    source_valid = hand_mask & (offset < new_hand_count)
    for field in range(0, 3):
        value = tl.load(
            input_hand + (row * max_hand + source_slot) * 3 + field,
            mask=source_valid,
            other=-1,
        ).to(tl.int32)
        tl.store(
            output_hand + (row * max_hand + offset) * 3 + field,
            value,
            mask=hand_mask,
        )

    old_energy_count = tl.load(
        input_scalar + row * scalar_width + 16,
        mask=lane_valid,
        other=0,
    ).to(tl.int32)
    energy_mask = lane_valid & (offset < max_energy)
    existing = energy_mask & (offset < old_energy_count)
    appended = energy_mask & (offset == old_energy_count)
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
    for field in range(0, 4):
        prior = tl.load(
            input_energy + (row * max_energy + offset) * 4 + field,
            mask=existing,
            other=-1,
        ).to(tl.int32)
        appended_value = tl.where(
            field == 0,
            6,
            tl.where(field == 1, moved_id, tl.where(field == 2, moved_serial, moved_player)),
        )
        value = tl.where(appended, appended_value, prior)
        tl.store(
            output_energy + (row * max_energy + offset) * 4 + field,
            value,
            mask=energy_mask,
        )

    # Exact legal result for the admitted clean slice: Main, one required End.
    legal_mask = lane_valid & (offset < 12)
    legal = tl.full((block,), -1, tl.int32)
    legal = tl.where(offset == 0, 0, legal)  # SelectType Main (API value)
    legal = tl.where(offset == 1, 0, legal)  # SelectContext Main (API value)
    legal = tl.where(offset == 2, 1, legal)
    legal = tl.where(offset == 3, 1, legal)
    legal = tl.where(offset == 4, 1, legal)
    legal = tl.where(offset == 5, 14, legal)  # SelectOptionType End
    tl.store(output_legal + row * 12 + offset, legal, mask=legal_mask)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def publish(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def target_from(current: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    if int(selected["inPlayArea"]) != AREA_BENCH:
        raise ValueError("fixture escaped clean hand-to-Bench coverage")
    actor = int(current["yourIndex"])
    return current["players"][actor]["bench"][int(selected["inPlayIndex"])]


def scalar_row(current: dict[str, Any], selected: dict[str, Any]) -> list[int]:
    actor = int(current["yourIndex"])
    player = current["players"][actor]
    target = target_from(current, selected)
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
        int(target["id"]),
        int(target["serial"]),
        int(target["playerIndex"]),
        int(target["hp"]),
        int(target["maxHp"]),
        int(bool(target["appearThisTurn"])),
        len(target.get("energies") or []),
        int(int(current["result"]) != -1),
    ]


def hand_row(current: dict[str, Any]) -> list[list[int]]:
    actor = int(current["yourIndex"])
    hand = current["players"][actor]["hand"]
    if hand is None or len(hand) > MAX_HAND:
        raise ValueError("fixture has hidden or over-capacity selecting hand")
    values = [
        [int(card["id"]), int(card["serial"]), int(card["playerIndex"])]
        for card in hand
    ]
    return values + [[-1, -1, -1] for _ in range(MAX_HAND - len(values))]


def energy_row(current: dict[str, Any], selected: dict[str, Any]) -> list[list[int]]:
    target = target_from(current, selected)
    types = target.get("energies") or []
    cards = target.get("energyCards") or []
    if len(types) != len(cards) or len(types) > MAX_TARGET_ENERGY:
        raise ValueError("fixture target Energy representation is inconsistent")
    values = [
        [
            int(energy_type),
            int(card["id"]),
            int(card["serial"]),
            int(card["playerIndex"]),
        ]
        for energy_type, card in zip(types, cards)
    ]
    return values + [[-1, -1, -1, -1] for _ in range(MAX_TARGET_ENERGY - len(values))]


def legal_row(select: dict[str, Any]) -> list[int]:
    options = select.get("option") or []
    option = options[0] if options else {}
    return [
        int(select.get("type", -1)),
        int(select.get("context", -1)),
        int(select.get("minCount", -1)),
        int(select.get("maxCount", -1)),
        len(options),
        int(option.get("type", -1)),
        int(option.get("area", -1)),
        int(option.get("index", -1)),
        int(option.get("playerIndex", -1)),
        int(option.get("inPlayArea", -1)),
        int(option.get("inPlayIndex", -1)),
        int(option.get("attackId", -1)),
    ]


def pack(
    fixtures: list[dict[str, Any]], device: torch.device
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    inputs = {
        "scalar": torch.tensor(
            [scalar_row(row["before"]["current"], row["selected_option"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
        "hand": torch.tensor(
            [hand_row(row["before"]["current"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
        "energy": torch.tensor(
            [energy_row(row["before"]["current"], row["selected_option"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
        "selected": torch.tensor(
            [
                [
                    int(row["selected_option"]["index"]),
                    int(row["selected_option"]["inPlayArea"]),
                    int(row["selected_option"]["inPlayIndex"]),
                ]
                for row in fixtures
            ],
            dtype=torch.int32,
            device=device,
        ),
    }
    targets = {
        "scalar": torch.tensor(
            [scalar_row(row["after"]["current"], row["selected_option"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
        "hand": torch.tensor(
            [hand_row(row["after"]["current"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
        "energy": torch.tensor(
            [energy_row(row["after"]["current"], row["selected_option"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
        "legal": torch.tensor(
            [legal_row(row["after"]["select"]) for row in fixtures],
            dtype=torch.int32,
            device=device,
        ),
    }
    return inputs, targets


def allocate_outputs(n: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "scalar": torch.empty((n, SCALAR_WIDTH), dtype=torch.int32, device=device),
        "hand": torch.empty((n, MAX_HAND, CARD_WIDTH), dtype=torch.int32, device=device),
        "energy": torch.empty((n, MAX_TARGET_ENERGY, ENERGY_WIDTH), dtype=torch.int32, device=device),
        "legal": torch.empty((n, LEGAL_WIDTH), dtype=torch.int32, device=device),
    }


def launch(inputs: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
    n = int(inputs["scalar"].shape[0])
    clean_basic_energy_attach_kernel[(n,)](
        inputs["scalar"],
        inputs["hand"],
        inputs["energy"],
        inputs["selected"],
        outputs["scalar"],
        outputs["hand"],
        outputs["energy"],
        outputs["legal"],
        n=n,
        scalar_width=SCALAR_WIDTH,
        max_hand=MAX_HAND,
        max_energy=MAX_TARGET_ENERGY,
        block=64,
        num_warps=2,
    )


def reconstruct_current(
    before: dict[str, Any],
    selected: dict[str, Any],
    scalar: list[int],
    hand: list[list[int]],
    energy: list[list[int]],
) -> dict[str, Any]:
    current = copy.deepcopy(before)
    scalar_values = dict(zip(SCALAR_KEYS, scalar))
    for key in ("turn", "turnActionCount", "yourIndex", "firstPlayer", "result"):
        current[key] = int(scalar_values[key])
    for key in ("supporterPlayed", "stadiumPlayed", "energyAttached", "retreated"):
        current[key] = bool(scalar_values[key])
    actor = int(current["yourIndex"])
    player = current["players"][actor]
    hand_count = int(scalar_values["handCount"])
    player["handCount"] = hand_count
    player["hand"] = [
        {"id": int(card[0]), "serial": int(card[1]), "playerIndex": int(card[2])}
        for card in hand[:hand_count]
    ]
    target = target_from(current, selected)
    target["id"] = int(scalar_values["target_id"])
    target["serial"] = int(scalar_values["target_serial"])
    target["playerIndex"] = int(scalar_values["target_playerIndex"])
    target["hp"] = int(scalar_values["target_hp"])
    target["maxHp"] = int(scalar_values["target_maxHp"])
    target["appearThisTurn"] = bool(scalar_values["target_appearThisTurn"])
    energy_count = int(scalar_values["target_energy_count"])
    target["energies"] = [int(value[0]) for value in energy[:energy_count]]
    target["energyCards"] = [
        {"id": int(value[1]), "serial": int(value[2]), "playerIndex": int(value[3])}
        for value in energy[:energy_count]
    ]
    return current


def reconstruct_select(legal: list[int]) -> dict[str, Any]:
    option_count = int(legal[4])
    options: list[dict[str, int]] = []
    if option_count:
        option: dict[str, int] = {"type": int(legal[5])}
        optional = (
            ("area", 6),
            ("index", 7),
            ("playerIndex", 8),
            ("inPlayArea", 9),
            ("inPlayIndex", 10),
            ("attackId", 11),
        )
        for key, index in optional:
            if int(legal[index]) != -1:
                option[key] = int(legal[index])
        options.append(option)
    return {
        "type": int(legal[0]),
        "context": int(legal[1]),
        "minCount": int(legal[2]),
        "maxCount": int(legal[3]),
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


def repeat_inputs(
    inputs: dict[str, torch.Tensor], lanes: int
) -> dict[str, torch.Tensor]:
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
    parser.add_argument("--benchmark-lanes", type=int, default=65536)
    parser.add_argument("--benchmark-iters", type=int, default=50)
    parser.add_argument("--expected-libcg-sha256", default=OFFICIAL_LIB_SHA256)
    args = parser.parse_args()
    if args.benchmark_lanes < 1 or args.benchmark_iters < 1:
        raise ValueError("finite benchmark lanes/iters must both be >= 1")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device("cuda:0")
    gpu = torch.cuda.get_device_name(device)
    if "3080 Ti" not in gpu:
        raise SystemExit(f"refusing non-3080 Ti GPU: {gpu}")
    fixture_report = json.loads(args.fixtures.read_text())
    if fixture_report.get("status") != "complete":
        raise RuntimeError("official fixture report is not complete")
    if fixture_report.get("official_lib_sha256") != args.expected_libcg_sha256:
        raise RuntimeError(
            "official fixture digest does not match the required competition libcg"
        )
    fixtures = fixture_report.get("fixtures") or []
    if not fixtures:
        raise RuntimeError("official fixture corpus is empty")
    report: dict[str, Any] = {
        "schema": "poke_bot.official_basic_energy_attach_cuda/v1",
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
            "target_energy_capacity": MAX_TARGET_ENERGY,
            "card_identity_fields": ["id", "serial", "playerIndex"],
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
        for index, key in enumerate(("type", "card_id", "serial", "playerIndex")):
            field_mismatches[f"state.target_energy.{key}"] = int(
                (outputs["energy"][:, :, index] != targets["energy"][:, :, index]).sum().item()
            )
        for index, key in enumerate(LEGAL_KEYS):
            field_mismatches[f"legal.{key}"] = int(
                (outputs["legal"][:, index] != targets["legal"][:, index]).sum().item()
            )

        cpu_outputs = {key: value.cpu().tolist() for key, value in outputs.items()}
        public_path_mismatches: Counter[str] = Counter()
        legal_path_mismatches: Counter[str] = Counter()
        public_fixture_mismatches = 0
        legal_fixture_mismatches = 0
        terminal_fixture_mismatches = 0
        first_mismatches: list[dict[str, Any]] = []
        for index, row in enumerate(fixtures):
            reconstructed_current = reconstruct_current(
                row["before"]["current"],
                row["selected_option"],
                cpu_outputs["scalar"][index],
                cpu_outputs["hand"][index],
                cpu_outputs["energy"][index],
            )
            current_paths = diff_paths(row["after"]["current"], reconstructed_current)
            if current_paths:
                public_fixture_mismatches += 1
                public_path_mismatches.update(current_paths)
            reconstructed_select = reconstruct_select(cpu_outputs["legal"][index])
            select_paths = diff_paths(row["after"]["select"], reconstructed_select)
            if select_paths:
                legal_fixture_mismatches += 1
                legal_path_mismatches.update(select_paths)
            expected_terminal = bool(row["after"]["terminal"])
            actual_terminal = bool(cpu_outputs["scalar"][index][SCALAR_INDEX["terminal"]])
            if expected_terminal != actual_terminal:
                terminal_fixture_mismatches += 1
            if (current_paths or select_paths or expected_terminal != actual_terminal) and len(first_mismatches) < 8:
                first_mismatches.append({
                    "fixture": index,
                    "public_paths": current_paths[:16],
                    "legal_paths": select_paths[:16],
                    "terminal_expected": expected_terminal,
                    "terminal_actual": actual_terminal,
                })

        oracle_check_failures = sum(
            not all(bool(value) for value in row.get("oracle_checks", {}).values())
            for row in fixtures
        )
        nonzero_fields = {key: value for key, value in field_mismatches.items() if value}
        category_mismatches = {
            "step": max(public_fixture_mismatches, legal_fixture_mismatches, terminal_fixture_mismatches),
            "legal": legal_fixture_mismatches,
            "terminal": terminal_fixture_mismatches,
            "public_state": public_fixture_mismatches,
            "oracle_fixture_semantics": oracle_check_failures,
        }
        exact = not nonzero_fields and not any(category_mismatches.values())
        report["parity"] = {
            "exact": exact,
            "step_exact": category_mismatches["step"] == 0,
            "legal_exact": category_mismatches["legal"] == 0,
            "terminal_exact": category_mismatches["terminal"] == 0,
            "public_state_exact": category_mismatches["public_state"] == 0,
            "category_fixture_mismatches": category_mismatches,
            "field_element_mismatches": field_mismatches,
            "nonzero_field_element_mismatches": nonzero_fields,
            "public_path_mismatches": dict(public_path_mismatches.most_common()),
            "legal_path_mismatches": dict(legal_path_mismatches.most_common()),
            "first_mismatches": first_mismatches,
            "result_d2h_bytes": sum(value.numel() * value.element_size() for value in outputs.values()),
        }
        if not exact:
            report["status"] = "failed"
            raise RuntimeError(
                f"official Basic Energy Attach parity failed: {category_mismatches}; {nonzero_fields}"
            )

        lanes = max(len(fixtures), int(args.benchmark_lanes))
        bench_inputs = repeat_inputs(inputs, lanes)
        bench_outputs = allocate_outputs(lanes, device)
        for _ in range(5):
            launch(bench_inputs, bench_outputs)
        torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        wall_started = time.perf_counter()
        start_event.record()
        for _ in range(args.benchmark_iters):
            launch(bench_inputs, bench_outputs)
        end_event.record()
        torch.cuda.synchronize(device)
        wall_s = time.perf_counter() - wall_started
        kernel_s = start_event.elapsed_time(end_event) / 1000.0
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
