#!/usr/bin/env python3
"""Regression tests for the official Basic Bench Play CUDA slice."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import unittest
from pathlib import Path


LAB = Path(os.environ.get("CUDA_SIM_LAB", "/home/inzi/cuda-sim-lab"))
OUTPUTS = LAB / "outputs"
FIXTURES = OUTPUTS / "official-basic-bench-play-fixtures.json"
CUDA_REPORT = OUTPUTS / "official-basic-bench-play-cuda.json"
UNIT = LAB / "pokebot-cuda-3080-bench-play-parity-once.service"
OFFICIAL_SHA = "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def reference_current(row: dict[str, object]) -> dict[str, object]:
    current = copy.deepcopy(row["before"]["current"])
    actor = int(current["yourIndex"])
    player = current["players"][actor]
    source_index = int(row["selected_option"]["index"])
    moved = player["hand"].pop(source_index)
    player["handCount"] -= 1
    player["bench"].append({
        "id": moved["id"],
        "serial": moved["serial"],
        "playerIndex": moved["playerIndex"],
        "hp": 90,
        "maxHp": 90,
        "appearThisTurn": True,
        "energies": [],
        "energyCards": [],
        "tools": [],
        "preEvolution": [],
    })
    current["turnActionCount"] += 1
    return current


def expected_options(current: dict[str, object]) -> list[dict[str, int]]:
    actor = int(current["yourIndex"])
    player = current["players"][actor]
    in_play = [(4, 0)] + [(5, index) for index in range(len(player["bench"]))]
    options: list[dict[str, int]] = []
    for hand_index, card in enumerate(player["hand"]):
        if int(card["id"]) == 22 and len(player["bench"]) < int(player["benchMax"]):
            options.append({"type": 7, "index": hand_index})
        elif int(card["id"]) == 6 and not bool(current["energyAttached"]):
            for area, index in in_play:
                options.append({
                    "type": 8,
                    "area": 2,
                    "index": hand_index,
                    "inPlayArea": area,
                    "inPlayIndex": index,
                })
    options.append({"type": 14})
    return options


def reference_select(current: dict[str, object]) -> dict[str, object]:
    return {
        "type": 0,
        "context": 0,
        "minCount": 1,
        "maxCount": 1,
        "remainDamageCounter": 0,
        "remainEnergyCost": 0,
        "option": expected_options(current),
        "deck": None,
        "contextCard": None,
        "effect": None,
    }


class OfficialBasicBenchPlaySliceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_report = json.loads(FIXTURES.read_text())
        cls.cuda_report = json.loads(CUDA_REPORT.read_text())

    def test_official_corpus_provenance_and_diversity(self) -> None:
        report = self.fixture_report
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["official_lib_sha256"], OFFICIAL_SHA)
        self.assertEqual(report["fixture_count"], 256)
        self.assertFalse(report["production_eligible"])
        rows = report["fixtures"]
        self.assertEqual({row["before"]["current"]["yourIndex"] for row in rows}, {0, 1})
        self.assertEqual({row["before"]["current"]["firstPlayer"] for row in rows}, {0, 1})
        self.assertGreaterEqual(len({row["selected_option"]["index"] for row in rows}), 40)
        self.assertEqual(
            {len(row["before"]["current"]["players"][row["before"]["current"]["yourIndex"]]["bench"]) for row in rows},
            {0, 1, 2},
        )
        legal_counts = [len(row["after"]["select"]["option"]) for row in rows]
        self.assertLessEqual(min(legal_counts), 15)
        self.assertGreaterEqual(max(legal_counts), 200)

    def test_every_official_transition_matches_independent_reference(self) -> None:
        for row in self.fixture_report["fixtures"]:
            self.assertEqual(row["official_error"], 0)
            self.assertTrue(all(row["oracle_checks"].values()))
            expected_current = reference_current(row)
            self.assertEqual(canonical(row["after"]["current"]), canonical(expected_current))
            self.assertEqual(row["after"]["select"], reference_select(expected_current))
            self.assertFalse(row["after"]["terminal"])
            self.assertEqual(
                row["after"]["public_state_sha256"],
                hashlib.sha256(canonical(row["after"]["current"]).encode()).hexdigest(),
            )

    def test_cuda_report_exact_with_complete_legal_rebuild(self) -> None:
        report = self.cuda_report
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["official_lib_sha256"], OFFICIAL_SHA)
        self.assertEqual(report["official_fixture_count"], 256)
        self.assertEqual(report["gpu_uuid"], "72bf89ff-6e52-01d4-a49b-90a7addfd632")
        parity = report["parity"]
        self.assertTrue(parity["exact"])
        self.assertTrue(parity["step_exact"])
        self.assertTrue(parity["legal_exact"])
        self.assertTrue(parity["terminal_exact"])
        self.assertTrue(parity["public_state_exact"])
        self.assertTrue(all(value == 0 for value in parity["category_fixture_mismatches"].values()))
        self.assertTrue(all(value == 0 for value in parity["field_element_mismatches"].values()))
        self.assertEqual(parity["first_mismatches"], [])
        self.assertEqual(report["implemented_state"]["legal_option_capacity"], 384)
        self.assertFalse(report["full_seeded_game_parity"])
        self.assertFalse(report["full_card_effect_parity"])
        self.assertFalse(report["full_engine_transition_coverage"])
        self.assertFalse(report["production_eligible"])

    def test_benchmark_and_service_are_finite(self) -> None:
        benchmark = self.cuda_report["finite_benchmark"]
        self.assertEqual(benchmark["lanes"], 16384)
        self.assertEqual(benchmark["iterations"], 25)
        self.assertEqual(benchmark["transitions"], 16384 * 25)
        self.assertEqual(benchmark["measured_bulk_h2d_bytes"], 0)
        self.assertEqual(benchmark["measured_bulk_d2h_bytes"], 0)
        unit = UNIT.read_text()
        self.assertIn("Type=oneshot", unit)
        self.assertIn("TimeoutStartSec=300", unit)
        self.assertIn("GPU-72bf89ff-6e52-01d4-a49b-90a7addfd632", unit)
        self.assertNotIn("Restart=always", unit)
        self.assertNotIn("cuda_simulator_stage2.py", unit)
        self.assertNotIn("[Install]", unit)


if __name__ == "__main__":
    unittest.main(verbosity=2)
