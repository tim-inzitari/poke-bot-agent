#!/usr/bin/env python3
"""Durable semantic, CUDA-report, and finite-service regression tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import unittest
from collections import Counter
from pathlib import Path


LAB = Path(os.environ.get("CUDA_SIM_LAB", "/home/inzi/cuda-sim-lab"))
OUTPUTS = LAB / "outputs"
FIXTURES = OUTPUTS / "official-basic-energy-attach-fixtures.json"
CUDA_REPORT = OUTPUTS / "official-basic-energy-attach-cuda.json"
UNIT = LAB / "pokebot-cuda-3080-attach-parity-once.service"
OFFICIAL_SHA = "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def reference_current(row: dict[str, object]) -> dict[str, object]:
    before = copy.deepcopy(row["before"]["current"])
    selected = row["selected_option"]
    actor = int(before["yourIndex"])
    player = before["players"][actor]
    source_index = int(selected["index"])
    moved = player["hand"].pop(source_index)
    player["handCount"] -= 1
    target = player["bench"][int(selected["inPlayIndex"])]
    target["energies"].append(6)
    target["energyCards"].append(moved)
    before["energyAttached"] = True
    before["turnActionCount"] += 1
    return before


class OfficialBasicEnergyAttachSliceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_report = json.loads(FIXTURES.read_text())
        cls.cuda_report = json.loads(CUDA_REPORT.read_text())

    def test_official_corpus_provenance_and_diversity(self) -> None:
        report = self.fixture_report
        self.assertEqual(report["schema"], "poke_bot.official_basic_energy_attach_fixtures/v1")
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["official_lib_sha256"], OFFICIAL_SHA)
        self.assertEqual(report["fixture_count"], 256)
        self.assertFalse(report["production_eligible"])
        rows = report["fixtures"]
        self.assertEqual({row["before"]["current"]["yourIndex"] for row in rows}, {0, 1})
        self.assertEqual({row["before"]["current"]["firstPlayer"] for row in rows}, {0, 1})
        self.assertGreaterEqual(len({row["selected_option"]["index"] for row in rows}), 10)
        self.assertGreaterEqual(len({row["selected_option"]["inPlayIndex"] for row in rows}), 3)
        energy_counts = [
            len(
                row["before"]["current"]["players"][row["before"]["current"]["yourIndex"]]
                ["bench"][row["selected_option"]["inPlayIndex"]]["energies"]
            )
            for row in rows
        ]
        self.assertEqual(min(energy_counts), 0)
        self.assertGreaterEqual(max(energy_counts), 30)

    def test_every_official_transition_matches_independent_reference(self) -> None:
        expected_select = {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "option": [{"type": 14}],
            "deck": None,
            "contextCard": None,
            "effect": None,
        }
        for row in self.fixture_report["fixtures"]:
            self.assertEqual(row["official_error"], 0)
            self.assertTrue(all(row["oracle_checks"].values()))
            self.assertEqual(canonical(row["after"]["current"]), canonical(reference_current(row)))
            self.assertEqual(row["after"]["select"], expected_select)
            self.assertFalse(row["after"]["terminal"])
            self.assertEqual(
                row["before"]["public_state_sha256"],
                hashlib.sha256(canonical(row["before"]["current"]).encode()).hexdigest(),
            )
            self.assertEqual(
                row["after"]["public_state_sha256"],
                hashlib.sha256(canonical(row["after"]["current"]).encode()).hexdigest(),
            )

    def test_cuda_report_has_zero_exact_parity_mismatches(self) -> None:
        report = self.cuda_report
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["official_lib_sha256"], OFFICIAL_SHA)
        self.assertEqual(report["official_fixture_count"], 256)
        self.assertEqual(report["gpu_uuid"], "72bf89ff-6e52-01d4-a49b-90a7addfd632")
        self.assertTrue(report["finite_run"])
        parity = report["parity"]
        self.assertTrue(parity["exact"])
        self.assertTrue(parity["step_exact"])
        self.assertTrue(parity["legal_exact"])
        self.assertTrue(parity["terminal_exact"])
        self.assertTrue(parity["public_state_exact"])
        self.assertTrue(all(value == 0 for value in parity["category_fixture_mismatches"].values()))
        self.assertTrue(all(value == 0 for value in parity["field_element_mismatches"].values()))
        self.assertEqual(parity["first_mismatches"], [])
        self.assertFalse(report["full_seeded_game_parity"])
        self.assertFalse(report["full_card_effect_parity"])
        self.assertFalse(report["full_engine_transition_coverage"])
        self.assertFalse(report["production_eligible"])

    def test_benchmark_and_service_are_strictly_finite(self) -> None:
        benchmark = self.cuda_report["finite_benchmark"]
        self.assertEqual(benchmark["lanes"], 65536)
        self.assertEqual(benchmark["iterations"], 50)
        self.assertEqual(benchmark["transitions"], 65536 * 50)
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
