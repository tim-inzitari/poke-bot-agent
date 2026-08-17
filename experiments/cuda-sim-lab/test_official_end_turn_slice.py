#!/usr/bin/env python3
"""Durable semantic/report regression tests for the clean End CUDA slice."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path("/home/pokebot/cuda-sim-lab/outputs")
FIXTURES = ROOT / "official-clean-end-turn-fixtures.json"
CUDA_REPORT = ROOT / "official-clean-end-turn-cuda.json"
OFFICIAL_SHA = "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"


def reference(before: dict[str, object]) -> dict[str, int]:
    current = int(before["your_index"])
    next_player = 1 - current
    d0, d1 = int(before["deck_count_0"]), int(before["deck_count_1"])
    h0, h1 = int(before["hand_count_0"]), int(before["hand_count_1"])
    next_deck = d0 if next_player == 0 else d1
    terminal = next_deck == 0
    draw0 = next_player == 0 and not terminal
    draw1 = next_player == 1 and not terminal
    return {
        "turn": int(before["turn"]) + 1,
        "your_index": current if terminal else next_player,
        "first_player": int(before["first_player"]),
        "result": current if terminal else -1,
        "deck_count_0": d0 - int(draw0),
        "deck_count_1": d1 - int(draw1),
        "hand_count_0": h0 + int(draw0),
        "hand_count_1": h1 + int(draw1),
        "select_type": 0,
        "select_context": 0,
        "select_min": 1,
        "select_max": 1,
        "terminal": int(terminal),
    }


class OfficialEndTurnSliceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_report = json.loads(FIXTURES.read_text())
        cls.cuda_report = json.loads(CUDA_REPORT.read_text())

    def test_corpus_provenance_and_coverage(self) -> None:
        report = self.fixture_report
        self.assertEqual(report["official_lib_sha256"], OFFICIAL_SHA)
        self.assertEqual(report["games"], 16)
        self.assertEqual(report["terminal_games"], 16)
        self.assertGreaterEqual(report["fixture_count"], 1500)
        self.assertEqual(report["terminal_transition_count"], 16)
        self.assertEqual(
            {row["before"]["first_player"] for row in report["fixtures"]},
            {0, 1},
        )

    def test_every_official_transition_matches_reference(self) -> None:
        for row in self.fixture_report["fixtures"]:
            expected = reference(row["before"])
            actual = row["after"]
            for key, value in expected.items():
                observed = int(actual["result"] != -1) if key == "terminal" else int(actual[key])
                self.assertEqual(observed, value, (row["game"], row["step"], key))

    def test_cuda_differential_report_is_exact_and_scoped(self) -> None:
        report = self.cuda_report
        self.assertEqual(report["official_lib_sha256"], OFFICIAL_SHA)
        self.assertTrue(report["parity"]["exact"])
        self.assertTrue(all(value == 0 for value in report["parity"]["field_mismatches"].values()))
        self.assertFalse(report["full_engine_transition_coverage"])
        self.assertFalse(report["production_eligible"])
        self.assertEqual(report["throughput"]["measured_bulk_h2d_bytes"], 0)
        self.assertEqual(report["throughput"]["measured_bulk_d2h_bytes"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
