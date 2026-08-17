"""Static browser contracts for server-owned physical-game materialization."""

from __future__ import annotations

import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "replay_inspector" / "web" / "app.js"


def _function_body(source: str, name: str, next_name: str) -> str:
    start = source.index(f"function {name}")
    end = source.index(f"function {next_name}", start + 1)
    return source[start:end]


class ReplayInspectorGameMaterializationWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = APP.read_text(encoding="utf-8")

    def test_browser_requests_only_the_selected_trace_address(self) -> None:
        load_trace_start = self.app.index("async function loadTrace()")
        load_trace_end = self.app.index("async function loadParameter", load_trace_start)
        load_trace = self.app[load_trace_start:load_trace_end]

        self.assertIn(
            "await fetchBaseTraceCached(state.stepIndex, state.stage)", load_trace
        )
        self.assertNotIn("fetchMaterializedBaseTrace", self.app)
        for fragment in (
            "gameMaterialization: null",
            "function gameTraceAddresses()",
            "function ensureGameMaterialization()",
            "function enqueueGameTrace(",
            "async function runGameMaterialization(",
        ):
            self.assertNotIn(fragment, self.app)

    def test_selected_address_joins_its_inflight_request_and_cache_is_bounded(self) -> None:
        cached = _function_body(self.app, "fetchBaseTraceCached", "stagesForStep")

        self.assertIn(
            "if (state.traceFetches.has(key)) return state.traceFetches.get(key);",
            cached,
        )
        self.assertIn("state.traceCache.delete(key);", cached)
        self.assertIn("state.traceCache.set(key, cached);", cached)
        self.assertIn("const MAX_BROWSER_BASE_TRACES = 8;", self.app)
        self.assertIn(
            "while (state.traceCache.size > MAX_BROWSER_BASE_TRACES)", cached
        )
        self.assertIn("new AbortController()", cached)

    def test_navigation_does_not_abort_same_game_work_and_failures_are_retryable(self) -> None:
        cached = _function_body(self.app, "fetchBaseTraceCached", "stagesForStep")
        load_trace_start = self.app.index("async function loadTrace()")
        load_trace_end = self.app.index("async function loadParameter", load_trace_start)
        load_trace = self.app[load_trace_start:load_trace_end]

        self.assertIn("state.traceFetches.delete(key);", cached)
        self.assertIn("state.traceAbortControllers.delete(key);", cached)
        self.assertNotIn("job.failures", self.app)
        self.assertNotIn("abortTraceFetchesExcept", self.app)
        self.assertNotIn("controller.abort", load_trace)

    def test_only_submission_or_game_changes_clear_browser_trace_state(self) -> None:
        clear_cache = _function_body(
            self.app, "clearGameTraceCache", "fetchBaseTraceCached"
        )

        self.assertIn(
            "for (const controller of state.traceAbortControllers.values()) controller.abort();",
            clear_cache,
        )
        self.assertIn("state.traceCache.clear();", clear_cache)
        self.assertIn("state.traceFetches.clear();", clear_cache)
        self.assertEqual(self.app.count("clearGameTraceCache();"), 2)
        self.assertIn("async function selectSubmission", self.app)
        self.assertIn("async function selectGame", self.app)

    def test_playground_scales_remain_a_separate_decision_request(self) -> None:
        influence_start = self.app.index("async function loadDecisionInfluence()")
        influence_end = self.app.index("function resetDecisionInfluence()", influence_start)
        influence = self.app[influence_start:influence_end]

        self.assertIn("fetchJson(decisionInfluencePath(scales))", influence)
        self.assertNotIn("fetchBaseTraceCached", influence)

    def test_browser_has_no_fixed_trace_timeout(self) -> None:
        self.assertNotIn("TRACE_REQUEST_TIMEOUT_MS", self.app)
        self.assertNotIn("Selected trace reconstruction exceeded 20 seconds", self.app)


if __name__ == "__main__":
    unittest.main()
