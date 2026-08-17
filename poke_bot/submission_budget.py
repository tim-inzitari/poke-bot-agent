"""Hard-cap-aware search budgeting for Kaggle submission agents.

The competition process has one game-wide wall-clock cap.  This allocator
starts its clock at module import, reserves enough time for ordinary policy
inference on the remaining calls, and can spend only the surplus on trusted
belief MCTS when an explicitly enabled research configuration is evaluated.
Canonical Kaggle submissions keep search disabled and use the frozen neural
policy directly.  The dormant search path falls back for only the current
decision after a failed or shallow search and may retry on the next eligible
decision; only the game clock may force game-wide greedy operation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Mapping, Optional


SCHEMA = "poke_bot.submission_search_config/v1"


@dataclass(frozen=True)
class SubmissionSearchPlan:
    search: bool
    move_time_s: float = 0.0
    max_sims: int = 0
    reason: str = ""


@dataclass
class SubmissionSearchBudget:
    started_at: float
    hard_cap_s: float = 600.0
    internal_deadline_s: float = 540.0
    final_greedy_reserve_s: float = 20.0
    total_search_budget_s: float = 400.0
    baseline_call_s: float = 0.20
    maximum_calls: int = 340
    expected_search_decisions: int = 64
    maximum_move_s: float = 4.0
    minimum_move_s: float = 0.50
    minimum_sims: int = 50
    maximum_sims: int = 50
    search_failure_behavior: str = "greedy_current_decision_then_retry"
    game_wide_greedy_only_for_time_budget: bool = True
    safety_factor: float = 0.80
    enabled: bool = True
    calls_used: int = 0
    searches_used: int = 0
    search_seconds_used: float = 0.0
    simulation_rate_ema: Optional[float] = None
    consecutive_search_failures: int = 0
    disabled_reason: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
        *,
        started_at: Optional[float] = None,
    ) -> "SubmissionSearchBudget":
        if config.get("schema") != SCHEMA:
            raise ValueError("submission search config schema changed")
        budget = cls(
            started_at=float(
                time.monotonic() if started_at is None else started_at
            ),
            enabled=config.get("enabled") is True,
            hard_cap_s=float(config["hard_cap_s"]),
            internal_deadline_s=float(config["internal_deadline_s"]),
            final_greedy_reserve_s=float(config["final_greedy_reserve_s"]),
            total_search_budget_s=float(config["total_search_budget_s"]),
            baseline_call_s=float(config["baseline_call_s"]),
            maximum_calls=int(config["maximum_calls"]),
            expected_search_decisions=int(
                config["expected_search_decisions"]
            ),
            maximum_move_s=float(config["maximum_move_s"]),
            minimum_move_s=float(config["minimum_move_s"]),
            minimum_sims=int(config["minimum_sims"]),
            maximum_sims=int(config["maximum_sims"]),
            search_failure_behavior=str(config["search_failure_behavior"]),
            game_wide_greedy_only_for_time_budget=(
                config.get("game_wide_greedy_only_for_time_budget") is True
            ),
            safety_factor=float(config["safety_factor"]),
        )
        budget._validate()
        return budget

    def _validate(self) -> None:
        if (
            self.hard_cap_s != 600.0
            or not 0 < self.internal_deadline_s < self.hard_cap_s
            or self.final_greedy_reserve_s != 20.0
            or self.internal_deadline_s
            > self.hard_cap_s - self.final_greedy_reserve_s
            or not 0 < self.total_search_budget_s < self.internal_deadline_s
            or self.maximum_calls != 340
            or self.baseline_call_s <= 0
            or self.expected_search_decisions <= 0
            or not 0 < self.minimum_move_s <= self.maximum_move_s
            or not 1 <= self.minimum_sims <= self.maximum_sims
            or not 0 < self.safety_factor < 1
            or self.search_failure_behavior
            != "greedy_current_decision_then_retry"
            or self.game_wide_greedy_only_for_time_budget is not True
        ):
            raise ValueError("unsafe Kaggle submission search budget")

    def reset(self, *, started_at: Optional[float] = None) -> None:
        self.started_at = float(
            time.monotonic() if started_at is None else started_at
        )
        self.calls_used = 0
        self.searches_used = 0
        self.search_seconds_used = 0.0
        self.simulation_rate_ema = None
        self.consecutive_search_failures = 0
        self.disabled_reason = None

    @staticmethod
    def _search_worthy(obs_dict: Mapping[str, Any]) -> bool:
        current = obs_dict.get("current")
        select = obs_dict.get("select")
        if not isinstance(current, Mapping) or not isinstance(select, Mapping):
            return False
        options = list(select.get("option") or ())
        count = len(options)
        minimum = max(0, int(select.get("minCount", 0) or 0))
        maximum = min(count, int(select.get("maxCount", 0) or 0))
        if count <= 1 or maximum <= 0:
            return False
        # Selecting every available option has no strategic branch.
        if minimum == maximum == count:
            return False
        return True

    def plan(
        self,
        obs_dict: Mapping[str, Any],
        *,
        now: Optional[float] = None,
    ) -> SubmissionSearchPlan:
        current_time = float(time.monotonic() if now is None else now)
        self.calls_used += 1
        if not self.enabled:
            return SubmissionSearchPlan(False, reason="disabled_by_config")
        if self.disabled_reason:
            return SubmissionSearchPlan(False, reason=self.disabled_reason)
        if not self._search_worthy(obs_dict):
            return SubmissionSearchPlan(False, reason="forced_or_trivial")

        elapsed = max(0.0, current_time - self.started_at)
        if elapsed >= self.hard_cap_s - self.final_greedy_reserve_s:
            return SubmissionSearchPlan(
                False,
                reason="final_greedy_reserve",
            )
        wall_remaining = self.internal_deadline_s - elapsed
        remaining_calls = max(0, self.maximum_calls - self.calls_used)
        baseline_reserve = remaining_calls * self.baseline_call_s
        search_remaining = min(
            self.total_search_budget_s - self.search_seconds_used,
            wall_remaining - baseline_reserve,
        )
        remaining_searches = max(
            1, self.expected_search_decisions - self.searches_used
        )
        move_time = min(
            self.maximum_move_s,
            search_remaining / remaining_searches,
        )
        if move_time < self.minimum_move_s:
            return SubmissionSearchPlan(False, reason="deadline_reserve")

        sims = self.minimum_sims
        if self.simulation_rate_ema is not None:
            safe_capacity = int(
                math.floor(
                    self.simulation_rate_ema
                    * move_time
                    * self.safety_factor
                )
            )
            if safe_capacity < self.minimum_sims:
                return SubmissionSearchPlan(
                    False,
                    reason="minimum_trusted_sims_do_not_fit",
                )
            sims = min(self.maximum_sims, safe_capacity)
        return SubmissionSearchPlan(
            True,
            move_time_s=float(move_time),
            max_sims=int(sims),
            reason="trusted_belief_mcts",
        )

    def record_search(
        self,
        *,
        elapsed_s: float,
        completed_sims: int,
        succeeded: bool,
    ) -> None:
        elapsed = max(0.0, float(elapsed_s))
        self.search_seconds_used += elapsed
        self.searches_used += 1
        if not succeeded or int(completed_sims) < self.minimum_sims:
            self.consecutive_search_failures += 1
            return
        self.consecutive_search_failures = 0
        rate = float(completed_sims) / max(elapsed, 1e-9)
        self.simulation_rate_ema = (
            rate
            if self.simulation_rate_ema is None
            else 0.70 * self.simulation_rate_ema + 0.30 * rate
        )
