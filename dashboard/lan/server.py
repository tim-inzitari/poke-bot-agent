#!/usr/bin/env python3
"""Small dependency-free LAN dashboard server intended to live on Bert."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import subprocess
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
INDEX = ROOT / "index.html"
REMOTE_SNAPSHOT = "/home/inzi/poke-bot-agent/scripts/run_live_dashboard_snapshot.py"
REMOTE_PYTHON = "/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
LOCAL_SNAPSHOT = ROOT / "fleet_host_snapshot.py"
RATE_STATE = ROOT / "fleet_rate_state.json"
GOAL_PROJECTION = ROOT / "current_goal_requirements.json"
UI_VERSION_TOKEN = b"__DASHBOARD_UI_VERSION__"


def dashboard_ui_version() -> str:
    """Return a content identity shared by the page and status endpoint."""
    return hashlib.sha256(INDEX.read_bytes()).hexdigest()[:16]


def rendered_index() -> bytes:
    source = INDEX.read_bytes()
    return source.replace(UI_VERSION_TOKEN, dashboard_ui_version().encode("ascii"))


class SnapshotCache:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.value: dict[str, Any] = {
            "ok": False,
            "error": "waiting for first telemetry sample",
            "dashboard_host": socket.gethostname(),
        }
        self.stopping = threading.Event()
        self.rate_history: dict[str, list[tuple[float, float, str]]] = {}
        self.decision_density: dict[str, float] = {}
        self.last_valid_rates: dict[str, dict[str, Any]] = {}
        self._last_rate_state_save = 0.0
        self.network_latency: dict[str, Any] = {}
        self._last_latency_sample = 0.0
        try:
            candidate = json.loads(RATE_STATE.read_text())
            if isinstance(candidate, dict):
                self.last_valid_rates = {
                    str(key): item
                    for key, item in candidate.items()
                    if isinstance(item, dict)
                }
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            pass

    def _persist_rate_state(self) -> None:
        """Persist phase-bound last-good rates across dashboard restarts."""
        now = time.monotonic()
        if now - self._last_rate_state_save < 5.0:
            return
        self._last_rate_state_save = now
        temporary = RATE_STATE.with_suffix(".tmp")
        try:
            temporary.write_text(json.dumps(self.last_valid_rates, separators=(",", ":")))
            temporary.replace(RATE_STATE)
        except OSError:
            pass

    @staticmethod
    def _apply_goal_projection(value: dict[str, Any]) -> None:
        """Overlay owner planning changes without mutating the live runtime.

        Execution facts still come from the selector-owned remote snapshot.
        This local compatibility projection is limited to the user-facing
        required-goal roster, deck naming, and prospective typed design
        contracts, which may advance while the active learner intentionally
        remains on an immutable runtime tree.
        """

        try:
            projection = json.loads(GOAL_PROJECTION.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return
        overrides = projection.get("current_owner_overrides") or {}
        plan = overrides.get("required_specialist_plan") or {}
        teal = overrides.get("teal_mask_ogerpon_ex") or {}
        version_namespaces = overrides.get("version_namespaces") or {}
        guide_projection = (
            (projection.get("verified_snapshot") or {}).get(
                "current_deck_guides"
            )
            or {}
        )
        guide_policy = guide_projection.get("goal_path_guidance") or {}
        future_guide_scope = (
            overrides.get("future_guide_strategic_branch_scope") or {}
        )
        future_head_action = future_guide_scope
        future_setup_head = (
            future_guide_scope.get("setup_board_outcome_head") or {}
        )
        teal_legacy_guide = (
            overrides.get("teal_guide_weight_nonwinning_reduction") or {}
        )
        slowking_replacement = (
            overrides.get("slowking_specialist_replacement") or {}
        )
        slowking_plan = slowking_replacement.get("slowking") or {}
        crustle_plan = slowking_replacement.get("crustle") or {}
        post_fleet = (
            overrides.get("post_fleet_alakazam_grimms_refresh") or {}
        )
        slowking_failure = (
            overrides.get("slowking_failed_experiment_alakazam_transition")
            or {}
        )
        protocol = value.get("specialist_protocol")
        if not isinstance(protocol, dict):
            return
        priority = protocol.get("training_priority")
        if not isinstance(priority, dict):
            priority = {}
            protocol["training_priority"] = priority
        removed = [
            str(item)
            for item in (plan.get("removed_specialist_ids") or [])
            if str(item)
        ]
        prefix = [
            str(item)
            for item in (plan.get("strict_post_spidops_prefix") or [])
            if str(item)
        ]
        planned_order = [
            str(item)
            for item in (
                plan.get("ordered_unfinished_ids_after_active") or []
            )
            if str(item)
        ]
        if prefix:
            priority["strict_post_spidops_prefix"] = {
                "decision_revision": plan.get("goal_revision"),
                "ids": prefix,
                "missing_input_behavior": plan.get(
                    "missing_strict_prefix_input_behavior"
                ),
                "activation": plan.get("activation_boundary"),
                "source": "dashboard_goal_compatibility_projection",
            }
        if removed:
            priority["owner_removal"] = {
                "decision_revision": plan.get("goal_revision"),
                "specialist_ids": removed,
                "selection_eligible": bool(
                    plan.get("removed_ids_selection_eligible")
                ),
                "counts_toward_completion": bool(
                    plan.get("removed_ids_count_toward_completion")
                ),
                "preserve_historical_corpus_router_and_audit_artifacts": bool(
                    plan.get("removed_ids_historical_artifacts_preserved")
                ),
                "source": "dashboard_goal_compatibility_projection",
            }
            if planned_order:
                removed_set = set(removed)
                priority["ordered_unfinished_ids_after_active"] = [
                    item for item in planned_order if item not in removed_set
                ]
            else:
                ordered = priority.get("ordered_unfinished_ids_after_active")
                if isinstance(ordered, list):
                    removed_set = set(removed)
                    priority["ordered_unfinished_ids_after_active"] = [
                        str(item)
                        for item in ordered
                        if str(item) and str(item) not in removed_set
                    ]
        elif planned_order:
            priority["ordered_unfinished_ids_after_active"] = planned_order
        rows = protocol.get("specialists")
        if isinstance(rows, list):
            removed_set = set(removed)
            filtered = [
                row
                for row in rows
                if isinstance(row, dict)
                and str(row.get("id") or "") not in removed_set
            ]
            if bool(slowking_plan.get("required_specialist")) and not any(
                str(row.get("id") or "") == "slowking"
                for row in filtered
            ):
                filtered.append(
                    {
                        "id": "slowking",
                        "name": "Slowking Combo / Toolbox",
                        "status": "blocked",
                        "planning_status": slowking_plan.get("status"),
                        "required_specialist": True,
                        "completion_eligible": bool(
                            slowking_plan.get("completion_eligible")
                        ),
                        "selector_eligible": bool(
                            slowking_plan.get("selection_eligible")
                        ),
                        "active": False,
                        "frozen": False,
                        "training_order": {
                            "predecessor": slowking_plan.get(
                                "predecessor_specialist"
                            ),
                            "successor": slowking_plan.get(
                                "successor_specialist"
                            ),
                            "strict_order": slowking_plan.get(
                                "strict_post_teal_order"
                            ),
                            "after_completion": slowking_plan.get(
                                "after_completion"
                            ),
                        },
                        "combo_head_coverage": slowking_plan.get(
                            "combo_head_coverage"
                        ),
                        "specialist_parameter_budget": slowking_plan.get(
                            "specialist_parameter_budget"
                        ),
                        "projection_only": True,
                        "source": (
                            "dashboard_goal_compatibility_projection"
                        ),
                    }
                )
            for row in filtered:
                if (
                    row.get("id") == "slowking"
                    and slowking_failure.get("status") == "activated"
                ):
                    row["status"] = "failed_experiment"
                    row["active"] = False
                    row["frozen"] = False
                    row["public_mix_eligible"] = False
                    row["terminal_disposition"] = (
                        slowking_failure.get("terminal_disposition")
                    )
                    row["terminal_receipt"] = (
                        slowking_failure.get("failed_experiment_receipt")
                    )
                if row.get("id") == "teal-mask-ogerpon-ex":
                    row["name"] = str(
                        teal.get("display_name")
                        or "Slop Box (Teal Mask Ogerpon ex)"
                    )
                    row["deck_family_name"] = teal.get("deck_family_name")
                    row["secondary_search_alias"] = teal.get(
                        "secondary_search_alias"
                    )
            protocol["specialists"] = filtered
            counts: dict[str, int] = {}
            for row in filtered:
                status = str(row.get("status") or "")
                counts[status] = counts.get(status, 0) + 1
            protocol["status_counts"] = counts
        retained_opponents = [
            row
            for row in (
                protocol.get("retained_non_specialist_opponents") or []
            )
            if isinstance(row, dict)
            and str(row.get("id") or "") != "crustle"
        ]
        crustle_opponent = dict(
            crustle_plan.get("public_practice_gate_opponent") or {}
        )
        if (
            crustle_plan.get("required_specialist") is False
            and crustle_plan.get("matchup_route_preserved") is True
            and crustle_opponent
        ):
            retained_opponents.append(
                {
                    "id": "crustle",
                    "name": "Crustle",
                    "display_status": str(
                        crustle_plan.get("display_status")
                        or (
                            "historical_artifacts_preserved_"
                            "inference_only_not_planned_for_training"
                        )
                    ),
                    "role_label": (
                        "PUBLIC OPPONENT + ACTIVE ROUTE, "
                        "NO SPECIALIST TRAIN"
                    ),
                    "required_specialist": False,
                    "selection_eligible": False,
                    "completion_eligible": False,
                    "training_authorized": False,
                    "submission_authorized": False,
                    "matchup_route_preserved": True,
                    "stable_matchup_slot": crustle_plan.get(
                        "stable_matchup_slot"
                    ),
                    "stable_matchup_slot_status": crustle_plan.get(
                        "stable_matchup_slot_status"
                    ),
                    "public_practice_gate_opponent": crustle_opponent,
                    "inference_only": (
                        crustle_opponent.get("inference_only") is True
                    ),
                    "historical_artifacts_preserved": True,
                    "future_specialist_training_planned": False,
                    "projection_only": True,
                    "source": "dashboard_goal_compatibility_projection",
                }
            )
        protocol["retained_non_specialist_opponents"] = retained_opponents
        if post_fleet:
            protocol["post_fleet_refresh"] = {
                "goal_revision": post_fleet.get("goal_revision"),
                "phase_id": post_fleet.get("phase_id"),
                "status": post_fleet.get("status"),
                "trigger": dict(post_fleet.get("trigger") or {}),
                "ordered_specialist_ids": list(
                    post_fleet.get("ordered_specialist_ids") or []
                ),
                "order_is_strict": (
                    post_fleet.get("order_is_strict") is True
                ),
                "release_gates": dict(
                    post_fleet.get("release_gates") or {}
                ),
                "first_refresh": dict(
                    post_fleet.get("first_refresh") or {}
                ),
                "core_hot_start_selector": post_fleet.get(
                    "core_hot_start_selector"
                ),
                "first_alakazam_prefer_compatible_immutable_alakazam_migration": (
                    post_fleet.get(
                        "first_alakazam_prefer_compatible_"
                        "immutable_alakazam_migration"
                    )
                    is True
                ),
                "first_alakazam_migration_failure_fallback": (
                    post_fleet.get(
                        "first_alakazam_migration_failure_fallback"
                    )
                ),
                "source": "dashboard_goal_compatibility_projection",
            }
        if slowking_failure:
            protocol["terminal_specialist_transition"] = {
                **slowking_failure,
                "source": "dashboard_goal_compatibility_projection",
            }
        required = plan.get("required_specialists_total")
        if isinstance(required, int) and required > 0:
            protocol["required_target_count"] = required
            projected_rows = protocol.get("specialists")
            if isinstance(projected_rows, list):
                frozen_rows = [
                    row
                    for row in projected_rows
                    if isinstance(row, dict)
                    and (
                        str(row.get("status") or "") == "passed_frozen"
                        or bool(row.get("frozen"))
                    )
                ]
                active_id = str(protocol.get("active_specialist") or "")
                active_rows = [
                    row
                    for row in projected_rows
                    if isinstance(row, dict)
                    and (
                        bool(row.get("active"))
                        or (
                            active_id
                            and str(row.get("id") or "") == active_id
                        )
                    )
                    and row not in frozen_rows
                ]
                progress = protocol.get("program_progress")
                if not isinstance(progress, dict):
                    progress = {}
                    protocol["program_progress"] = progress
                frozen_count = len(frozen_rows)
                active_count = min(len(active_rows), 1)
                unfinished_count = max(required - frozen_count, 0)
                remaining_after_active = max(
                    unfinished_count - active_count, 0
                )
                progress.update(
                    {
                        "required_specialists_total": required,
                        "completed_frozen": frozen_count,
                        "completed_specialist_ids": [
                            str(row.get("id") or "")
                            for row in frozen_rows
                            if str(row.get("id") or "")
                        ],
                        "active_specialists": active_count,
                        "active_specialist_ids": [
                            str(row.get("id") or "")
                            for row in active_rows[:1]
                            if str(row.get("id") or "")
                        ],
                        "remaining_unfinished": unfinished_count,
                        "remaining_after_active": remaining_after_active,
                        "population_transition_ready": (
                            unfinished_count == 0
                        ),
                    }
                )
                next_action = str(protocol.get("next_action") or "")
                progress_text = (
                    f"{unfinished_count} specialists remain unfinished "
                    "including the active specialist; "
                    f"{remaining_after_active} remain after it."
                )
                if re.search(
                    r"\d+ specialists remain unfinished including the "
                    r"active specialist; \d+ remain after it\.",
                    next_action,
                ):
                    next_action = re.sub(
                        r"\d+ specialists remain unfinished including the "
                        r"active specialist; \d+ remain after it\.",
                        progress_text,
                        next_action,
                    )
                elif next_action:
                    next_action = f"{next_action.rstrip()} {progress_text}"
                else:
                    next_action = progress_text
                next_action = re.sub(
                    r"all \d+ specialists",
                    f"all {required} specialists",
                    next_action,
                )
                protocol["next_action"] = next_action
        if isinstance(guide_policy, dict) and guide_policy:
            protocol["current_deck_guide_weight_policy"] = {
                **guide_policy,
                "guide_curriculum_revision": future_guide_scope.get(
                    "guide_curriculum_revision"
                ),
                "strategic_branch_scope_revision": future_guide_scope.get(
                    "strategic_branch_scope_revision"
                ),
                "head_action_scope_revision": future_guide_scope.get(
                    "head_action_scope_revision"
                ),
                "learning_effect": (
                    "literal_multiplier_on_bounded_guide_conditioned_"
                    "strategic_head_curriculum"
                ),
                "gradient_effect": (
                    "scales_guide_conditioned_strategic_head_gradient_"
                    "contribution"
                ),
                "direct_policy_cross_entropy_allowed": bool(
                    future_guide_scope.get(
                        "direct_policy_cross_entropy_allowed"
                    )
                ),
                "bootstrap_weight_ramp": guide_projection.get(
                    "bootstrap_weight_ramp"
                ),
                "bootstrap_maximum_weight": guide_projection.get(
                    "maximum_weight"
                ),
                "bootstrap_maximum_weight_scope": guide_projection.get(
                    "maximum_weight_scope"
                ),
                "maximum_post_bootstrap_auxiliary_weight": (
                    guide_projection.get(
                        "maximum_post_bootstrap_auxiliary_weight"
                    )
                ),
                "post_bootstrap_behavior": guide_projection.get(
                    "post_bootstrap_behavior"
                ),
                "source": "dashboard_goal_compatibility_projection",
            }
        if isinstance(future_guide_scope, dict) and future_guide_scope:
            legacy_weight = teal_legacy_guide.get(
                "active_iteration_13_weight"
            )
            if legacy_weight is None:
                legacy_weight = teal_legacy_guide.get("target_weight")
            protocol["current_deck_guide_training_modes"] = {
                "active_started_lineage": {
                    "specialist_id": "teal-mask-ogerpon-ex",
                    "display_name": "Slop Box (Teal Mask Ogerpon ex)",
                    "is_active": (
                        str(protocol.get("active_specialist") or "")
                        == "teal-mask-ogerpon-ex"
                    ),
                    "scope": "already_started_legacy_run",
                    "mode": "confidence_weighted_policy_cross_entropy",
                    "guide_weight": legacy_weight,
                    "revision_51_retrofit_allowed": False,
                    "runtime_input_authority": False,
                    "action_selection_authority": False,
                    "serving_authority": False,
                },
                "future_lineage": {
                    "scope": future_guide_scope.get("scope"),
                    "effective_from_specialist": future_guide_scope.get(
                        "prospective_effective_specialist"
                    ),
                    "guide_curriculum_revision": future_guide_scope.get(
                        "guide_curriculum_revision"
                    ),
                    "mode": future_guide_scope.get(
                        "training_target_mode"
                    ),
                    "direct_policy_cross_entropy_allowed": bool(
                        future_guide_scope.get(
                            "direct_policy_cross_entropy_allowed"
                        )
                    ),
                    "guide_runtime_input_allowed": bool(
                        future_guide_scope.get(
                            "guide_runtime_input_allowed"
                        )
                    ),
                    "guide_action_selection_allowed": bool(
                        future_guide_scope.get(
                            "guide_action_selection_allowed"
                        )
                    ),
                    "replace_observed_outcome_targets_allowed": bool(
                        future_guide_scope.get(
                            "replace_observed_outcome_targets_allowed"
                        )
                    ),
                    "curriculum_focus": future_guide_scope.get(
                        "curriculum_focus"
                    ),
                    "fused_policy_learning_authority": (
                        future_guide_scope.get(
                            "fused_policy_learning_authority"
                        )
                    ),
                    "activation_requires_prestage_validation_receipt": (
                        future_guide_scope.get(
                            "activation_requires_prestage_validation_receipt"
                        )
                        is True
                    ),
                },
                "future_head_action_contract": {
                    "head_action_scope_revision": future_guide_scope.get(
                        "head_action_scope_revision"
                    ),
                    "all_future_heads_must_influence_actions": (
                        future_guide_scope.get(
                            "all_future_heads_must_influence_actions"
                        )
                        is True
                    ),
                    "owner_decision_revision": future_head_action.get(
                        "owner_decision_revision"
                    ),
                    "schema": future_head_action.get(
                        "decision_fusion_schema"
                    ),
                    "preserve_v1_additive_residual": (
                        future_head_action.get(
                            "parent_v1_fusion_residual_preserved"
                        )
                        is True
                    ),
                    "computation_role": future_head_action.get(
                        "required_computation_role"
                    ),
                    "fusion_role": (
                        (future_head_action.get("allowed_fusion_roles") or [])
                        or [None]
                    )[0],
                    "action_influence": future_head_action.get(
                        "required_action_influence"
                    ),
                    "state_head_action_conditioning": (
                        future_head_action.get(
                            "state_head_action_conditioning"
                        )
                    ),
                    "option_head_action_conditioning": (
                        future_head_action.get(
                            "option_head_action_conditioning"
                        )
                    ),
                    "route_architecture": future_head_action.get(
                        "action_route_granularity"
                    ),
                    "existing_learned_decision_source_count": (
                        future_head_action.get(
                            "existing_learned_decision_source_count"
                        )
                    ),
                    "canonical_learned_decision_source_count_with_setup": (
                        future_head_action.get(
                            "canonical_learned_decision_source_count_"
                            "with_setup"
                        )
                    ),
                    "setup_source_included_when_present": (
                        future_head_action.get(
                            "setup_source_included_when_present"
                        )
                        is True
                    ),
                    "guide_is_sole_no_route_exception": (
                        future_head_action.get(
                            "guide_is_only_action_route_exception"
                        )
                        is True
                    ),
                    "route_reduction": future_head_action.get(
                        "route_aggregation"
                    ),
                    "aggregate_absolute_cap": future_head_action.get(
                        "aggregate_route_delta_logit_cap"
                    ),
                    "zero_safe_final_projections": (
                        future_head_action.get(
                            "route_final_projection_initialization"
                        )
                        == "exact_zero"
                    ),
                    "independent_means_pre_fusion_computation_not_action_isolation": (
                        future_head_action.get(
                            "independent_means_pre_fusion_computation_"
                            "not_action_isolation"
                        )
                        is True
                    ),
                    "direct_action_selection_authority": (
                        future_head_action.get(
                            "direct_action_selection_authority"
                        )
                        is True
                    ),
                    "fusion_selects_action": (
                        future_head_action.get("fusion_selects_action")
                        is True
                    ),
                    "materially_influences_fused_logits": (
                        future_head_action.get(
                            "materially_influences_fused_logits"
                        )
                        is True
                    ),
                    "runtime_enabled": (
                        future_head_action.get("runtime_enabled") is True
                    ),
                    "runtime_activation_requirement": (
                        future_head_action.get(
                            "runtime_activation_requirement"
                        )
                    ),
                    "setup_board_outcome_head": {
                        "id": future_setup_head.get("id"),
                        "owner_decision_revision": future_setup_head.get(
                            "owner_decision_revision"
                        ),
                        "computation_role": future_setup_head.get(
                            "computation_role"
                        ),
                        "fusion_role": future_setup_head.get("fusion_role"),
                        "action_influence": future_setup_head.get(
                            "action_influence"
                        ),
                        "causal_input": future_setup_head.get("causal_input"),
                        "fusion_route_initialization": (
                            future_setup_head.get(
                                "fusion_route_initialization"
                            )
                        ),
                    },
                },
                "source": "dashboard_goal_compatibility_projection",
            }
        protocol["goal_projection"] = {
            "source": str(GOAL_PROJECTION),
            "goal_revision": max(
                int(plan.get("goal_revision") or 0),
                int(teal.get("goal_revision") or 0),
                int(version_namespaces.get("goal_revision") or 0),
                int(post_fleet.get("goal_revision") or 0),
                int(future_guide_scope.get("goal_revision") or 0),
                int(slowking_replacement.get("goal_revision") or 0),
                int(slowking_failure.get("goal_revision") or 0),
            ),
            "execution_facts_overridden": False,
        }

    @staticmethod
    def _ping_average_ms(command: list[str]) -> float | None:
        try:
            proc = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=6,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        match = re.search(
            r"(?:round-trip|rtt)[^=]*=\s*[^/]+/([^/]+)/", proc.stdout
        )
        return float(match.group(1)) if match else None

    def _refresh_network_latency(self) -> dict[str, Any]:
        now = time.time()
        if self.network_latency and now - self._last_latency_sample < 600.0:
            return dict(self.network_latency)
        commands = {
            "inzi_to_elmo_ms": [
                "/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                "inzi@192.168.1.151", "ping", "-c", "3", "-W", "3", "192.168.1.143",
            ],
            "inzi_to_bert_ms": [
                "/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
                "inzi@192.168.1.151", "ping", "-c", "3", "-W", "3", "192.168.1.158",
            ],
            "bert_to_elmo_ms": [
                "/sbin/ping", "-c", "3", "-W", "3000", "192.168.1.143",
            ],
        }
        samples: dict[str, float | None] = {}
        sample_lock = threading.Lock()

        def sample(key: str, command: list[str]) -> None:
            result = self._ping_average_ms(command)
            with sample_lock:
                samples[key] = result

        threads = [
            threading.Thread(target=sample, args=(key, command), daemon=True)
            for key, command in commands.items()
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=7.0)
        self._last_latency_sample = now
        self.network_latency = {
            **samples,
            "sampled_at": now,
            "refresh_interval_s": 600,
        }
        return dict(self.network_latency)

    def _retain_phase_rate(
        self,
        host_key: str,
        *,
        identity: str,
        sampled_at: float,
        gps: float | None,
        sps: float | None,
        source: str,
        estimated: bool,
    ) -> tuple[float | None, float | None, str, bool, bool, float | None]:
        """Hold the last positive rate through counter idle/gaps and phases.

        A fixed dispatch lets a fast host finish its assigned jobs before the
        slower local tail.  Its monotonic counter then correctly stops, but a
        zero/blank dashboard card looks like broken telemetry.  Preserve the
        last measured active rate, explicitly marked held and aged.  The next
        positive counter sample atomically replaces it.
        """
        useful = bool(
            (isinstance(gps, (int, float)) and gps > 0)
            or (isinstance(sps, (int, float)) and sps > 0)
        )
        if useful:
            cached = {
                "identity": identity,
                "sampled_at": float(sampled_at),
                "gps": gps,
                "sps": sps,
                "source": source,
                "estimated": bool(estimated),
            }
            self.last_valid_rates[host_key] = cached
            self._persist_rate_state()
            return gps, sps, source, estimated, True, 0.0

        cached = self.last_valid_rates.get(host_key) or {}
        held_gps = cached.get("gps")
        held_sps = cached.get("sps")
        if not isinstance(held_gps, (int, float)) and not isinstance(
            held_sps, (int, float)
        ):
            return gps, sps, source, estimated, False, None
        age_s = max(0.0, float(sampled_at) - float(cached.get("sampled_at") or sampled_at))
        same_phase = cached.get("identity") == identity
        held_scope = "same-phase" if same_phase else "prior-phase"
        return (
            float(held_gps) if isinstance(held_gps, (int, float)) else gps,
            float(held_sps) if isinstance(held_sps, (int, float)) else sps,
            f"last active {held_scope} rate (telemetry idle; {cached.get('source') or 'worker counters'})",
            bool(cached.get("estimated")),
            False,
            age_s,
        )

    def _retain_collection_sps(
        self,
        *,
        identity: str,
        sampled_at: float,
        sps: float | None,
        source: str,
    ) -> tuple[float | None, bool, float | None, str]:
        """Bridge short SPS-only gaps without carrying rates across phases.

        The collector's tqdm frame and the scheduler's wave summary are
        sampled asynchronously.  A frame can therefore contain a valid GPS
        and ``sps=0`` even though the immediately preceding frame contained a
        measured SPS.  Retain only the last *positive* SPS for the exact same
        run/iteration/stage; a new phase starts blank instead of inheriting a
        stale rate.
        """
        key = "fleet:collection-sps"
        if isinstance(sps, (int, float)) and float(sps) > 0.0:
            measured = float(sps)
            self.last_valid_rates[key] = {
                "identity": identity,
                "sampled_at": float(sampled_at),
                "sps": measured,
                "source": source,
            }
            self._persist_rate_state()
            return measured, False, 0.0, source

        cached = self.last_valid_rates.get(key) or {}
        cached_sps = cached.get("sps")
        if (
            cached.get("identity") == identity
            and isinstance(cached_sps, (int, float))
            and float(cached_sps) > 0.0
        ):
            age_s = max(
                0.0,
                float(sampled_at)
                - float(cached.get("sampled_at") or sampled_at),
            )
            return (
                float(cached_sps),
                True,
                age_s,
                "last positive same-phase SPS; live telemetry sample gap",
            )
        return sps, False, None, source

    def _counter_rate(
        self,
        key: str,
        *,
        sampled_at: float,
        counter: float | int | None,
        identity: str,
        window_s: float = 15.0,
    ) -> float | None:
        if not isinstance(counter, (int, float)):
            return None
        now = float(sampled_at)
        value = float(counter)
        history = self.rate_history.setdefault(key, [])
        if history and (history[-1][2] != identity or value < history[-1][1]):
            history.clear()
        history.append((now, value, identity))
        cutoff = now - max(2.0, float(window_s))
        while len(history) > 2 and history[1][0] < cutoff:
            history.pop(0)
        first = history[0]
        last = history[-1]
        elapsed = last[0] - first[0]
        if len(history) < 2 or elapsed < 0.75:
            return None
        return max(0.0, (last[1] - first[1]) / elapsed)

    def _annotate_fleet_rates(self, value: dict[str, Any]) -> None:
        """Attach measured per-host GPS and exact/estimated SPS.

        Remote controllers expose monotonic completed-job counters.  The
        trainer's scheduler independently counts completed local games, so
        Inzi GPS comes from its emitted ``local_gps`` rather than subtracting
        asynchronous remote samples from a fleet-wide rate.  Until every path
        emits side-specific decision counters, estimated SPS is explicitly
        marked from the fleet's observed decisions/game.
        """
        curriculum = value.get("curriculum") or {}
        queues = curriculum.get("scheduler_queues") or {}
        progress = curriculum.get("progress") or {}
        progress_current = progress.get("current")
        progress_total = progress.get("total")
        all_games_claimed = queues.get("unassigned") == 0
        claimed_results_pending = (
            max(0, int(progress_total) - int(progress_current))
            if isinstance(progress_current, (int, float))
            and isinstance(progress_total, (int, float))
            else None
        )
        remote_dispatch = curriculum.get("remote_dispatch") or {}
        fleet = value.get("fleet") or {}
        now = float(value.get("observed_at") or time.time())
        run_name = str(curriculum.get("run") or "unknown")
        stage = str(progress.get("stage") or curriculum.get("stage") or "")
        iteration = progress.get("iteration", curriculum.get("iteration"))
        phase_identity = f"{run_name}:{iteration}:{stage}"
        curriculum_active = bool(curriculum.get("active"))
        collecting = bool(
            curriculum_active
            and (
                stage.startswith("collect:")
                or stage.startswith("heldout")
                or stage == "promotion"
            )
        )
        training = bool(curriculum_active and stage.startswith("train"))
        remote_phase_active = bool(
            collecting
            and (
                stage == "collect:self_play"
                or
                int(progress.get("remotes") or 0) > 0
                or int(curriculum.get("remote_workers") or 0) > 0
                or int(curriculum.get("remote_request_sockets") or 0) > 0
            )
        )

        total_gps = None
        if collecting:
            # The run-bound tqdm rate is measured by the collector itself.
            # Dashboard samples arrive asynchronously and can see the same
            # counter twice, which previously turned a healthy live rate into
            # a false zero after remote subtraction.
            if isinstance(progress.get("gps"), (int, float)):
                total_gps = max(0.0, float(progress["gps"]))
            else:
                total_gps = self._counter_rate(
                    "fleet:games",
                    sampled_at=now,
                    counter=progress.get("current"),
                    identity=phase_identity,
                )
        total_sps = (
            max(0.0, float(progress["sps"]))
            if isinstance(progress.get("sps"), (int, float))
            else None
        )
        current_density = None
        if (
            collecting
            and total_gps is not None
            and total_gps > 0
            and total_sps is not None
            and total_sps > 0
        ):
            current_density = total_sps / total_gps
            # A malformed/tail zero should not erase the last useful estimate.
            if 1.0 <= current_density <= 10000.0:
                self.decision_density[run_name] = current_density
        density = self.decision_density.get(run_name)
        # Formal heldout/promotion games intentionally do not retain replay
        # trajectories, so their collector emits sps=0.  Reconcile that phase
        # with the latest committed collection's exact games/decision rates
        # rather than displaying a false zero or a density from another run.
        timing = curriculum.get("iteration_timing") or {}
        receipt_gps = timing.get("latest_gps")
        receipt_sps = timing.get("latest_sps")
        if (
            current_density is None
            and isinstance(receipt_gps, (int, float))
            and float(receipt_gps) > 0.0
            and isinstance(receipt_sps, (int, float))
            and float(receipt_sps) > 0.0
        ):
            receipt_density = float(receipt_sps) / float(receipt_gps)
            if 1.0 <= receipt_density <= 10000.0:
                density = receipt_density
                self.decision_density[run_name] = receipt_density

        remote_gps: dict[str, float | None] = {}
        remote_sps: dict[str, float | None] = {}
        # The Elmo worker can legitimately have every advertised socket in use
        # during a full dispatch. In that state a dashboard-only health
        # connection may not get a control slot, while the trainer still emits
        # measured per-wave local/remote rates. Use that run-bound telemetry as
        # a fallback rather than making the fleet card go blank.
        scheduler_local_gps = None
        scheduler_local_sps = None
        scheduler_remote_gps = None
        scheduler_remote_sps = None
        scheduler_wave_gps = None
        scheduler_wave_sps = None
        buffered_results = 0
        if collecting and bool(curriculum.get("source_current")):
            for event in reversed(value.get("recent_events") or []):
                event_text = str(event)
                remote_match = re.search(
                    r"\bremote_gps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                local_match = re.search(
                    r"\blocal_gps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                local_sps_match = re.search(
                    r"\blocal_sps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                remote_sps_match = re.search(
                    r"\bremote_sps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                wave_match = re.search(
                    r"\bwave_gps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                wave_sps_match = re.search(
                    r"\bwave_sps=([0-9]+(?:\.[0-9]+)?)", event_text
                )
                buffer_match = re.search(
                    r"'memory_items':\s*(\d+).*?'spool_files':\s*(\d+)",
                    event_text,
                )
                if buffer_match and buffered_results <= 0:
                    buffered_results = int(buffer_match.group(1)) + int(
                        buffer_match.group(2)
                    )
                if remote_match or local_match or wave_match:
                    if remote_match:
                        scheduler_remote_gps = float(remote_match.group(1))
                    if local_match:
                        scheduler_local_gps = float(local_match.group(1))
                    if local_sps_match:
                        scheduler_local_sps = float(local_sps_match.group(1))
                    if remote_sps_match:
                        scheduler_remote_sps = float(remote_sps_match.group(1))
                    if wave_match:
                        scheduler_wave_gps = float(wave_match.group(1))
                    if wave_sps_match:
                        scheduler_wave_sps = float(wave_sps_match.group(1))
                    break
        for host_key in ("elmo", "bert"):
            host = fleet.get(host_key) or {}
            worker = host.get("worker") or {}
            if (
                host_key == "bert"
                and host.get("production_active") is False
                and worker.get("testing")
            ):
                # Bert's Apple benchmark is deliberately outside the trainer
                # fleet. Preserve its own GPS/SPS/stage instead of replacing it
                # with production counters, and never subtract it from Inzi.
                test_identity = (
                    f"bert:test:{worker.get('command') or worker.get('optimization_variant') or 'idle'}"
                )
                live_gps = self._counter_rate(
                    "bert:test:games",
                    sampled_at=now,
                    counter=worker.get("jobs_completed"),
                    identity=test_identity,
                )
                live_sps = self._counter_rate(
                    "bert:test:decisions",
                    sampled_at=now,
                    counter=worker.get("decisions_completed"),
                    identity=test_identity,
                )
                if live_gps is not None:
                    worker["gps"] = live_gps
                if live_sps is not None:
                    worker["sps"] = live_sps
                worker["rate_stage"] = worker.get("optimization_stage")
                worker["rate_source"] = (
                    "Apple optimization live worker counters"
                    if live_gps is not None or live_sps is not None
                    else worker.get("rate_source")
                    or "Apple optimization latest completed topology"
                )
                worker["sps_estimated"] = False
                remote_gps[host_key] = None
                remote_sps[host_key] = None
                continue
            # Docker health checks can briefly match the worker command and
            # add a transient PID to controller_pids.  Basing identity on that
            # list resets the rate window every health probe.  The monotonic
            # counter rollback below already detects a real worker restart.
            identity = f"{host_key}:{worker.get('command') or 'worker'}"
            gps = (
                self._counter_rate(
                    f"{host_key}:games",
                    sampled_at=now,
                    counter=worker.get("jobs_completed"),
                    identity=f"{identity}:{phase_identity}",
                    window_s=3600.0,
                )
                if collecting
                else None
            )
            gps_from_scheduler = False
            # Aggregate remote scheduler telemetry is never assigned wholly to
            # Elmo: that made Elmo look fast and Bert look idle whenever one
            # health sample was delayed. Per-host cards use host counters only.
            exact_sps = (
                self._counter_rate(
                    f"{host_key}:decisions",
                    sampled_at=now,
                    counter=worker.get("decisions_completed"),
                    identity=f"{identity}:{phase_identity}",
                    window_s=3600.0,
                )
                if collecting
                else None
            )
            sps = exact_sps
            source = "trainer scheduler telemetry" if gps_from_scheduler else "worker-counters"
            estimated = False
            if sps is None and gps is not None and density is not None:
                sps = gps * density
                source = (
                    "trainer scheduler GPS + fleet decisions/game"
                    if gps_from_scheduler
                    else "worker-gps + fleet decisions/game"
                )
                estimated = True
            # Once a remote has drained its assigned/pulled work, its rolling
            # counter window still contains old completions and therefore
            # decays slowly toward zero.  Do not treat those decaying samples
            # as new measurements: retain the final genuinely-active GPS/SPS
            # until this host receives work in a later collection wave.
            active_jobs = worker.get("active_jobs")
            worker_slots = worker.get("workers")
            execution_slots = (
                max(1, int(worker_slots))
                if isinstance(worker_slots, (int, float))
                else None
            )
            request_target = remote_dispatch.get(
                f"{host_key}_request_sockets"
            )
            if not isinstance(request_target, (int, float)):
                request_target = None
            admitted = (
                max(0, int(active_jobs))
                if isinstance(active_jobs, (int, float))
                else None
            )
            fed_workers = (
                min(execution_slots, admitted)
                if execution_slots is not None and admitted is not None
                else None
            )
            queued_jobs = (
                max(0, admitted - execution_slots)
                if execution_slots is not None and admitted is not None
                else None
            )
            worker.update(
                execution_slots=execution_slots,
                admitted_requests=admitted,
                queued_jobs=queued_jobs,
                request_target=(int(request_target) if request_target is not None else None),
                feed_coverage=(
                    float(fed_workers) / float(execution_slots)
                    if execution_slots and fed_workers is not None
                    else None
                ),
            )
            full_sample_floor = (
                max(1, int(float(worker_slots) * 0.25))
                if isinstance(worker_slots, (int, float))
                else 1
            )
            progress_percent = float(progress.get("percent") or 0.0)
            allocation_draining = bool(
                collecting
                and progress_percent >= 90.0
                and isinstance(active_jobs, (int, float))
                and int(active_jobs) < full_sample_floor
            )
            allocation_idle = bool(
                isinstance(active_jobs, (int, float)) and int(active_jobs) <= 0
            )
            if allocation_draining:
                gps = None
                sps = None
                source = "allocation draining; full-concurrency rate frozen"
                estimated = False
            rate_live = gps is not None or sps is not None
            rate_age_s = 0.0 if rate_live else None
            if collecting:
                gps, sps, source, estimated, rate_live, rate_age_s = (
                    self._retain_phase_rate(
                        host_key,
                        identity=f"{identity}:{phase_identity}",
                        sampled_at=now,
                        gps=gps,
                        sps=sps,
                        source=source,
                        estimated=estimated,
                    )
                )
            if training:
                gps, sps, source, estimated, rate_live, rate_age_s = (
                    self._retain_phase_rate(
                        host_key,
                        identity=f"{identity}:{phase_identity}",
                        sampled_at=now,
                        gps=None,
                        sps=None,
                        source="optimizer phase",
                        estimated=False,
                    )
                )
                if gps is None and sps is None:
                    gps, sps, source, estimated = 0.0, 0.0, "optimizer phase", False
                    rate_live, rate_age_s = True, 0.0
            worker.update(
                gps=gps,
                sps=sps,
                rate_source=source,
                sps_estimated=estimated,
                rate_stage=stage,
                rate_live=rate_live,
                rate_age_s=rate_age_s,
            )
            if not bool(host.get("reachable", True)) or not bool(worker.get("active")):
                worker["allocation_state"] = "OFFLINE · no production allocation"
            elif training:
                worker["allocation_state"] = "ALLOCATION COMPLETE · optimizer phase"
            elif remote_phase_active:
                if all_games_claimed:
                    worker["allocation_state"] = (
                        "DRAINING · all games claimed"
                        + (
                            f" · {claimed_results_pending} fleet results pending"
                            if claimed_results_pending
                            else ""
                        )
                    )
                elif (
                    execution_slots is not None
                    and admitted is not None
                    and admitted >= execution_slots
                ):
                    worker["allocation_state"] = (
                        f"WORKING · {execution_slots}/{execution_slots} workers fed"
                        f" · {int(queued_jobs or 0)} queued"
                    )
                elif admitted is not None and admitted > 0:
                    worker["allocation_state"] = (
                        f"REFILLING · {int(fed_workers or 0)}/{execution_slots or '?'} "
                        "workers fed · queue empty"
                    )
                elif (
                    rate_live is True
                    and isinstance(gps, (int, float))
                    and float(gps) > 0.0
                ):
                    # ``active_jobs`` is an instantaneous sample while GPS is
                    # a monotonic completion-counter rate over the interval.
                    # Short remote games can finish between health polls; call
                    # that bursty feed, not starvation.
                    worker["allocation_state"] = (
                        f"BURSTY FEED · 0/{execution_slots or '?'} at sample · "
                        f"{float(gps):.2f} GPS completing"
                    )
                elif buffered_results > 0:
                    worker["allocation_state"] = (
                        f"RESULTS BUFFERED · {buffered_results} fleet games awaiting ingest"
                    )
                elif float(progress.get("percent") or 0.0) < 95.0:
                    worker["allocation_state"] = (
                        f"STARVED · 0/{execution_slots or '?'} workers fed · refill pending"
                    )
                elif rate_live is False and (
                    isinstance(gps, (int, float)) or isinstance(sps, (int, float))
                ):
                    worker["allocation_state"] = (
                        "ALLOCATION COMPLETE · waiting for remaining fleet"
                    )
                else:
                    worker["allocation_state"] = "READY · waiting for scheduler work"
            elif curriculum_active and (
                stage == "collect:public_mix"
                or stage.startswith("heldout")
                or stage == "promotion"
            ):
                worker["allocation_state"] = "ALLOCATION COMPLETE · local-only phase"
            else:
                worker["allocation_state"] = "READY · next self-play allocation"
            remote_gps[host_key] = gps
            remote_sps[host_key] = sps

        inzi = fleet.get("inzi") or {}
        inzi_worker = inzi.get("worker") or {}
        measured_remote_gps = [rate for rate in remote_gps.values() if rate is not None]
        measured_remote_sps = [rate for rate in remote_sps.values() if rate is not None]
        if density is None and measured_remote_gps and measured_remote_sps:
            remote_gps_sum = sum(measured_remote_gps)
            remote_sps_sum = sum(measured_remote_sps)
            if remote_gps_sum > 0.0 and remote_sps_sum > 0.0:
                remote_density = remote_sps_sum / remote_gps_sum
                if 1.0 <= remote_density <= 10000.0:
                    density = remote_density
                    self.decision_density[run_name] = remote_density
        local_only_phase = bool(
            (
                stage == "collect:public_mix"
                or stage.startswith("heldout")
                or stage == "promotion"
            )
            and not remote_phase_active
        )
        if collecting and local_only_phase and total_gps is not None:
            local_gps = max(0.0, float(total_gps))
            local_gps_source = "local-only collector completion counter"
        elif (
            collecting
            and scheduler_local_gps is not None
            and float(scheduler_local_gps) > 0.0
        ):
            local_gps = max(0.0, float(scheduler_local_gps))
            local_gps_source = "trainer scheduler local completion counter"
        elif collecting and total_gps is not None:
            local_gps = max(0.0, float(total_gps) - sum(measured_remote_gps))
            local_gps_source = "collector total minus remote counters"
        else:
            local_gps = None
            local_gps_source = "no active collection"
        local_sps = None
        local_sps_estimated = False
        if (
            collecting
            and scheduler_local_sps is not None
            and float(scheduler_local_sps) > 0.0
        ):
            local_sps = max(0.0, float(scheduler_local_sps))
        elif local_gps is not None and density is not None:
            local_sps = local_gps * density
            local_sps_estimated = True
        elif collecting and total_sps is not None:
            remote_sps_sum = sum(
                rate for rate in remote_sps.values() if rate is not None
            )
            local_sps = max(0.0, total_sps - remote_sps_sum)
            local_sps_estimated = True
        source = (
            f"{local_gps_source}; SPS from fleet decisions/game"
            if local_sps is not None and collecting and local_sps_estimated
            else f"{local_gps_source}; direct local decision counter"
            if local_sps is not None and collecting
            else local_gps_source
        )
        if training:
            local_gps, local_sps, source = 0.0, 0.0, "optimizer phase"
        inzi_worker.update(
            gps=local_gps,
            sps=local_sps,
            rate_source=source,
            sps_estimated=bool(local_sps_estimated),
            rate_stage=stage,
        )
        local_slots = max(1, int(inzi_worker.get("workers") or 96))
        leaf_servers = max(0, int(inzi_worker.get("leaf_servers") or 0))
        if collecting:
            current_games = progress.get("current")
            total_games = progress.get("total")
            remaining_games = (
                max(0, int(total_games) - int(current_games))
                if isinstance(current_games, (int, float))
                and isinstance(total_games, (int, float))
                else local_slots
            )
            active_games = min(local_slots, remaining_games)
            inzi_worker["active_games"] = active_games
            inzi_worker["allocation_state"] = (
                f"WORKING · {active_games} local game"
                f"{'s' if active_games != 1 else ''} active"
            )
            inzi_worker["allocation"] = (
                f"{local_slots} simulator slots · {leaf_servers} Blackwell policy leaves"
            )
        elif stage == "train:preparing":
            inzi_worker["active_games"] = 0
            inzi_worker["allocation_state"] = "REPLAY PREP · 0 simulation games"
            inzi_worker["allocation"] = (
                "CPU replay-window assembly · Blackwell waiting"
            )
        elif training:
            inzi_worker["active_games"] = 0
            inzi_worker["allocation_state"] = "OPTIMIZER · 0 simulation games"
            inzi_worker["allocation"] = "Blackwell full-model optimizer"
        else:
            inzi_worker["active_games"] = 0
            inzi_worker["allocation_state"] = "READY · next local allocation"
            inzi_worker["allocation"] = (
                f"{local_slots} simulator slots · {leaf_servers} Blackwell policy leaves"
            )
        fleet_total_gps = (
            scheduler_wave_gps
            if scheduler_wave_gps is not None and scheduler_wave_gps > 0
            else total_gps
        )
        estimated_fleet_sps = (
            float(fleet_total_gps) * float(density)
            if collecting
            and (stage.startswith("heldout") or stage == "promotion")
            and isinstance(fleet_total_gps, (int, float))
            and float(fleet_total_gps) > 0.0
            and (not isinstance(total_sps, (int, float)) or float(total_sps) <= 0.0)
            and isinstance(density, (int, float))
            and float(density) > 0.0
            else None
        )
        raw_fleet_total_sps = (
            scheduler_wave_sps
            if scheduler_wave_sps is not None and scheduler_wave_sps > 0
            else estimated_fleet_sps
            if estimated_fleet_sps is not None
            else total_sps
        )
        raw_fleet_rate_source = (
            "trainer scheduler generation telemetry"
            if scheduler_wave_sps is not None and scheduler_wave_sps > 0
            else "estimated from live GPS × latest committed decision density"
            if estimated_fleet_sps is not None
            else "collector ingest telemetry"
        )
        raw_fleet_sps_estimated = bool(
            estimated_fleet_sps is not None
            and not (scheduler_wave_sps is not None and scheduler_wave_sps > 0)
        )
        if collecting:
            (
                display_sps,
                display_sps_held,
                display_sps_age_s,
                display_sps_source,
            ) = self._retain_collection_sps(
                identity=phase_identity,
                sampled_at=now,
                sps=total_sps,
                source="collector cumulative sequence telemetry",
            )
            # Before the collector emits its first positive SPS frame, only a
            # scheduler rate measured in this exact phase may replace zero.
            # A decisions/game density retained from an earlier iteration is
            # intentionally limited to non-replay heldout/promotion phases.
            if (
                (not isinstance(display_sps, (int, float)) or display_sps <= 0)
                and isinstance(raw_fleet_total_sps, (int, float))
                and raw_fleet_total_sps > 0
            ):
                display_sps = raw_fleet_total_sps
                display_sps_source = raw_fleet_rate_source
                display_sps_held = False
                display_sps_age_s = 0.0
        else:
            display_sps = raw_fleet_total_sps
            display_sps_held = False
            display_sps_age_s = None
            display_sps_source = raw_fleet_rate_source
        value["fleet_rates"] = {
            "stage": stage,
            "iteration": iteration,
            "total_gps": fleet_total_gps,
            "total_sps": raw_fleet_total_sps,
            "display_sps": display_sps,
            "display_sps_held": display_sps_held,
            "display_sps_age_s": display_sps_age_s,
            "display_sps_source": display_sps_source,
            "display_sps_estimated": bool(
                raw_fleet_sps_estimated
                and display_sps == raw_fleet_total_sps
                and not display_sps_held
            ),
            "ingest_gps": total_gps,
            "ingest_sps": total_sps,
            "local_gps": scheduler_local_gps,
            "local_sps": scheduler_local_sps,
            "remote_gps": scheduler_remote_gps,
            "remote_sps": scheduler_remote_sps,
            "buffered_results": buffered_results,
            "decisions_per_game": density,
            "window_s": 15.0,
            "rate_source": raw_fleet_rate_source,
        }

    def _annotate_scheduler_queues(self, value: dict[str, Any]) -> None:
        """Join controller queue targets to live per-host queue flow.

        A remote's monotonic ``jobs_completed + active_jobs`` is the number of
        jobs admitted by that server. Its derivative is measured dispatch GPS;
        ``jobs_completed`` alone is measured drain GPS. This makes outbound
        starvation distinguishable from a healthy queue that is merely being
        consumed quickly.
        """
        curriculum = value.get("curriculum") or {}
        queues = curriculum.get("scheduler_queues") or {}
        if not isinstance(queues, dict):
            queues = {}
        fleet = value.get("fleet") or {}
        rates = value.get("fleet_rates") or {}
        observed_at = float(value.get("observed_at") or time.time())
        progress = curriculum.get("progress") or {}
        stage = str(progress.get("stage") or curriculum.get("stage") or "")
        generation_stage = bool(
            curriculum.get("active")
            and (stage.startswith("collect:") or stage in {"heldout", "promotion"})
        )
        identity = (
            f"{curriculum.get('run')}:{progress.get('iteration')}:"
            f"{stage}"
        )
        inzi_worker = ((fleet.get("inzi") or {}).get("worker") or {})
        local_active = inzi_worker.get("active_games")
        local_high_water = inzi_worker.get("workers")
        queues["local"] = {
            "active_or_claimed": local_active,
            "high_water": local_high_water,
            "dispatch_gps": inzi_worker.get("gps"),
            "drain_gps": inzi_worker.get("gps"),
            "source": "trainer local scheduler/generation counter",
        }

        endpoint_rows = queues.get("endpoints")
        if not isinstance(endpoint_rows, dict):
            endpoint_rows = {}
            queues["endpoints"] = endpoint_rows
        for host_key in ("elmo", "bert"):
            host = fleet.get(host_key) or {}
            worker = host.get("worker") or {}
            # Keep isolated hardware experiments visible in fleet telemetry,
            # but never present them as scheduler-eligible endpoints. Bert's
            # port-8776 MPS benchmark is deliberately outside production.
            if host.get("production_active") is False:
                endpoint_rows.pop(host_key, None)
                continue
            row = endpoint_rows.get(host_key)
            if not isinstance(row, dict):
                row = {}
                endpoint_rows[host_key] = row
            active_jobs = worker.get("active_jobs")
            completed = worker.get("jobs_completed")
            admitted_counter = (
                float(active_jobs) + float(completed)
                if isinstance(active_jobs, (int, float))
                and isinstance(completed, (int, float))
                else None
            )
            dispatch_gps = self._counter_rate(
                f"{host_key}:scheduler-admitted",
                sampled_at=observed_at,
                counter=admitted_counter,
                identity=identity,
                window_s=15.0,
            )
            execution = worker.get("execution_slots", worker.get("workers"))
            executing = (
                min(max(0, int(execution)), max(0, int(active_jobs)))
                if isinstance(execution, (int, float))
                and isinstance(active_jobs, (int, float))
                else None
            )
            server_queued = worker.get("queued_jobs")
            high_water = row.get("protected_high_water", row.get("protected_cap"))
            sockets = row.get("socket_capacity")
            controller_reserve = (
                max(0, int(high_water) - int(sockets))
                if isinstance(high_water, (int, float))
                and isinstance(sockets, (int, float))
                else row.get("controller_reserve_target")
            )
            row.update(
                executing=executing,
                server_queued=server_queued,
                server_admitted=active_jobs,
                controller_reserve_target=controller_reserve,
                dispatch_gps=dispatch_gps,
                drain_gps=worker.get("gps"),
                queue_delta_gps=(
                    float(dispatch_gps) - float(worker.get("gps"))
                    if isinstance(dispatch_gps, (int, float))
                    and isinstance(worker.get("gps"), (int, float))
                    else None
                ),
                flow_source="remote admitted/completed monotonic counters",
            )
        live_remote_demand = progress.get("remotes")
        if isinstance(live_remote_demand, (int, float)):
            queues["live_remote_worker_demand"] = max(
                0, int(live_remote_demand)
            )
        # ``progress.current/total`` changes grain with the phase. During
        # collection it counts games, but during learner prep/training it
        # counts optimizer batches. Never turn the remaining batch count into
        # an apparent rollout backlog after collection has completed.
        if not generation_stage:
            queues["unassigned"] = 0 if curriculum.get("active") and stage else None
            queues["unassigned_estimated"] = False
            queues["unassigned_source"] = (
                f"scheduler idle outside game-generation phase ({stage})"
                if stage
                else "scheduler phase unavailable"
            )
        elif not isinstance(queues.get("unassigned"), (int, float)):
            total = progress.get("total")
            completed = progress.get("current")
            reserved = sum(
                max(0, int(row.get("dispatch_reserved") or 0))
                for row in endpoint_rows.values()
                if isinstance(row, dict)
            )
            if isinstance(total, (int, float)):
                queues["unassigned"] = max(
                    0,
                    int(total)
                    - reserved
                    - max(0, int(local_active or 0))
                    - max(0, int(completed or 0)),
                )
                queues["unassigned_estimated"] = True
                queues["unassigned_source"] = (
                    "pre-heartbeat upper estimate: total - protected reservations "
                    "- local claim - ingested results"
                )
        else:
            queues["unassigned_estimated"] = False
            queues["unassigned_source"] = "trainer remaining counter"
        queues["results"] = {
            "waiting_ingest": rates.get("buffered_results"),
            "generation_gps": rates.get("total_gps"),
            "ingest_gps": rates.get("ingest_gps"),
            "source": "bounded RAM + disk result buffer",
        }
        queues["available"] = bool(
            queues.get("available") or curriculum.get("active")
        )
        queues["updated_at"] = observed_at
        curriculum["scheduler_queues"] = queues
        value["scheduler_queues"] = queues

    @staticmethod
    def _annotate_source_integrity(value: dict[str, Any]) -> None:
        """Publish one run-bound truth registry for every visible panel."""

        now = float(value.get("dashboard_sampled_at") or time.time())
        curriculum = value.get("curriculum") or {}
        service = value.get("service") or {}
        handoff = value.get("specialist_handoff") or {}
        protocol = value.get("specialist_protocol") or {}
        training = value.get("training") or {}
        model = value.get("model") or {}
        structure = model.get("checkpoint_structure") or {}
        expanded_heads = structure.get("expanded_head_training") or {}
        expanded_heads_required = bool(
            expanded_heads.get("available") is True
            or expanded_heads.get("actual_tensor_heads")
        )
        expert = value.get("expert_refresh") or {}
        queues = value.get("scheduler_queues") or {}
        fleet = value.get("fleet") or {}
        stage = str(curriculum.get("stage") or "")
        collecting = bool(
            curriculum.get("active")
            and (
                stage.startswith("collect:")
                or stage.startswith("heldout")
                or stage == "promotion"
            )
        )
        runtime_specialist = str(
            protocol.get("runtime_active_specialist") or ""
        )
        canonical_specialist = str(
            protocol.get("canonical_active_specialist") or ""
        )
        effective_protocol_specialist = (
            runtime_specialist or canonical_specialist
        )
        handoff_target_specialist = str(
            handoff.get("next_specialist_id") or ""
        )
        service_active = bool(
            service.get("active") and int(service.get("pid") or 0) > 0
        )
        post_fleet_refresh = protocol.get("post_fleet_refresh") or {}
        terminal_transition = protocol.get("terminal_specialist_transition") or {}
        final_refresh_current = bool(
            service_active
            and training.get("mode")
            in {
                "final_format_alakazam_ordinary_refresh",
                "final_format_alakazam_h10_rl",
            }
            and runtime_specialist == "alakazam"
            and int(post_fleet_refresh.get("goal_revision") or 0) >= 79
            and post_fleet_refresh.get("status")
            == "alakazam_ordinary_fallback_bootstrap_active"
            and terminal_transition.get("status") == "activated"
            and terminal_transition.get("terminal_disposition")
            == "failed_experiment"
        )
        handoff_active = bool(
            handoff.get("active")
            and int(handoff.get("pid") or 0) > 0
            and str(handoff.get("source") or "")
        )
        handoff_transition_current = bool(
            handoff.get("transition_current") is True
            and not handoff_active
            and not service_active
            and str(handoff.get("source") or "")
            and str(handoff.get("source_specialist_id") or "")
            and str(handoff.get("next_specialist_id") or "")
            and str(handoff.get("phase") or "")
            in {
                "next_specialist_selected",
                "next_specialist_bootstrap_frozen",
                "next_specialist_rl_armed",
                "next_specialist_rl_started",
            }
        )
        handoff_progress_current = bool(
            handoff_active
            and isinstance(handoff.get("updated_at"), (int, float))
            and now - float(handoff["updated_at"]) <= 35.0
            and (
                isinstance(handoff.get("epoch"), int)
                or str(handoff.get("phase") or handoff.get("stage") or "")
            )
        )
        expert_days = [
            row
            for row in (expert.get("days") or [])
            if isinstance(row, dict) and str(row.get("day") or row.get("date") or "")
        ]
        expert_archive_current = bool(
            expert.get("available") is True
            and expert.get("archive_window_ready") is True
            and (
                (
                    int(expert.get("total_days") or 0) == 20
                    and len(expert_days) == 20
                )
                or (
                    expert.get("complete") is True
                    and expert.get("assembled_manifest_ready") is True
                    and expert.get("filtered_corpus_ready") is True
                )
            )
        )
        protocol_specialists = protocol.get("specialists")
        protocol_roster_current = True
        protocol_frozen_pool_current = True
        protocol_model_roster_current = True
        specialist_ids: list[str] = []
        active_record_ids: set[str] = set()
        if isinstance(protocol_specialists, list):
            specialist_ids = [
                str(row.get("id") or "")
                for row in protocol_specialists
                if isinstance(row, dict)
            ]
            required_target_count = protocol.get("required_target_count")
            active_record_ids = {
                str(row.get("id") or "")
                for row in protocol_specialists
                if isinstance(row, dict) and row.get("active") is True
            }
            if handoff_active or handoff_transition_current:
                handoff_source_specialist = str(
                    handoff.get("source_specialist_id") or ""
                )
                allowed_transition_active_records = {frozenset()}
                if handoff_source_specialist:
                    allowed_transition_active_records.add(
                        frozenset({handoff_source_specialist})
                    )
                if handoff_target_specialist:
                    allowed_transition_active_records.add(
                        frozenset({handoff_target_specialist})
                    )
                active_records_current = (
                    frozenset(active_record_ids)
                    in allowed_transition_active_records
                )
            elif final_refresh_current:
                # The post-fleet refresh is a separately versioned derivative,
                # not a reopened specialist slot. No fleet row should become
                # active merely because Alakazam is being refreshed.
                active_records_current = not active_record_ids
            else:
                active_records_current = active_record_ids == {
                    effective_protocol_specialist
                }
            protocol_roster_current = bool(
                specialist_ids
                and all(specialist_ids)
                and len(set(specialist_ids)) == len(specialist_ids)
                and (
                    not isinstance(required_target_count, int)
                    or len(specialist_ids) == required_target_count
                )
                and active_records_current
            )
            frozen_state_ids = {
                str(row.get("id") or "")
                for row in protocol_specialists
                if isinstance(row, dict)
                and (
                    row.get("frozen") is True
                    or row.get("public_mix_eligible") is True
                )
            }
            frozen_runtime_ids = {
                str(row.get("specialist_id") or "")
                for row in (protocol.get("frozen_inference_opponents") or [])
                if isinstance(row, dict)
            }
            protocol_frozen_pool_current = frozen_state_ids == frozen_runtime_ids
            model_adapter_count = structure.get("adapter_expert_count")
            model_adapter_ids = structure.get("adapter_expert_ids")
            if isinstance(model_adapter_count, int):
                # The physical matchup bank is intentionally independent from
                # the required specialist-training plan. Owner-removed decks
                # keep immutable adapter slots, and newly planned decks may
                # remain route-dormant until a safe boundary. Validate the
                # checkpoint's physical roster against its own declared IDs,
                # never against required_target_count.
                registry_verified = structure.get(
                    "adapter_registry_verified"
                )
                protocol_model_roster_current = bool(
                    model_adapter_count > 0
                    and registry_verified is not False
                )
                if registry_verified is not True and isinstance(
                    model_adapter_ids, list
                ):
                    normalized_adapter_ids = [
                        str(value) for value in model_adapter_ids if str(value)
                    ]
                    protocol_model_roster_current = bool(
                        len(normalized_adapter_ids) == model_adapter_count
                        and len(set(normalized_adapter_ids))
                        == model_adapter_count
                    )
        # The protocol card describes the canonical specialist state, which
        # remains current while production is stopped.  A live process adds a
        # stronger reconciliation requirement: its selected specialist must
        # match the canonical pointer.  Do not make a stopped controller erase
        # an otherwise valid, parseable canonical protocol.
        completed_ids = set(
            (protocol.get("program_progress") or {}).get(
                "completed_specialist_ids", ()
            )
        )
        handoff_source_specialist = str(
            handoff.get("source_specialist_id") or ""
        )
        source_is_transition_owner = bool(
            handoff_source_specialist
            and (
                handoff_source_specialist in completed_ids
                or handoff_source_specialist == canonical_specialist
                or handoff_source_specialist
                == str(protocol.get("active_specialist") or "")
                or active_record_ids == {handoff_source_specialist}
            )
        )
        target_is_known_unfinished = bool(
            handoff_target_specialist
            and handoff_target_specialist in specialist_ids
            and handoff_target_specialist not in completed_ids
        )
        settled_handoff_protocol_current = bool(
            handoff_transition_current
            and handoff_source_specialist in completed_ids
            and target_is_known_unfinished
            and not active_record_ids
        )
        handoff_protocol_current = bool(
            handoff_active
            and handoff_progress_current
            and source_is_transition_owner
            and (
                (
                    protocol.get("phase") == "shared_core_derivation"
                    and not canonical_specialist
                    and not handoff_target_specialist
                )
                or (
                    target_is_known_unfinished
                    and (
                        handoff_target_specialist
                        == str(protocol.get("active_specialist") or "")
                        or canonical_specialist == handoff_source_specialist
                        or active_record_ids == {handoff_source_specialist}
                    )
                )
            )
        )
        protocol_identity_current = bool(
            protocol.get("available") is True
            and (
                protocol.get("canonical_pointer_stale") is not True
                or protocol.get("runtime_identity_reconciled") is True
                or handoff_protocol_current
                or settled_handoff_protocol_current
                or final_refresh_current
            )
            and protocol_roster_current
            and protocol_frozen_pool_current
            and protocol_model_roster_current
            and (
                handoff_protocol_current
                or settled_handoff_protocol_current
                or final_refresh_current
                or (
                    canonical_specialist
                    and not service_active
                )
                or (
                    runtime_specialist
                    and runtime_specialist
                    == str(protocol.get("active_specialist") or "")
                    and (
                        runtime_specialist == canonical_specialist
                        or protocol.get("runtime_identity_reconciled") is True
                    )
                )
            )
        )
        rows: dict[str, dict[str, Any]] = {
            "stage": {
                "required": True,
                "current": bool(
                    handoff_progress_current
                    or handoff_transition_current
                    or (
                        service_active
                        and int(service.get("restart_count") or 0) >= 0
                    )
                ),
                "identity": (
                    handoff.get("service", {}).get("name")
                    if handoff_active
                    else service.get("name")
                ),
                "source": (
                    handoff.get("source")
                    if handoff_active or handoff_transition_current
                    else "systemd user cgroup"
                ),
            },
            "progress": {
                "required": True,
                "current": bool(
                    handoff_progress_current
                    or handoff_transition_current
                    or (
                        curriculum.get("active")
                        and curriculum.get("source_current") is True
                        and str(curriculum.get("run") or "")
                    )
                ),
                "identity": (
                    f"handoff:{handoff.get('source_specialist_id')}:"
                    f"{handoff.get('epoch')}:{handoff.get('stage')}"
                    if handoff_active
                    else (
                        f"handoff:{handoff.get('source_specialist_id')}:"
                        f"{handoff.get('next_specialist_id')}:"
                        f"{handoff.get('phase')}"
                        if handoff_transition_current
                        else (
                            f"{curriculum.get('run') or 'unknown'}:"
                            f"{curriculum.get('iteration')}:{stage}"
                        )
                    )
                ),
                "source": (
                    handoff.get("source")
                    if handoff_active or handoff_transition_current
                    else curriculum.get("progress_status_source")
                    or curriculum.get("progress_source")
                ),
            },
            "model": {
                "required": True,
                "current": bool(
                    structure.get("verified") is True
                    and structure.get("checkpoint")
                    == model.get("active_checkpoint")
                    and structure.get("checkpoint_digest")
                    == model.get("active_checkpoint_digest")
                ),
                "identity": model.get("active_checkpoint_digest"),
                "source": structure.get("checkpoint"),
                "checks": {
                    "checkpoint_identity": bool(
                        structure.get("checkpoint")
                        == model.get("active_checkpoint")
                        and structure.get("checkpoint_digest")
                        == model.get("active_checkpoint_digest")
                    ),
                    "expanded_heads": bool(
                        not expanded_heads_required
                        or expanded_heads.get("verified") is True
                    ),
                    "legacy_v5_allowed": bool(
                        not expanded_heads_required
                        or expanded_heads.get("legacy_v5") is True
                        or expanded_heads.get("verified") is True
                    ),
                },
            },
            "expanded_heads": {
                "required": expanded_heads_required,
                "current": bool(
                    not expanded_heads_required
                    or (
                        expanded_heads.get("verified") is True
                        and structure.get("checkpoint_digest")
                        == model.get("active_checkpoint_digest")
                    )
                ),
                "identity": (
                    expanded_heads.get("contract_digest")
                    or model.get("active_checkpoint_digest")
                ),
                "source": structure.get("checkpoint"),
            },
            "protocol": {
                "required": True,
                "current": protocol_identity_current,
                "identity": (
                    runtime_specialist
                    or canonical_specialist
                    or handoff_target_specialist
                ),
                "source": protocol.get("source"),
                "checks": {
                    "canonical_pointer": bool(
                        (
                            effective_protocol_specialist
                            or handoff_protocol_current
                        )
                        and (
                            protocol.get("canonical_pointer_stale") is not True
                            or protocol.get("runtime_identity_reconciled") is True
                            or handoff_protocol_current
                            or settled_handoff_protocol_current
                            or final_refresh_current
                        )
                    ),
                    "specialist_roster": protocol_roster_current,
                    "frozen_pool": protocol_frozen_pool_current,
                    "model_roster": protocol_model_roster_current,
                    "live_runtime_identity": bool(
                        settled_handoff_protocol_current
                        or final_refresh_current
                        or not service_active
                        or (
                            runtime_specialist
                            and runtime_specialist
                            == str(protocol.get("active_specialist") or "")
                            and (
                                runtime_specialist == canonical_specialist
                                or protocol.get("runtime_identity_reconciled")
                                is True
                            )
                        )
                    ),
                },
            },
            "latest10": {
                "required": not (
                    handoff_active or handoff_transition_current
                ),
                "current": bool(
                    expert_archive_current
                    and expert.get("authoritative_for_active_run") is True
                ),
                "identity": (
                    f"{expert.get('window_start')}..{expert.get('window_end')}"
                ),
                "source": expert.get("source"),
                "checks": {
                    "archive_window": expert_archive_current,
                    "active_run_identity": (
                        expert.get("authoritative_for_active_run") is True
                    ),
                    "filtered_corpus_ready": (
                        expert.get("filtered_corpus_ready") is True
                    ),
                },
            },
            "scheduler": {
                "required": collecting,
                "current": bool(
                    not collecting
                    or (
                        queues.get("available") is True
                        and isinstance(queues.get("updated_at"), (int, float))
                        and now - float(queues["updated_at"]) <= 15.0
                    )
                ),
                "identity": stage,
                "source": queues.get("source"),
            },
        }
        # Cards that intentionally project the same authoritative object still
        # receive their own contract row. This keeps settings/reordering from
        # hiding whether a particular visible card is current.
        progress_current = rows["progress"]["current"]
        progress_source = rows["progress"]["source"]
        model_current = rows["model"]["current"]
        model_source = rows["model"]["source"]
        protocol_current = rows["protocol"]["current"]
        expert_current = rows["latest10"]["current"]
        bootstrap = value.get("bootstrap") or {}
        transition = value.get("transition") or {}
        baseline = value.get("baseline_eval") or {}
        gate = (
            (curriculum.get("gate_program") or {}).get("next_gate")
            or {}
        )
        replay = curriculum.get("replay_window") or {}
        rows.update(
            {
                "bootstrap": {
                    "required": True,
                    "current": bool(
                        (
                            handoff_progress_current
                            and training.get("mode") == "specialist_handoff"
                            and training.get("source") == handoff.get("source")
                        )
                        or handoff_transition_current
                        or (
                            progress_current
                            and bootstrap.get("compatibility_alias") is True
                            and bootstrap.get("alias_of") == "training"
                            and bootstrap.get("phase") == training.get("phase")
                        )
                    ),
                    "identity": (
                        f"{curriculum.get('iteration')}:{stage}"
                    ),
                    "source": progress_source,
                },
                "throughput": {
                    "required": True,
                    "current": bool(
                        progress_current or handoff_transition_current
                    ),
                    "identity": f"{curriculum.get('iteration')}:{stage}",
                    "source": progress_source,
                },
                "blackwell": {
                    "required": True,
                    "current": bool(value.get("gpus")),
                    "identity": "inzi:gpu1",
                    "source": "live nvidia-smi device telemetry",
                },
                "outcomes": {
                    "required": True,
                    "current": bool(
                        curriculum.get("last_committed_iteration") is not None
                        or curriculum.get("last_completed_iteration") is not None
                    ),
                    "identity": curriculum.get("last_committed_iteration")
                    if curriculum.get("last_committed_iteration") is not None
                    else curriculum.get("last_completed_iteration"),
                    "source": curriculum.get("commit_source")
                    or curriculum.get("latest_commit_source")
                    or curriculum.get("heldout_source"),
                },
                "adapterfleet": {
                    "required": False,
                    "current": bool(
                        model_current
                        and (
                            not (value.get("matchup_pipeline") or {}).get(
                                "adapter_fit", {}
                            ).get("active")
                        )
                    ),
                    "identity": model.get("active_checkpoint_digest"),
                    "source": model_source,
                },
                "replay": {
                    "required": bool(
                        stage.startswith("train:preparing")
                        or replay.get("active")
                    ),
                    "current": bool(
                        not (
                            stage.startswith("train:preparing")
                            or replay.get("active")
                        )
                        or replay.get("available") is True
                    ),
                    "identity": curriculum.get("iteration"),
                    "source": replay.get("source"),
                },
                "baseline": {
                    "required": False,
                    "current": bool(
                        not baseline
                        or baseline.get("historical") is True
                    ),
                    "identity": baseline.get("checkpoint"),
                    "source": baseline.get("source"),
                },
                "nextgate": {
                    "required": True,
                    "current": bool(
                        gate.get("available") is True
                        or gate.get("contract_valid") is True
                    ),
                    "identity": gate.get("checkpoint_digest")
                    or model.get("active_checkpoint_digest"),
                    "source": gate.get("contract_source")
                    or (curriculum.get("gate_program") or {}).get("source"),
                },
                "curriculum": {
                    "required": not handoff_active,
                    "current": bool(
                        handoff_active
                        or handoff_transition_current
                        or progress_current
                    ),
                    "identity": f"{curriculum.get('run')}:{stage}",
                    "source": progress_source,
                },
                "pure": {
                    "required": not handoff_active,
                    "current": bool(
                        handoff_active
                        or handoff_transition_current
                        or progress_current
                    ),
                    "identity": curriculum.get("run"),
                    "source": progress_source,
                },
                "command": {
                    "required": False,
                    "current": bool(service.get("command")),
                    "identity": service.get("pid"),
                    "source": "systemd user cgroup",
                },
                "raw": {
                    "required": False,
                    "current": True,
                    "identity": value.get("observed_at"),
                    "source": "complete /api/status snapshot",
                },
                "handoff": {
                    "required": bool(
                        handoff.get("active")
                        or handoff_transition_current
                    ),
                    "current": bool(
                        handoff_transition_current
                        or not handoff.get("active")
                        or (
                            handoff_progress_current
                            and source_is_transition_owner
                        )
                    ),
                    "identity": handoff.get("source_specialist_id"),
                    "source": handoff.get("source"),
                },
                "transition": {
                    "required": bool(transition.get("active")),
                    "current": bool(
                        transition.get("active")
                        or transition.get("historical") is True
                    ),
                    "identity": transition.get("phase"),
                    "source": transition.get("source"),
                },
            }
        )
        for host_key in ("inzi", "elmo", "bert"):
            host = fleet.get(host_key) or {}
            worker = host.get("worker") or {}
            allocation_complete = str(
                worker.get("allocation_state") or ""
            ).startswith("ALLOCATION COMPLETE")
            required = bool(
                host_key == "inzi"
                or (
                    curriculum.get("active")
                    and host.get("production_active") is not False
                )
            )
            rows[f"fleet_{host_key}"] = {
                "required": required,
                "current": bool(
                    host.get("reachable") is not False
                    and (
                        host_key == "inzi"
                        or not required
                        or (
                            worker.get("active") is True
                            and (
                                worker.get("health_current") is not False
                                or allocation_complete
                            )
                        )
                    )
                ),
                "identity": worker.get("command") or host.get("name"),
                "source": worker.get("rate_source")
                or host.get("telemetry_source")
                or "live host snapshot",
            }
        rows["hardware"] = {
            "required": True,
            "current": bool(
                rows["fleet_inzi"]["current"]
                and rows["fleet_elmo"]["current"]
                and rows["fleet_bert"]["current"]
            ),
            "identity": "inzi+elmo+bert",
            "source": "per-host live snapshots",
        }
        rows["fleet"] = dict(rows["hardware"])
        required_rows = [
            row for row in rows.values() if row.get("required") is True
        ]
        failed = [
            key
            for key, row in rows.items()
            if row.get("required") is True and row.get("current") is not True
        ]
        value["source_integrity"] = {
            "schema": "poke_bot.dashboard_source_integrity/v1",
            "current": not failed,
            "required_current": len(required_rows) - len(failed),
            "required_total": len(required_rows),
            "failed": failed,
            "rows": rows,
            "observed_at": now,
            "definition": (
                "Every live panel resolves from a run-bound authoritative source; "
                "historical fallbacks cannot satisfy a current-source check."
            ),
        }

    def _annotate_replay_progress(self, value: dict[str, Any]) -> None:
        """Promote replay-window loading into the canonical Curriculum bar."""
        curriculum = value.get("curriculum") or {}
        progress = curriculum.get("progress") or {}
        replay = curriculum.get("replay_window") or {}
        if not replay.get("available"):
            return
        now = float(value.get("observed_at") or time.time())
        run_name = str(curriculum.get("run") or "unknown")
        iteration = replay.get("iteration", curriculum.get("iteration"))
        current = replay.get("current")
        total = replay.get("total")
        rate_bps = (
            self._counter_rate(
                "replay-window:bytes",
                sampled_at=now,
                counter=current,
                identity=f"{run_name}:{iteration}:{replay.get('stage')}",
                window_s=12.0,
            )
            if str(replay.get("unit") or "") == "bytes"
            else None
        )
        if isinstance(rate_bps, (int, float)) and rate_bps > 0:
            replay["rate_bytes_per_sec"] = rate_bps
            if isinstance(current, (int, float)) and isinstance(total, (int, float)):
                replay["eta_s"] = max(0.0, (float(total) - float(current)) / rate_bps)
        if (
            str(progress.get("stage") or "") == "train:preparing"
            and str(replay.get("stage") or "") == "LOADING WINDOW"
            and isinstance(replay.get("percent"), (int, float))
            and isinstance(current, (int, float))
            and isinstance(total, (int, float))
        ):
            pct = max(0.0, min(100.0, float(replay["percent"])))
            filled = max(0, min(28, int(round(28.0 * pct / 100.0))))
            bar = "█" * filled + "░" * (28 - filled)
            current_gib = float(current) / (1024.0**3)
            total_gib = float(total) / (1024.0**3)
            rate_mib = (
                float(rate_bps) / (1024.0**2)
                if isinstance(rate_bps, (int, float)) and rate_bps > 0
                else None
            )
            eta_s = replay.get("eta_s")
            eta_text = (
                f"{int(float(eta_s)) // 60:02d}:{int(float(eta_s)) % 60:02d}"
                if isinstance(eta_s, (int, float))
                else "measuring"
            )
            timing = (
                f"{rate_mib:.1f} MiB/s, ETA {eta_text}"
                if rate_mib is not None
                else "measuring throughput"
            )
            progress.update(
                line=(
                    f"pure_rl replay-window iter={iteration}: {pct:5.1f}%|{bar}| "
                    f"{current_gib:.2f}/{total_gib:.2f} GiB [{timing}]"
                ),
                percent=pct,
                current=round(current_gib, 2),
                total=round(total_gib, 2),
                unit="GiB",
                rate=rate_mib,
                rate_unit="MiB/s",
                eta=eta_text,
                gps=None,
                sps=None,
                remotes=0,
            )

    def update(self) -> None:
        command = [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=4",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            "ControlPath=/tmp/pokebot-dashboard-ssh",
            "inzi@192.168.1.151",
            REMOTE_PYTHON,
            REMOTE_SNAPSHOT,
        ]
        try:
            proc = subprocess.run(
                command,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=15,
            )
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.strip() or f"ssh exited {proc.returncode}")
            value = json.loads(proc.stdout)
            local = subprocess.run(
                [
                    "/usr/bin/python3",
                    str(LOCAL_SNAPSHOT),
                    "--role",
                    "simulator",
                    "--name",
                    "Bert",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
            )
            try:
                bert = json.loads(local.stdout)
            except json.JSONDecodeError:
                bert = {
                    "reachable": False,
                    "name": "Bert",
                    "role": "simulator",
                    "error": local.stderr.strip() or "local telemetry unavailable",
                }
            trainer_command = str(
                (((value.get("curriculum") or {}).get("worker") or {}).get("command"))
                or ""
            )
            configured_endpoints = [
                str(endpoint)
                for endpoint in ((value.get("curriculum") or {}).get("remote_endpoints") or [])
            ]
            bert_worker = bert.get("worker") or {}
            bert_ready = bool(bert_worker.get("active") and bert_worker.get("listening"))
            bert_in_production = bert_ready and (
                "bert.local:8766" in trainer_command
                or any("bert.local:8766" in endpoint for endpoint in configured_endpoints)
            )
            if bert_in_production:
                bert["role"] = "production simulator"
                bert["production_active"] = True
                bert["assignment"] = (
                    "PRODUCTION · M4 CPU simulator"
                    if bool((value.get("curriculum") or {}).get("active"))
                    else "PRODUCTION READY · M4 CPU simulator"
                )
                bert_worker["testing"] = False
                bert_worker["production_active"] = True
                bert_worker.pop("optimization_stage", None)
                bert_worker.pop("optimization_variant", None)
                bert_worker.pop("optimization_device", None)
                bert["worker"] = bert_worker
            else:
                bert["role"] = "inactive · Apple optimization testing"
                bert["production_active"] = False
                optimization = bert.get("optimization") or {}
                stage = optimization.get("stage") or "CPU/MPS throughput and parity sweep"
                bert["assignment"] = f"M4 Apple optimization · {stage}"
            value.setdefault("fleet", {})["bert"] = bert
            self._apply_goal_projection(value)
            self._annotate_replay_progress(value)
            self._annotate_fleet_rates(value)
            self._annotate_scheduler_queues(value)
            value["network_latency"] = self._refresh_network_latency()
            value["dashboard_host"] = socket.gethostname()
            value["dashboard_sampled_at"] = time.time()
            self._annotate_source_integrity(value)
        except Exception as exc:  # keep the last good payload visible
            with self.lock:
                value = dict(self.value)
            if value.get("ok"):
                value.pop("error", None)
                value["telemetry_warning"] = str(exc)
            else:
                value["ok"] = False
                value["error"] = str(exc)
            value["dashboard_sampled_at"] = time.time()
        with self.lock:
            self.value = value

    def loop(self) -> None:
        while not self.stopping.is_set():
            started = time.monotonic()
            self.update()
            self.stopping.wait(max(0.25, 1.0 - (time.monotonic() - started)))

    def get(self) -> dict[str, Any]:
        with self.lock:
            return dict(self.value)


CACHE = SnapshotCache()


class DashboardHTTPServer(ThreadingHTTPServer):
    """Keep the LAN dashboard responsive across many open refresh tabs."""

    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128


class Handler(BaseHTTPRequestHandler):
    server_version = "PokeBotDashboard/1.0"

    def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.partition("?")[0]
        if path in ("/", "/index.html"):
            self.send_bytes(rendered_index(), "text/html; charset=utf-8")
            return
        if path == "/api/status":
            payload = CACHE.get()
            payload["dashboard_ui_version"] = dashboard_ui_version()
            body = json.dumps(payload, separators=(",", ":")).encode()
            self.send_bytes(body, "application/json; charset=utf-8")
            return
        if path == "/health":
            payload = CACHE.get()
            payload["dashboard_ui_version"] = dashboard_ui_version()
            status = HTTPStatus.OK if payload.get("ok") else HTTPStatus.SERVICE_UNAVAILABLE
            self.send_bytes(json.dumps(payload).encode(), "application/json", status)
            return
        self.send_bytes(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    worker = threading.Thread(target=CACHE.loop, name="telemetry", daemon=True)
    worker.start()
    server = DashboardHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    finally:
        CACHE.stopping.set()
        server.server_close()


if __name__ == "__main__":
    main()
