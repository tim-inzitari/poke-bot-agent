from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from poke_bot.prize_plan_targets_v2 import (
    HORIZONS,
    PRIZE_PLAN_AUTHORITY_KEY,
    PrizePlanTargetError,
    _fit_monotone_potential_table,
    build_prize_plan_target_overlay_day,
    canonical_bytes,
    canonical_sha256,
    finalize_prize_plan_target_set,
    fit_prize_plan_potential_v2,
    sha256_file,
)


FIT_CONFIG = {
    "algorithm": "alternating_weighted_2d_isotonic_pava/v1",
    "smoothing_prior_strength": 8.0,
    "max_iterations": 10_000,
    "convergence_tolerance": 1e-10,
}


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(value))
    return sha256_file(path)


def _contract(path: Path, *, top_goal_revision: int = 23) -> str:
    target = {
        "row_schema": "poke_bot.alakazam_prize_plan_target_overlay/v2",
        "manifest_schema": "poke_bot.alakazam_prize_plan_target_set_manifest/v2",
        "day_manifest_schema": "poke_bot.alakazam_prize_plan_target_day_manifest/v2",
        "day_materialization_receipt_schema": "poke_bot.alakazam_prize_plan_target_day_materialization_receipt/v2",
        "target_set_materialization_receipt_schema": "poke_bot.alakazam_prize_plan_target_set_materialization_receipt/v2",
        "row_unit": "one_complete_recorded_chosen_action_program",
        "row_join_identity": [
            "utc_day",
            "source_archive_sha256",
            "source_member",
            "episode_id",
            "acting_seat",
            "env_step",
            "program_identity",
        ],
        "horizon_definition": "next_complete_same_seat_actions_with_all_intervening_opponent_activity_included",
        "segment_start": "public_pre_action_state_for_complete_same_seat_action_i_plus_k",
        "segment_end": "public_pre_action_state_for_complete_same_seat_action_i_plus_k_plus_1",
        "public_evidence_only": [
            "public_remaining_prize_counts",
            "exact_public_transition_and_event_evidence",
            "sealed_complete_action_and_public_observation_alignment",
        ],
        "hidden_prize_identity_or_other_hidden_information_allowed": False,
        "terminal_after_state_inference_allowed": False,
        "terminal_observed_z_is_direct_plan_reward_or_actor_term": False,
        "prize_race_potential": {
            "fit_manifest_schema": "poke_bot.alakazam_prize_plan_phi_fit_manifest/v2",
            "fit_receipt_schema": "poke_bot.alakazam_prize_plan_phi_fit_receipt/v2",
            "frozen_table_schema": "poke_bot.alakazam_prize_plan_phi_table/v2",
            "definition": "Phi(our_remaining,opponent_remaining)=2*P_iso(win|counts)-1",
            "fit_scope": "sealed_train_split_only",
            "fit_examples": "causally_available_public_count_pairs_with_observed_completed_trajectory_win_indicator",
            "smoothing_required": True,
            "monotone_constraints": {
                "Phi_when_our_remaining_count_falls": "must_not_decrease",
                "Phi_when_opponent_remaining_count_falls": "must_not_increase",
            },
            "fit_input_manifest_sha256_bound": True,
            "fit_configuration_sha256_bound": True,
            "frozen_table_sha256_bound": True,
            "validation_evaluation_or_runtime_refit_allowed": False,
        },
        "segment_shaping_reward": "rP_t=gamma*Phi(s_t_plus_1)-Phi(s_t)",
        "gamma": {
            "must_be_explicit_fixed_and_receipt_bound_before_materialization_or_actor_use": True,
            "may_silently_default": False,
            "fit_or_tune_on_validation_evaluation_or_runtime": False,
        },
        "horizon_return": "sum_{k=0}^{h-1}gamma^k*rP_{t+k}_over_exact_complete_same_seat_segments",
        "H3_return_requires_exact_segment_count": 3,
        "missing_ambiguous_nonmonotone_or_terminal_censored_evidence_behavior": "mask_target_and_interval_never_assign_zero",
        "m3_requires_all_h3_segments_available": True,
        "closest_valid_diagnostic_target_only_if_exact_target_is_impossible": True,
        "materialization_failure_behavior": "record_measured_schema_or_evidence_blocker_keep_legacy_active_never_fabricate_labels",
    }
    authority = {
        "owner_goal_revision": 23,
        "public_prize_plan_target": target,
        "sidecar_strategy": {
            "default_safe_implementation": "separately_versioned_prize_plan_v2_sidecar",
            "sidecar_schema": "poke_bot.alakazam_prize_plan_v2_sidecar/v1",
            "plan_horizons_to_train_and_receipt": [1, 3, 6, 12],
        },
        "actor_advantage": {
            "enabled_formula": "(z-V_existing(s))+0.025*m3*c3*(Q_plan_3(s,a)-V_plan_3(s))",
            "selected_nonzero_cumulative_prize_horizon": 3,
            "simultaneous_or_additive_H1_H3_H6_H12_actor_terms_allowed": False,
        },
    }
    return _write_json(
        path,
        {
            # The r23 semantic authority survives unrelated later wrapper
            # revisions; target artifacts bind both identities separately.
            "goal_revision": top_goal_revision,
            PRIZE_PLAN_AUTHORITY_KEY: authority,
        },
    )


def _player(prizes: int) -> dict[str, Any]:
    return {"prize": [None] * prizes}


def _step(*, seat: int, own: int, opponent: int, action: list[int] | None, status: str = "ACTIVE") -> list[dict[str, Any]]:
    players = [_player(opponent), _player(opponent)]
    players[seat] = _player(own)
    return [
        {
            "observation": {"current": {"yourIndex": index, "players": players}},
            "action": action if index == seat else [],
            "status": status,
        }
        for index in range(2)
    ]


def _episode(
    *, episode_id: str, seat: int, counts: list[tuple[int, int]], reward: float = 1.0
) -> dict[str, Any]:
    # The raw action for observation N is in frame N+1.
    steps = [_step(seat=seat, own=own, opponent=opponent, action=None) for own, opponent in counts]
    steps.append(_step(seat=seat, own=counts[-1][0], opponent=counts[-1][1], action=[0]))
    for position in range(1, len(steps)):
        steps[position][seat]["action"] = [0]
    rewards = [reward, -reward] if seat == 0 else [-reward, reward]
    return {"id": episode_id, "statuses": ["DONE", "DONE"], "rewards": rewards, "steps": steps}


def _overlay(path: Path, *, day: str, raw_sha: str, member: str, episode_id: str, seat: int, count: int) -> str:
    rows = []
    for index in range(count):
        rows.append(
            {
                "schema": "poke_bot.alakazam_recent20_rtp_complete_action_overlay/v1",
                "utc_day": day,
                "source_archive_sha256": raw_sha,
                "source_member": member,
                "episode_id": episode_id,
                "acting_seat": seat,
                "env_step": index,
                "program_identity": f"{day}-program-{index}",
                "selected_action_program": [0],
                "recorded_successor_program_identity": f"{day}-program-{index + 1}" if index + 1 < count else None,
                "complete_action_program_reconstructed": True,
                "hidden_information_fields_present": False,
                "stages": [{"factorized_stage": 0}],
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(canonical_bytes(row) for row in rows))
    return sha256_file(path)


def _raw_zip(path: Path, payload: Mapping[str, Any], member: str = "episode.json") -> str:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(member, canonical_bytes(payload))
    return sha256_file(path)


def test_weighted_2d_isotonic_fit_is_monotone_and_deterministic() -> None:
    counts = {(own, opp): {"labeled_count": 0, "win_count": 0, "draw_count": 0, "loss_count": 0} for own in range(1, 7) for opp in range(1, 7)}
    counts[(6, 6)] = {"labeled_count": 10, "win_count": 9, "draw_count": 0, "loss_count": 1}
    counts[(1, 1)] = {"labeled_count": 10, "win_count": 1, "draw_count": 0, "loss_count": 9}
    table, diagnostics = _fit_monotone_potential_table(counts, config=FIT_CONFIG)
    assert diagnostics["monotonicity_validated"] is True
    values = {(row["our_remaining"], row["opponent_remaining"]): row["phi"] for row in table["cells"]}
    assert values[(1, 1)] >= values[(6, 1)]
    assert values[(1, 6)] >= values[(1, 1)]
    assert canonical_sha256(table) == canonical_sha256(_fit_monotone_potential_table(counts, config=FIT_CONFIG)[0])


def test_day_builder_masks_missing_horizons_and_uses_analytic_no_clip_transform(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_sha = _contract(contract_path)
    member = "episode.json"
    payload = _episode(
        episode_id="e1", seat=0, counts=[(6, 6), (5, 6), (4, 6), (1, 6)]
    )
    raw_zip = tmp_path / "raw.zip"
    raw_sha = _raw_zip(raw_zip, payload, member)
    overlay = tmp_path / "overlay.jsonl"
    overlay_sha = _overlay(overlay, day="2026-07-23", raw_sha=raw_sha, member=member, episode_id="e1", seat=0, count=4)
    # The small direct Phi fit fixture still requires all 20 identities.  Use
    # twenty copies only to exercise its explicit inventory check; decoding is
    # intentionally only from the train entry below in this isolated test.
    day_inputs = []
    for offset, day in enumerate([f"2026-07-{item:02d}" for item in range(23, 32)] + [f"2026-08-{item:02d}" for item in range(1, 12)]):
        source_overlay = overlay if day == "2026-07-23" else tmp_path / f"overlay-{offset}.jsonl"
        source_raw = raw_zip if day == "2026-07-23" else tmp_path / f"raw-{offset}.zip"
        day_raw_sha = raw_sha
        if day != "2026-07-23":
            # Supply train-only support that makes the public Prize race
            # potential meaningfully separate (6,6) from (1,6).  This gives a
            # genuine raw H3 return above one, proving the model target is
            # divided by its analytic bound rather than clipped.
            if offset <= 6:
                day_counts, day_reward = [(6, 6)], -1.0
            else:
                day_counts, day_reward = [(1, 6)], 1.0
            day_payload = _episode(
                episode_id=f"e-{offset}",
                seat=0,
                counts=day_counts,
                reward=day_reward,
            )
            day_raw_sha = _raw_zip(source_raw, day_payload, member)
            _overlay(source_overlay, day=day, raw_sha=day_raw_sha, member=member, episode_id=f"e-{offset}", seat=0, count=len(day_counts))
        day_inputs.append({"utc_day": day, "split": "train" if offset < 14 else "validation" if offset < 17 else "evaluation", "complete_action_overlay_path": str(source_overlay), "complete_action_overlay_sha256": sha256_file(source_overlay), "raw_episode_zip_path": str(source_raw), "raw_episode_zip_sha256": day_raw_sha})
    high_separation_fit_config = {
        **FIT_CONFIG,
        "smoothing_prior_strength": 0.01,
    }
    fit = fit_prize_plan_potential_v2(day_inputs=day_inputs, output_root=tmp_path / "phi", goal_contract_path=contract_path, expected_goal_contract_sha256=contract_sha, fit_configuration=high_separation_fit_config)
    result = build_prize_plan_target_overlay_day(
        complete_action_overlay_path=overlay,
        raw_episode_zip_path=raw_zip,
        output_root=tmp_path / "day",
        utc_day="2026-07-23",
        split="train",
        goal_contract_path=contract_path,
        expected_goal_contract_sha256=contract_sha,
        phi_fit_manifest_path=Path(fit["fit_manifest_path"]),
        expected_phi_fit_manifest_sha256=fit["fit_manifest_sha256"],
        gamma=1.0,
        expected_complete_action_overlay_sha256=overlay_sha,
        expected_raw_episode_zip_sha256=raw_sha,
    )
    manifest = json.loads(Path(result["day_manifest_path"]).read_text())
    target = manifest["target_shard"]
    rows = [json.loads(line) for line in (Path(result["output_root"]) / target["path"]).read_text().splitlines()]
    h3 = rows[0]["prize_plan_returns"]["h3"]
    assert h3["mask"] is True
    assert h3["raw_return_value"] > 1.0
    assert h3["model_target_value"] == pytest.approx(h3["raw_return_value"] / 2.0)
    assert len(rows[0]["causal_segments"]) == 3
    assert "segments" not in h3
    assert rows[0]["prize_plan_returns"]["h6"]["mask"] is False
    assert rows[0]["prize_plan_returns"]["h6"]["model_target_value"] is None


def test_finalizer_emits_portable_target_only_inventory(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    contract_sha = _contract(contract_path, top_goal_revision=24)
    member = "episode.json"
    dates = [f"2026-07-{item:02d}" for item in range(23, 32)] + [
        f"2026-08-{item:02d}" for item in range(1, 12)
    ]
    inputs: list[dict[str, Any]] = []
    for offset, day in enumerate(dates):
        # A unique archive/member/group per day exercises whole-day and group
        # split fencing while keeping the artifact fixture deliberately tiny.
        counts = [(6, 6), (5, 6), (4, 6), (1, 6)] if offset == 0 else [(6, 6)]
        reward = -1.0 if offset in {1, 2, 3, 4, 5, 6} else 1.0
        raw = tmp_path / f"{day}.zip"
        payload = _episode(episode_id=f"e-{offset}", seat=0, counts=counts, reward=reward)
        raw_sha = _raw_zip(raw, payload, member)
        overlay = tmp_path / f"{day}.jsonl"
        overlay_sha = _overlay(
            overlay,
            day=day,
            raw_sha=raw_sha,
            member=member,
            episode_id=f"e-{offset}",
            seat=0,
            count=len(counts),
        )
        split = "train" if offset < 14 else "validation" if offset < 17 else "evaluation"
        inputs.append(
            {
                "utc_day": day,
                "split": split,
                "complete_action_overlay_path": str(overlay),
                "complete_action_overlay_sha256": overlay_sha,
                "raw_episode_zip_path": str(raw),
                "raw_episode_zip_sha256": raw_sha,
            }
        )
    fit = fit_prize_plan_potential_v2(
        day_inputs=inputs,
        output_root=tmp_path / "phi",
        goal_contract_path=contract_path,
        expected_goal_contract_sha256=contract_sha,
        fit_configuration={**FIT_CONFIG, "smoothing_prior_strength": 0.01},
    )
    (tmp_path / "days").mkdir()
    day_roots = []
    overlay_shards = []
    for item in inputs:
        output = tmp_path / "days" / item["utc_day"]
        built = build_prize_plan_target_overlay_day(
            complete_action_overlay_path=item["complete_action_overlay_path"],
            raw_episode_zip_path=item["raw_episode_zip_path"],
            output_root=output,
            utc_day=item["utc_day"],
            split=item["split"],
            goal_contract_path=contract_path,
            expected_goal_contract_sha256=contract_sha,
            phi_fit_manifest_path=Path(fit["fit_manifest_path"]),
            expected_phi_fit_manifest_sha256=fit["fit_manifest_sha256"],
            gamma=1.0,
            expected_complete_action_overlay_sha256=item["complete_action_overlay_sha256"],
            expected_raw_episode_zip_sha256=item["raw_episode_zip_sha256"],
        )
        day_roots.append(Path(built["output_root"]))
        overlay_shards.append(
            {
                "utc_day": item["utc_day"],
                "split": item["split"],
                "sha256": item["complete_action_overlay_sha256"],
                "size_bytes": Path(item["complete_action_overlay_path"]).stat().st_size,
            }
        )
    overlay_manifest_path = tmp_path / "overlay-manifest.json"
    overlay_manifest_sha = _write_json(
        overlay_manifest_path,
        {
            "schema": "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1",
            "overlay_shards": overlay_shards,
        },
    )
    finalized = finalize_prize_plan_target_set(
        day_artifact_roots=day_roots,
        output_root=tmp_path / "target-set",
        goal_contract_path=contract_path,
        expected_goal_contract_sha256=contract_sha,
        phi_fit_manifest_path=Path(fit["fit_manifest_path"]),
        expected_phi_fit_manifest_sha256=fit["fit_manifest_sha256"],
        complete_action_overlay_manifest_path=overlay_manifest_path,
        expected_complete_action_overlay_manifest_sha256=overlay_manifest_sha,
        gamma=1.0,
    )
    manifest = json.loads(Path(finalized["target_set_manifest_path"]).read_text())
    assert manifest["schema"] == "poke_bot.alakazam_prize_plan_target_set_manifest/v2"
    assert manifest["goal_contract"]["goal_revision"] == 24
    assert manifest["goal_contract"]["required_authority"] == PRIZE_PLAN_AUTHORITY_KEY
    assert manifest["target_value_transform"]["schema"] == "poke_bot.alakazam_prize_plan_target_value_transform/v2"
    assert manifest["phi_fit"]["fit_scope"] == "sealed_train_split_only"
    assert len(manifest["target_days"]) == 20
    roles = {row["role"] for row in manifest["portable_objects"]}
    assert {"goal_contract", "phi_table", "phi_fit_manifest", "phi_fit_receipt", "target_shard", "target_day_manifest", "target_day_receipt"}.issubset(roles)
    assert manifest["information_boundary"]["raw_zip_or_feature_or_complete_action_overlay_payload_copied"] is False
