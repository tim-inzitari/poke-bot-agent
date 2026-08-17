from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pytest
import poke_bot.action_critic_targets as target_module

from poke_bot.action_critic_targets import (
    COMPLETE_ACTION_OVERLAY_SCHEMA,
    SPLIT_BY_DAY,
    TARGET_DAY_MANIFEST_SCHEMA,
    TARGET_OVERLAY_SCHEMA,
    TARGET_SET_MANIFEST_SCHEMA,
    WINDOW_DAYS,
    ActionCriticTargetError,
    build_action_critic_target_overlay_day,
    canonical_bytes,
    finalize_action_critic_target_set,
    resolve_action_critic_target_set_day,
    sha256_file,
)
from scripts.build_alakazam_action_critic_targets import main as target_main

DAY = "2026-07-23"
MEMBER = "episodes/critic-target-episode.json"
EPISODE = "critic-target-episode"


def _player(prizes: int, *, private_marker: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        # The zone has opaque public placeholders.  The target builder must
        # take its length only, never inspect the individual values.
        "prize": [None] * prizes,
    }
    if private_marker is not None:
        result["hand"] = [{"id": private_marker, "serial": private_marker + 1000}]
        result["privateState"] = {"unrevealed_prize_ids": [private_marker + 2000]}
    return result


def _observation(*, seat: int, own_prizes: int, opponent_prizes: int) -> dict[str, Any]:
    players = [_player(opponent_prizes, private_marker=811), _player(opponent_prizes)]
    players[seat] = _player(own_prizes, private_marker=733)
    players[1 - seat] = _player(opponent_prizes, private_marker=977)
    return {"current": {"yourIndex": seat, "players": players}}


def _zip_payload(
    path: Path,
    *,
    prize_counts: list[tuple[int, int]],
    seat: int = 0,
    rewards: list[float] | None = None,
    statuses: list[str] | None = None,
) -> str:
    steps: list[list[dict[str, Any]]] = [
        [{"action": [0]}, {"action": [0]}] for _ in range(len(prize_counts) + 1)
    ]
    for env_step, (own, opponent) in enumerate(prize_counts):
        steps[env_step][seat]["observation"] = _observation(
            seat=seat, own_prizes=own, opponent_prizes=opponent
        )
    payload = {
        "id": EPISODE,
        "rewards": [1.0, -1.0] if rewards is None else rewards,
        "statuses": ["DONE", "DONE"] if statuses is None else statuses,
        "steps": steps,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(MEMBER, json.dumps(payload, sort_keys=True))
    return sha256_file(path)


def _overlay_row(
    *,
    archive_sha: str,
    env_step: int,
    program: str,
    successor: str | None,
    seat: int = 0,
    outcome: float | None = 1.0,
) -> dict[str, Any]:
    return {
        "schema": COMPLETE_ACTION_OVERLAY_SCHEMA,
        "utc_day": DAY,
        "source_archive_sha256": archive_sha,
        "source_member": MEMBER,
        "episode_id": EPISODE,
        "acting_seat": seat,
        "env_step": env_step,
        "program_identity": program,
        "selected_action_program": [0],
        "recorded_successor_program_identity": successor,
        "recorded_outcome": outcome,
        "complete_action_program_reconstructed": True,
        "hidden_information_fields_present": False,
        "stages": [{"base_ref": {"option_start": 0, "option_count": 2}}],
    }


def _write_overlay(path: Path, rows: list[dict[str, Any]]) -> str:
    path.write_bytes(b"".join(canonical_bytes(row) for row in rows))
    return sha256_file(path)


def _goal_contract(path: Path) -> str:
    path.write_bytes(
        canonical_bytes(
            {
                "schema": "poke_bot.alakazam_elmo_rule_derivative_goal/v1",
                "goal_revision": 22,
                "root_handoff_revision": 324,
                "revision_21_draw_safe_critic_actor_canary": {
                    "owner_goal_revision": 21,
                    "root_owner_revision": 323,
                    "actor_advantage": {
                        "enabled_formula": "(z-V_existing(s))+0.05*m1*(Q_prize^1(s,a)-V_prize^1(s))",
                        "complete_action_value_broadcast_identically_across_selected_factorized_stages": True,
                        "actor_gradient_into_sidecar_allowed": False,
                    },
                    "target_overlay": {
                        "schema": TARGET_OVERLAY_SCHEMA,
                        "manifest_schema": TARGET_SET_MANIFEST_SCHEMA,
                        "row_join_identity": [
                            "utc_day",
                            "source_archive_sha256",
                            "source_member",
                            "episode_id",
                            "acting_seat",
                            "env_step",
                            "program_identity",
                        ],
                        "group_key": [
                            "source_archive_sha256",
                            "episode_id",
                            "acting_seat",
                        ],
                        "group_order": "strictly_increasing_env_step_no_duplicates",
                        "public_state_endpoint": "steps[env_step][acting_seat].observation.current",
                        "raw_action_alignment": "selected_complete_action_is_carried_by_steps[env_step+1][acting_seat].action_and_must_already_equal_the_sealed_complete_action_overlay",
                        "prize_count": {
                            "source": "length_only_of_public_current.players[seat].prize_or_exact_public_count_alias",
                            "valid_inclusive_range": [1, 6],
                            "zero_behavior": "mask_as_setup_or_uninitialized_never_treat_as_real_zero_progress",
                            "card_identities_copied_or_consumed": False,
                        },
                        "horizon_definition": {
                            "values": [1, 2, 3],
                            "start": "pre_action_public_prize_counts_at_complete_action_i",
                            "end": "pre_action_public_prize_counts_at_complete_action_i_plus_h_for_same_group",
                            "own_taken": "own_remaining_before-own_remaining_after",
                            "opponent_taken": "opponent_remaining_before-opponent_remaining_after",
                            "target": "clip((own_taken-opponent_taken)/3,-1,+1)",
                            "terminal_ending_interval": "mask_when_no_later_complete_same_seat_action_exists_do_not_infer_a_terminal_after_state",
                            "invalid_or_non_monotone_behavior": "mask_that_horizon_never_assign_zero",
                        },
                        "required_terminal_fields": [
                            "z",
                            "z_mask",
                            "win_target_one_only_for_z_plus1",
                            "win_target_mask",
                        ],
                        "required_per_horizon_fields": [
                            "h",
                            "mask",
                            "unavailable_reason",
                            "future_program_identity",
                            "future_env_step",
                            "own_remaining_before",
                            "own_remaining_after",
                            "opponent_remaining_before",
                            "opponent_remaining_after",
                            "own_taken",
                            "opponent_taken",
                            "differential",
                        ],
                        "target_set_manifest_must_bind": [
                            "goal_contract_sha256",
                            "base_pack_completion_sha256",
                            "complete_action_overlay_manifest_sha256",
                            "all_20_raw_episode_zip_sha256s",
                            "all_20_target_shard_sha256s_sizes_rows_and_split",
                            "train_validation_evaluation_day_lists",
                            "episode_and_seat_group_split_disjointness",
                            "terminal_and_each_horizon_mask_coverage",
                            "zero_prize_setup_mask_count",
                            "non_monotone_mask_count",
                        ],
                        "hidden_information_simulator_search_rtp_mcts_or_unchosen_targets_allowed": False,
                    },
                },
            }
        )
    )
    return sha256_file(path)


def _load_target_rows(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(result["target_shard_path"]).read_text(encoding="utf-8").splitlines()
    ]


def _contains_key(value: Any, wanted: str) -> bool:
    if isinstance(value, dict):
        return wanted in value or any(_contains_key(child, wanted) for child in value.values())
    if isinstance(value, list):
        return any(_contains_key(child, wanted) for child in value)
    return False


def _contains_scalar(value: Any, wanted: Any) -> bool:
    if isinstance(value, dict):
        return any(_contains_scalar(child, wanted) for child in value.values())
    if isinstance(value, list):
        return any(_contains_scalar(child, wanted) for child in value)
    return value == wanted


def test_atomic_directory_publication_is_no_replace_and_fsyncs_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / ".private-source"
    source.mkdir()
    (source / "objects").mkdir()
    destination = tmp_path / "published"
    synced: list[Path] = []
    original_fsync_directory = target_module._fsync_directory

    def tracking_fsync(path: Path | str) -> None:
        synced.append(Path(path).resolve())
        original_fsync_directory(path)

    monkeypatch.setattr(target_module, "_fsync_directory", tracking_fsync)
    target_module._atomic_publish_directory_noreplace(source, destination)
    assert not source.exists()
    assert destination.is_dir()
    assert synced[-1] == tmp_path.resolve()

    competing = tmp_path / ".private-competing"
    competing.mkdir()
    marker = destination / "original-marker"
    marker.write_text("unchanged", encoding="utf-8")
    with pytest.raises(ActionCriticTargetError, match="output root already exists"):
        target_module._atomic_publish_directory_noreplace(competing, destination)
    assert competing.is_dir()
    assert marker.read_text(encoding="utf-8") == "unchanged"


def test_builds_content_addressed_target_overlay_with_masked_horizons(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract_sha = _goal_contract(contract)
    raw_zip = tmp_path / "raw.zip"
    # Pre-action same-seat counts: the first action takes one Prize, the next
    # interval loses one Prize to the opponent, and the final interval takes
    # two more.  The h=3 target is therefore (3 - 1) / 3.
    raw_sha = _zip_payload(
        raw_zip,
        prize_counts=[(6, 6), (5, 6), (5, 5), (3, 5)],
    )
    overlay = tmp_path / "complete-action.jsonl"
    overlay_sha = _write_overlay(
        overlay,
        [
            _overlay_row(
                archive_sha=raw_sha, env_step=0, program="p0", successor="p1"
            ),
            _overlay_row(
                archive_sha=raw_sha, env_step=1, program="p1", successor="p2"
            ),
            _overlay_row(
                archive_sha=raw_sha, env_step=2, program="p2", successor="p3"
            ),
            _overlay_row(
                archive_sha=raw_sha, env_step=3, program="p3", successor=None
            ),
        ],
    )

    result = build_action_critic_target_overlay_day(
        complete_action_overlay_path=overlay,
        raw_episode_zip_path=raw_zip,
        output_root=tmp_path / "sealed-targets",
        utc_day=DAY,
        split="train",
        goal_contract_path=contract,
        expected_goal_contract_sha256=contract_sha,
        expected_complete_action_overlay_sha256=overlay_sha,
        expected_raw_episode_zip_sha256=raw_sha,
    )

    rows = _load_target_rows(result)
    assert [row["program_identity"] for row in rows] == ["p0", "p1", "p2", "p3"]
    assert all(row["schema"] == TARGET_OVERLAY_SCHEMA for row in rows)
    assert all(row["terminal_win"] == {"mask": True, "unavailable_reason": None, "value": 1.0} for row in rows)
    assert rows[0]["z"] == 1.0
    assert rows[0]["z_mask"] is True
    assert rows[0]["win_target_one_only_for_z_plus1"] == 1.0
    assert rows[0]["prize_differential"]["h1"]["differential"] == pytest.approx(1 / 3)
    assert rows[0]["prize_differential"]["h2"]["differential"] == pytest.approx(0.0)
    assert rows[0]["prize_differential"]["h3"]["differential"] == pytest.approx(2 / 3)
    assert rows[1]["prize_differential"]["h1"]["differential"] == pytest.approx(-1 / 3)
    assert rows[1]["prize_differential"]["h2"]["differential"] == pytest.approx(1 / 3)
    assert rows[1]["prize_differential"]["h3"] == {
        "h": 3,
        "mask": False,
        "unavailable_reason": "no_later_same_seat_recorded_program",
        "future_program_identity": None,
        "future_env_step": None,
        "own_remaining_before": 5,
        "own_remaining_after": None,
        "opponent_remaining_before": 6,
        "opponent_remaining_after": None,
        "own_taken": None,
        "opponent_taken": None,
        "differential": None,
    }
    assert rows[3]["prize_differential"]["h1"]["mask"] is False
    assert rows[3]["prize_differential"]["h1"]["differential"] is None

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema"] == TARGET_DAY_MANIFEST_SCHEMA
    assert manifest["goal_contract"]["sha256"] == contract_sha
    assert manifest["raw_episode_zip"]["sha256"] == raw_sha
    assert manifest["complete_action_overlay"]["sha256"] == overlay_sha
    assert manifest["coverage"]["counts"]["prize_h1_labeled"] == 3
    assert manifest["coverage"]["counts"]["prize_h3_labeled"] == 1
    assert manifest["information_boundary"]["prize_zone_card_identities_read"] is False
    assert manifest["information_boundary"]["search_or_planner_called"] is False
    assert not _contains_key(rows, "privateState")
    # The opaque values embedded in the raw hands/prize private state never
    # become an output byte.
    assert not _contains_scalar(rows, 977)


def test_current_goal_contract_wrapper_binds_embedded_revision_21_semantics(
    tmp_path: Path,
) -> None:
    current_contract = (
        Path(__file__).resolve().parents[1]
        / "goals"
        / "alakazam-elmo-rule-derivative"
        / "contract.json"
    )
    current_contract_sha = sha256_file(current_contract)
    current_contract_body = json.loads(current_contract.read_text(encoding="utf-8"))
    raw_zip = tmp_path / "raw.zip"
    raw_sha = _zip_payload(raw_zip, prize_counts=[(6, 6)])
    overlay = tmp_path / "complete-action.jsonl"
    overlay_sha = _write_overlay(
        overlay,
        [_overlay_row(archive_sha=raw_sha, env_step=0, program="p0", successor=None)],
    )

    result = build_action_critic_target_overlay_day(
        complete_action_overlay_path=overlay,
        raw_episode_zip_path=raw_zip,
        output_root=tmp_path / "current-contract-targets",
        utc_day=DAY,
        split="train",
        goal_contract_path=current_contract,
        expected_goal_contract_sha256=current_contract_sha,
        expected_complete_action_overlay_sha256=overlay_sha,
        expected_raw_episode_zip_sha256=raw_sha,
    )

    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["goal_contract"]["sha256"] == current_contract_sha
    assert manifest["goal_contract"]["goal_revision"] == current_contract_body["goal_revision"]
    assert manifest["goal_contract"]["critic_semantic_owner_goal_revision"] == 21


@pytest.mark.parametrize("drift", ["raw_action", "horizon_endpoint", "manifest_bindings"])
def test_goal_contract_refuses_critic_target_semantic_drift(
    tmp_path: Path, drift: str
) -> None:
    contract = tmp_path / "contract.json"
    _goal_contract(contract)
    payload = json.loads(contract.read_text(encoding="utf-8"))
    target = payload["revision_21_draw_safe_critic_actor_canary"]["target_overlay"]
    if drift == "raw_action":
        target["raw_action_alignment"] = "steps[env_step].action"
    elif drift == "horizon_endpoint":
        target["horizon_definition"]["end"] = "post_action_frame"
    else:
        target["target_set_manifest_must_bind"] = target[
            "target_set_manifest_must_bind"
        ][:-1]
    contract.write_bytes(canonical_bytes(payload))

    with pytest.raises(ActionCriticTargetError, match="semantics drifted"):
        target_module._load_goal_contract(
            contract, expected_sha256=sha256_file(contract)
        )


def test_incomplete_or_ambiguous_interval_masks_only_that_prize_target(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract_sha = _goal_contract(contract)
    raw_zip = tmp_path / "raw.zip"
    raw_sha = _zip_payload(raw_zip, prize_counts=[(6, 6), (5, 6)])
    overlay = tmp_path / "complete-action.jsonl"
    overlay_sha = _write_overlay(
        overlay,
        [
            _overlay_row(
                archive_sha=raw_sha,
                env_step=0,
                program="p0",
                successor="not-the-recorded-next-program",
            ),
            _overlay_row(archive_sha=raw_sha, env_step=1, program="p1", successor=None),
        ],
    )

    result = build_action_critic_target_overlay_day(
        complete_action_overlay_path=overlay,
        raw_episode_zip_path=raw_zip,
        output_root=tmp_path / "sealed-targets",
        utc_day=DAY,
        split="train",
        goal_contract_path=contract,
        expected_goal_contract_sha256=contract_sha,
        expected_complete_action_overlay_sha256=overlay_sha,
        expected_raw_episode_zip_sha256=raw_sha,
    )
    first, second = _load_target_rows(result)
    assert first["terminal_win"]["mask"] is True
    assert first["prize_differential"]["h1"] == {
        "h": 1,
        "mask": False,
        "unavailable_reason": "same_seat_successor_program_link_incomplete_or_ambiguous",
        "future_program_identity": "p1",
        "future_env_step": 1,
        "own_remaining_before": 6,
        "own_remaining_after": 5,
        "opponent_remaining_before": 6,
        "opponent_remaining_after": 6,
        "own_taken": None,
        "opponent_taken": None,
        "differential": None,
    }
    assert second["prize_differential"]["h1"]["mask"] is False
    assert second["prize_differential"]["h1"]["differential"] is None


def test_refuses_hidden_complete_action_input_and_digest_drift(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract_sha = _goal_contract(contract)
    raw_zip = tmp_path / "raw.zip"
    raw_sha = _zip_payload(raw_zip, prize_counts=[(6, 6)])
    overlay = tmp_path / "complete-action.jsonl"
    row = _overlay_row(archive_sha=raw_sha, env_step=0, program="p0", successor=None)
    row["opponent_hand_identities"] = [999]
    overlay_sha = _write_overlay(overlay, [row])

    with pytest.raises(ActionCriticTargetError, match="forbidden hidden-state"):
        build_action_critic_target_overlay_day(
            complete_action_overlay_path=overlay,
            raw_episode_zip_path=raw_zip,
            output_root=tmp_path / "hidden-refused",
            utc_day=DAY,
            split="train",
            goal_contract_path=contract,
            expected_goal_contract_sha256=contract_sha,
            expected_complete_action_overlay_sha256=overlay_sha,
            expected_raw_episode_zip_sha256=raw_sha,
        )
    assert not (tmp_path / "hidden-refused").exists()

    clean_overlay = tmp_path / "clean-complete-action.jsonl"
    clean_sha = _write_overlay(
        clean_overlay,
        [_overlay_row(archive_sha=raw_sha, env_step=0, program="p0", successor=None)],
    )
    with pytest.raises(ActionCriticTargetError, match="raw episode ZIP SHA-256 mismatch"):
        build_action_critic_target_overlay_day(
            complete_action_overlay_path=clean_overlay,
            raw_episode_zip_path=raw_zip,
            output_root=tmp_path / "digest-refused",
            utc_day=DAY,
            split="train",
            goal_contract_path=contract,
            expected_goal_contract_sha256=contract_sha,
            expected_complete_action_overlay_sha256=clean_sha,
            expected_raw_episode_zip_sha256="sha256:" + "0" * 64,
        )
    assert not (tmp_path / "digest-refused").exists()


def test_raw_zip_rejects_duplicate_physical_source_member(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract_sha = _goal_contract(contract)
    raw_zip = tmp_path / "duplicate-member.zip"
    _zip_payload(raw_zip, prize_counts=[(6, 6)])
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(raw_zip, "a", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                MEMBER,
                json.dumps(
                    {
                        "id": EPISODE,
                        "rewards": [1.0, -1.0],
                        "statuses": ["DONE", "DONE"],
                        "steps": [],
                    },
                    sort_keys=True,
                ),
            )
    raw_sha = sha256_file(raw_zip)
    overlay = tmp_path / "complete-action.jsonl"
    overlay_sha = _write_overlay(
        overlay,
        [_overlay_row(archive_sha=raw_sha, env_step=0, program="p0", successor=None)],
    )

    with pytest.raises(ActionCriticTargetError, match="duplicate physical member name"):
        build_action_critic_target_overlay_day(
            complete_action_overlay_path=overlay,
            raw_episode_zip_path=raw_zip,
            output_root=tmp_path / "duplicate-member-refused",
            utc_day=DAY,
            split="train",
            goal_contract_path=contract,
            expected_goal_contract_sha256=contract_sha,
            expected_complete_action_overlay_sha256=overlay_sha,
            expected_raw_episode_zip_sha256=raw_sha,
        )
    assert not (tmp_path / "duplicate-member-refused").exists()


def test_masks_numeric_timeout_reward_instead_of_treating_it_as_terminal_win(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "contract.json"
    contract_sha = _goal_contract(contract)
    raw_zip = tmp_path / "timeout.zip"
    raw_sha = _zip_payload(
        raw_zip,
        prize_counts=[(6, 6)],
        statuses=["TIMEOUT", "TIMEOUT"],
    )
    overlay = tmp_path / "complete-action.jsonl"
    overlay_sha = _write_overlay(
        overlay,
        [_overlay_row(archive_sha=raw_sha, env_step=0, program="p0", successor=None)],
    )
    result = build_action_critic_target_overlay_day(
        complete_action_overlay_path=overlay,
        raw_episode_zip_path=raw_zip,
        output_root=tmp_path / "timeout-targets",
        utc_day=DAY,
        split="train",
        goal_contract_path=contract,
        expected_goal_contract_sha256=contract_sha,
        expected_complete_action_overlay_sha256=overlay_sha,
        expected_raw_episode_zip_sha256=raw_sha,
    )
    row = _load_target_rows(result)[0]
    assert row["terminal_win"] == {
        "mask": False,
        "unavailable_reason": "terminal_statuses_not_exact_done_pair",
        "value": None,
    }


@pytest.mark.parametrize(
    ("counts", "reason"),
    [
        (
            [(0, 0), (5, 5)],
            "pre_action_public_prize_count_outside_valid_1_to_6_range",
        ),
        ([(5, 5), (6, 5)], "non_monotone_public_prize_count"),
    ],
)
def test_setup_zero_and_non_monotone_public_prize_intervals_mask_not_zero(
    tmp_path: Path, counts: list[tuple[int, int]], reason: str
) -> None:
    contract = tmp_path / "contract.json"
    contract_sha = _goal_contract(contract)
    raw_zip = tmp_path / "raw.zip"
    raw_sha = _zip_payload(raw_zip, prize_counts=counts)
    overlay = tmp_path / "complete-action.jsonl"
    overlay_sha = _write_overlay(
        overlay,
        [
            _overlay_row(archive_sha=raw_sha, env_step=0, program="p0", successor="p1"),
            _overlay_row(archive_sha=raw_sha, env_step=1, program="p1", successor=None),
        ],
    )
    result = build_action_critic_target_overlay_day(
        complete_action_overlay_path=overlay,
        raw_episode_zip_path=raw_zip,
        output_root=tmp_path / "masked-targets",
        utc_day=DAY,
        split="train",
        goal_contract_path=contract,
        expected_goal_contract_sha256=contract_sha,
        expected_complete_action_overlay_sha256=overlay_sha,
        expected_raw_episode_zip_sha256=raw_sha,
    )
    h1 = _load_target_rows(result)[0]["prize_differential"]["h1"]
    assert h1["mask"] is False
    assert h1["differential"] is None
    assert h1["unavailable_reason"] == reason
    counter = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))["coverage"]["counts"]
    if reason == "non_monotone_public_prize_count":
        assert counter["non_monotone_mask_count"] == 1
    else:
        assert counter["zero_prize_setup_mask_count"] == 1


def _masked_horizon(horizon: int) -> dict[str, Any]:
    return {
        "h": horizon,
        "mask": False,
        "unavailable_reason": "no_later_same_seat_recorded_program",
        "future_program_identity": None,
        "future_env_step": None,
        "own_remaining_before": 6,
        "own_remaining_after": None,
        "opponent_remaining_before": 6,
        "opponent_remaining_after": None,
        "own_taken": None,
        "opponent_taken": None,
        "differential": None,
    }


def _sealed_day_artifact(
    root: Path, *, day: str, split: str, contract_sha: str, overlay_sha: str
) -> Path:
    root.mkdir()
    (root / "objects").mkdir()
    (root / "manifests").mkdir()
    (root / "receipts").mkdir()
    archive_sha = "sha256:" + ("a" if day < "2026-08-01" else "b") * 64
    target_row = {
        "schema": TARGET_OVERLAY_SCHEMA,
        "owner_goal_revision": 21,
        "goal_contract_goal_revision": 22,
        "utc_day": day,
        "split": split,
        "source_archive_sha256": archive_sha,
        "source_member": f"{day}.json",
        "episode_id": f"episode-{day}",
        "acting_seat": 0,
        "env_step": 1,
        "program_identity": f"program-{day}",
        "z": 1.0,
        "z_mask": True,
        "win_target_one_only_for_z_plus1": 1.0,
        "win_target_mask": True,
        "prize_differential": {f"h{h}": _masked_horizon(h) for h in (1, 2, 3)},
        "target_only": True,
        "hidden_information_fields_present": False,
    }
    target_path = root / "objects" / "targets.jsonl"
    target_path.write_bytes(canonical_bytes(target_row))
    target_sha = sha256_file(target_path)
    counts = {
        "complete_action_programs": 1,
        "terminal_win_labeled": 1,
        "prize_h1_masked": 1,
        "prize_h2_masked": 1,
        "prize_h3_masked": 1,
    }
    manifest = {
        "schema": TARGET_DAY_MANIFEST_SCHEMA,
        "owner_goal_revision": 21,
        "goal_contract": {
            "sha256": contract_sha,
            "goal_revision": 22,
            "critic_semantic_owner_goal_revision": 21,
            "required_authority": "revision_21_draw_safe_critic_actor_canary",
        },
        "utc_day": day,
        "split": split,
        "complete_action_overlay": {"sha256": overlay_sha},
        "raw_episode_zip": {
            "sha256": archive_sha,
            "size_bytes": 123,
            "source_archive_sha256_verified": True,
        },
        "target_shard": {
            "path": "objects/targets.jsonl",
            "sha256": target_sha,
            "size_bytes": target_path.stat().st_size,
            "row_count": 1,
        },
        "coverage": {"counts": counts},
    }
    manifest_path = root / "manifests" / "manifest.json"
    manifest_path.write_bytes(canonical_bytes(manifest))
    manifest_sha = sha256_file(manifest_path)
    receipt = {
        "schema": "poke_bot.alakazam_action_critic_target_day_receipt/v1",
        "owner_goal_revision": 21,
        "goal_contract_goal_revision": 22,
        "critic_semantic_owner_goal_revision": 21,
        "goal_contract_sha256": contract_sha,
        "manifest_path": "manifests/manifest.json",
        "manifest_sha256": manifest_sha,
        "coverage": manifest["coverage"],
        "complete_action_overlay_sha256": overlay_sha,
        "raw_episode_zip_sha256": archive_sha,
        "target_shard_sha256": target_sha,
        "target_shard_size_bytes": target_path.stat().st_size,
        "target_row_count": 1,
    }
    (root / "receipts" / "receipt.json").write_bytes(canonical_bytes(receipt))
    return root


def test_finalizer_binds_exact_20_day_set_and_split_disjointness(tmp_path: Path) -> None:
    contract = tmp_path / "contract.json"
    contract_sha = _goal_contract(contract)
    base = tmp_path / "base-completion.json"
    base.write_bytes(
        canonical_bytes({"schema": "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1"})
    )
    base_sha = sha256_file(base)
    overlay_shards = []
    for number, day in enumerate(WINDOW_DAYS):
        split = "train" if number < 14 else "validation" if number < 17 else "evaluation"
        overlay_shards.append(
            {"utc_day": day, "split": split, "sha256": "sha256:" + f"{number:064x}"}
        )
    overlay_manifest = tmp_path / "overlay-manifest.json"
    overlay_manifest.write_bytes(
        canonical_bytes(
            {
                "schema": "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1",
                "overlay_shards": overlay_shards,
            }
        )
    )
    overlay_manifest_sha = sha256_file(overlay_manifest)
    roots = [
        _sealed_day_artifact(
            tmp_path / f"day-{index}",
            day=day,
            split=item["split"],
            contract_sha=contract_sha,
            overlay_sha=item["sha256"],
        )
        for index, (day, item) in enumerate(zip(WINDOW_DAYS, overlay_shards, strict=True))
    ]
    result = finalize_action_critic_target_set(
        day_artifact_roots=list(reversed(roots)),
        output_root=tmp_path / "target-set",
        goal_contract_path=contract,
        expected_goal_contract_sha256=contract_sha,
        base_pack_completion_path=base,
        expected_base_pack_completion_sha256=base_sha,
        complete_action_overlay_manifest_path=overlay_manifest,
        expected_complete_action_overlay_manifest_sha256=overlay_manifest_sha,
    )
    sealed = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert sealed["schema"] == TARGET_SET_MANIFEST_SCHEMA
    assert [item["utc_day"] for item in sealed["target_days"]] == list(WINDOW_DAYS)
    assert sealed["episode_and_seat_group_split_disjoint"] is True
    assert result["day_count"] == 20
    assert sealed["goal_contract_goal_revision"] == 22
    assert sealed["critic_semantic_owner_goal_revision"] == 21
    assert sealed["required_critic_authority"] == "revision_21_draw_safe_critic_actor_canary"
    for name in (
        "goal_contract",
        "base_pack_completion",
        "complete_action_overlay_manifest",
    ):
        assert not Path(sealed[name]["path"]).is_absolute()
        assert (Path(result["output_root"]) / sealed[name]["path"]).is_file()
    first = sealed["target_days"][0]
    assert first["day_artifact_root"] == f"days/{WINDOW_DAYS[0]}"
    assert first["day_manifest_path"].startswith(f"days/{WINDOW_DAYS[0]}/")
    assert first["day_receipt_path"].startswith(f"days/{WINDOW_DAYS[0]}/")
    assert not Path(first["day_manifest_path"]).is_absolute()
    assert not Path(first["day_receipt_path"]).is_absolute()
    assert not Path(first["target_shard"]["path"]).is_absolute()

    original_root = Path(result["output_root"])
    manifest_relative = Path(result["manifest_path"]).relative_to(original_root)
    relocated_root = tmp_path / "target-set-relocated"
    original_root.rename(relocated_root)
    relocated = json.loads((relocated_root / manifest_relative).read_text(encoding="utf-8"))
    resolved = resolve_action_critic_target_set_day(relocated_root, relocated["target_days"][0])
    assert resolved["target_shard_path"].is_file()
    assert resolved["target_shard_path"].is_relative_to(relocated_root)


@pytest.mark.parametrize(
    ("drift", "message"),
    [
        ("count", "exactly 20 shard entries"),
        ("duplicate_day", "days must be unique"),
        ("split", "split drifted"),
    ],
)
def test_finalizer_rejects_overlay_inventory_before_day_dict_collapse(
    tmp_path: Path, drift: str, message: str
) -> None:
    contract = tmp_path / "contract.json"
    contract_sha = _goal_contract(contract)
    base = tmp_path / "base-completion.json"
    base.write_bytes(
        canonical_bytes({"schema": "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1"})
    )
    base_sha = sha256_file(base)
    overlay_shards = [
        {
            "utc_day": day,
            "split": SPLIT_BY_DAY[day],
            "sha256": "sha256:" + f"{index:064x}",
        }
        for index, day in enumerate(WINDOW_DAYS)
    ]
    if drift == "count":
        overlay_shards.pop()
    elif drift == "duplicate_day":
        overlay_shards[-1] = dict(overlay_shards[0])
    else:
        overlay_shards[0]["split"] = "evaluation"
    overlay_manifest = tmp_path / "overlay-manifest.json"
    overlay_manifest.write_bytes(
        canonical_bytes(
            {
                "schema": "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1",
                "overlay_shards": overlay_shards,
            }
        )
    )
    roots = []
    for index in range(len(WINDOW_DAYS)):
        root = tmp_path / f"unread-day-{index}"
        root.mkdir()
        roots.append(root)

    with pytest.raises(ActionCriticTargetError, match=message):
        finalize_action_critic_target_set(
            day_artifact_roots=roots,
            output_root=tmp_path / "target-set-refused",
            goal_contract_path=contract,
            expected_goal_contract_sha256=contract_sha,
            base_pack_completion_path=base,
            expected_base_pack_completion_sha256=base_sha,
            complete_action_overlay_manifest_path=overlay_manifest,
            expected_complete_action_overlay_manifest_sha256=sha256_file(overlay_manifest),
        )
    assert not (tmp_path / "target-set-refused").exists()


def _rewrite_sealed_day_rows(root: Path, rows: list[dict[str, Any]]) -> None:
    """Keep the fake sealed-day receipt internally consistent after a test mutation."""

    target_path = root / "objects" / "targets.jsonl"
    target_path.write_bytes(b"".join(canonical_bytes(row) for row in rows))
    target_sha = sha256_file(target_path)
    counts = {
        "complete_action_programs": len(rows),
        "terminal_win_labeled": len(rows),
        "prize_h1_masked": len(rows),
        "prize_h2_masked": len(rows),
        "prize_h3_masked": len(rows),
    }
    manifest_path = root / "manifests" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["target_shard"] = {
        "path": "objects/targets.jsonl",
        "sha256": target_sha,
        "size_bytes": target_path.stat().st_size,
        "row_count": len(rows),
    }
    manifest["coverage"] = {"counts": counts}
    manifest_path.write_bytes(canonical_bytes(manifest))
    manifest_sha = sha256_file(manifest_path)
    receipt_path = root / "receipts" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["manifest_sha256"] = manifest_sha
    receipt["coverage"] = manifest["coverage"]
    receipt["target_shard_sha256"] = target_sha
    receipt["target_shard_size_bytes"] = target_path.stat().st_size
    receipt["target_row_count"] = len(rows)
    receipt_path.write_bytes(canonical_bytes(receipt))


def test_finalizer_refuses_source_member_mismatch_within_raw_episode(
    tmp_path: Path,
) -> None:
    contract = tmp_path / "contract.json"
    contract_sha = _goal_contract(contract)
    base = tmp_path / "base-completion.json"
    base.write_bytes(
        canonical_bytes({"schema": "poke_bot.alakazam_recent20_semantic_tensor_pack_completion/v1"})
    )
    base_sha = sha256_file(base)
    overlay_shards = [
        {
            "utc_day": day,
            "split": "train" if index < 14 else "validation" if index < 17 else "evaluation",
            "sha256": "sha256:" + f"{index:064x}",
        }
        for index, day in enumerate(WINDOW_DAYS)
    ]
    overlay_manifest = tmp_path / "overlay-manifest.json"
    overlay_manifest.write_bytes(
        canonical_bytes(
            {
                "schema": "poke_bot.alakazam_recent20_rtp_overlay_manifest/v1",
                "overlay_shards": overlay_shards,
            }
        )
    )
    overlay_manifest_sha = sha256_file(overlay_manifest)
    roots = [
        _sealed_day_artifact(
            tmp_path / f"day-{index}",
            day=day,
            split=item["split"],
            contract_sha=contract_sha,
            overlay_sha=item["sha256"],
        )
        for index, (day, item) in enumerate(zip(WINDOW_DAYS, overlay_shards, strict=True))
    ]
    altered_path = roots[0] / "objects" / "targets.jsonl"
    original = json.loads(altered_path.read_text(encoding="utf-8"))
    conflicting = dict(original)
    conflicting["source_member"] = "different-member-for-same-episode.json"
    conflicting["env_step"] = original["env_step"] + 1
    conflicting["program_identity"] = original["program_identity"] + "-second"
    _rewrite_sealed_day_rows(roots[0], [original, conflicting])

    with pytest.raises(ActionCriticTargetError, match="source_member conflicts"):
        finalize_action_critic_target_set(
            day_artifact_roots=roots,
            output_root=tmp_path / "target-set-refused",
            goal_contract_path=contract,
            expected_goal_contract_sha256=contract_sha,
            base_pack_completion_path=base,
            expected_base_pack_completion_sha256=base_sha,
            complete_action_overlay_manifest_path=overlay_manifest,
            expected_complete_action_overlay_manifest_sha256=overlay_manifest_sha,
        )
    assert not (tmp_path / "target-set-refused").exists()


def test_cli_requires_exact_identities_and_publishes_new_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    contract = tmp_path / "contract.json"
    contract_sha = _goal_contract(contract)
    raw_zip = tmp_path / "raw.zip"
    raw_sha = _zip_payload(raw_zip, prize_counts=[(6, 6)])
    overlay = tmp_path / "complete-action.jsonl"
    overlay_sha = _write_overlay(
        overlay,
        [_overlay_row(archive_sha=raw_sha, env_step=0, program="p0", successor=None)],
    )
    output = tmp_path / "cli-targets"
    assert target_main(
        [
            "--complete-action-overlay",
            str(overlay),
            "--raw-episode-zip",
            str(raw_zip),
            "--output-root",
            str(output),
            "--utc-day",
            DAY,
            "--split",
            "train",
            "--goal-contract",
            str(contract),
            "--expected-goal-contract-sha256",
            contract_sha,
            "--expected-complete-action-overlay-sha256",
            overlay_sha,
            "--expected-raw-episode-zip-sha256",
            raw_sha,
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    assert Path(printed["output_root"]) == output.resolve()
    assert Path(printed["receipt_path"]).is_file()
