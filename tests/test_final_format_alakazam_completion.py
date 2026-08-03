import pytest

from scripts.complete_final_format_alakazam_refresh import (
    _completion_disposition,
    _handler_is_terminal_and_queued,
)


def test_measured_pass_disposition() -> None:
    result = _completion_disposition(
        {"completion_authority": "measured_both_gates_pass"}
    )
    assert result["status"] == "passed_frozen_registered"
    assert result["current_gate_pass"] is True
    assert result["measured_gate_pass"] is True


def test_owner_ceiling_is_not_mislabeled_as_pass() -> None:
    result = _completion_disposition(
        {"completion_authority": "explicit_owner_ceiling_acceptance"}
    )
    assert result["status"] == "ceiling_accepted_frozen_registered"
    assert result["current_gate_pass"] is False
    assert result["measured_gate_pass"] is False
    assert result["failed_gate_results_preserved"] is True


def test_missing_or_unknown_completion_authority_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="authority is absent or invalid"):
        _completion_disposition({})
    with pytest.raises(RuntimeError, match="authority is absent or invalid"):
        _completion_disposition({"completion_authority": "mystery"})


@pytest.mark.parametrize("phase", ["submissions_queued", "complete_handoff_started"])
def test_terminal_queue_accepts_pre_and_post_handoff_phase(phase: str) -> None:
    handler = {
        "schema": "poke_bot.passed_gate_handler/v1",
        "phase": phase,
        "submission_mode": "queue_and_continue",
        "queued_submissions": [{"copy_number": 1}],
        "handoff_started": phase == "complete_handoff_started",
    }
    assert _handler_is_terminal_and_queued(handler) is True


def test_post_handoff_phase_requires_started_receipt() -> None:
    handler = {
        "schema": "poke_bot.passed_gate_handler/v1",
        "phase": "complete_handoff_started",
        "submission_mode": "queue_and_continue",
        "queued_submissions": [{"copy_number": 1}],
        "handoff_started": False,
    }
    assert _handler_is_terminal_and_queued(handler) is False
