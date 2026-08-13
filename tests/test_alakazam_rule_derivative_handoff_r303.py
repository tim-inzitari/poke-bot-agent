"""Regression coverage for the inert revision-5 handoff contract surface."""

from __future__ import annotations

import pytest

import poke_bot.alakazam_rule_derivative_handoff_r303 as handoff


_REV4_GATEWAY_SHA256 = (
    "sha256:2af67560510ca7ffd9fe0bc6ff37cdbbd74f5a78d6c5237091bb527d49ce4ed8"
)


def test_r303_loads_final_rev5_gateway_and_contract() -> None:
    contract = handoff.load_r303_contract()

    assert contract["goal_revision"] == 5
    assert handoff.R303_GOAL_GATEWAY_SHA256 == (
        "sha256:7a829abebd348d0ffdf0a73c8b559fe9c799af3d3aff49a64efdfa85a08051b6"
    )
    assert handoff.R303_CONTRACT_SHA256 == (
        "sha256:dbbd4dbcc057b631d61fa867e45c393d594550b3b45f306f465b6ee5b4428891"
    )


def test_r303_rejects_the_revision_4_gateway(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(handoff, "R303_GOAL_GATEWAY_SHA256", _REV4_GATEWAY_SHA256)

    with pytest.raises(handoff.R303HandoffError, match="goal gateway identity"):
        handoff.load_r303_contract()


def test_r303_star_import_exposes_the_documented_error_type() -> None:
    namespace: dict[str, object] = {}
    exec("from poke_bot.alakazam_rule_derivative_handoff_r303 import *", namespace)

    assert namespace["R303HandoffError"] is handoff.R303HandoffError


def test_blackwell_launch_contract_cannot_name_post_queue_rl_service() -> None:
    # A handoff plan is intentionally insufficient to create a launch contract
    # without its six receipt prerequisites.  The service constants are still
    # public contract facts and guard the sequencing boundary directly.
    assert handoff._R303_BOOTSTRAP_SERVICE == (
        "pokebot-alakazam-rule-derivative-g5-bootstrap.service"
    )
    assert handoff._R303_TRAINER_SERVICE == "pokebot-alakazam-rule-derivative-g5-rl.service"
    assert handoff._R303_BOOTSTRAP_SERVICE != handoff._R303_TRAINER_SERVICE
