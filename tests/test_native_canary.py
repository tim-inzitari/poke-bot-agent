"""Opt-in native pre-launch test; never selected by the quick profile."""

from __future__ import annotations

import os

import pytest

from scripts.native_prelaunch_canary import main


pytestmark = [
    pytest.mark.native,
    pytest.mark.gpu,
    pytest.mark.integration,
    pytest.mark.slow,
]


def test_native_prelaunch_canary() -> None:
    if os.environ.get("POKEBOT_RUN_NATIVE_CANARY") != "1":
        pytest.skip("set POKEBOT_RUN_NATIVE_CANARY=1 or use test_canary.sh")
    assert main(["--mode", "canary", "--timeout-seconds", "1100"]) == 0
