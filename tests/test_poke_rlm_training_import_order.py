"""Regression coverage for the PokeRLM/RTP training import boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_poke_rlm_training_can_import_before_rtp_pipeline() -> None:
    """A fresh interpreter must not initialize RTP's pipeline during this import."""
    environment = os.environ.copy()
    source_path = str(_REPOSITORY_ROOT)
    existing_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_path
        else os.pathsep.join((source_path, existing_path))
    )
    script = (
        "import sys\n"
        "import poke_bot.poke_rlm.training as training\n"
        "from poke_bot.poke_rlm.training import (\n"
        "    PokeRLMTrainConfig,\n"
        "    train_poke_rlm_shadow,\n"
        ")\n"
        "assert training.PokeRLMTrainConfig is PokeRLMTrainConfig\n"
        "assert callable(train_poke_rlm_shadow)\n"
        "assert 'poke_bot.recursive_turn_planner.pipeline' not in sys.modules\n"
        "from poke_bot.recursive_turn_planner import (\n"
        "    ArchetypeRTPJob,\n"
        "    run_archetype_rtp_pipeline,\n"
        ")\n"
        "assert ArchetypeRTPJob.__name__ == 'ArchetypeRTPJob'\n"
        "assert callable(run_archetype_rtp_pipeline)\n"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
