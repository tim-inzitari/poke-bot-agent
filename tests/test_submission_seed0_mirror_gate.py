from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_submission_seed0_mirror import (
    MirrorValidationError,
    win_progress_signature,
)
from submission import main as submission_main


def _observation(*, turn: int, prize0: int = 6, hp0: int = 100) -> dict:
    return {
        "current": {
            "turn": turn,
            "result": -1,
            "yourIndex": turn % 2,
            "players": [
                {
                    "prize": [None] * prize0,
                    "active": [
                        {
                            "id": 743,
                            "hp": hp0,
                            "maxHp": 140,
                            "energies": [],
                            "energyCards": [],
                            "serial": 11,
                        }
                    ],
                    "bench": [],
                    "deckCount": turn % 3,
                    "hand": [{"id": 1 + turn}],
                    "discard": [{"id": 2 + turn}],
                },
                {
                    "prize": [None] * 6,
                    "active": [
                        {
                            "id": 741,
                            "hp": 70,
                            "maxHp": 70,
                            "energies": [],
                            "energyCards": [],
                            "serial": 71,
                        }
                    ],
                    "bench": [],
                },
            ],
        },
        "select": {
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "option": [{"type": 14}],
        },
    }


def test_recycling_zones_do_not_fake_win_progress() -> None:
    before = _observation(turn=20)
    after = _observation(turn=84)
    assert win_progress_signature(before) == win_progress_signature(after)


def test_prize_or_damage_change_is_real_progress() -> None:
    base = win_progress_signature(_observation(turn=20))
    assert base != win_progress_signature(_observation(turn=21, prize0=5))
    assert base != win_progress_signature(_observation(turn=21, hp0=90))


def test_builder_invokes_exact_seed0_mirror_gate() -> None:
    source = Path("scripts/build_submission.sh").read_text(encoding="utf-8")
    assert "validate_submission_seed0_mirror.py" in source
    assert "--seed 0" in source
    assert "--mirror-games 64" in source
    assert "--max-stagnant-turns 768" in source
    assert '"no_progress_escape_turns": 512' in source
    assert "64_exact_package_mirrors_all_terminal" in source
    assert "seed0_mirror_evidence_sha256" in source


def test_mirror_failure_type_is_hard_error() -> None:
    assert issubclass(MirrorValidationError, RuntimeError)
    with pytest.raises(MirrorValidationError):
        raise MirrorValidationError("timeout")


def test_package_escape_is_inert_until_512_then_immediately_ends() -> None:
    submission_main._reset_submission_no_progress_escape()
    submission_main._NO_PROGRESS_ESCAPE_TURNS = 512
    for turn in range(512):
        assert submission_main._submission_no_progress_escape(
            _observation(turn=turn)
        ) is None
    assert submission_main._submission_no_progress_escape(
        _observation(turn=512)
    ) == [0]
    # Once latched, package inference is bypassed at every later MAIN prompt.
    assert submission_main._submission_no_progress_escape(
        _observation(turn=513)
    ) == [0]


def test_package_escape_never_invents_end_outside_main() -> None:
    submission_main._reset_submission_no_progress_escape()
    submission_main._NO_PROGRESS_ESCAPE_TURNS = 512
    submission_main._NO_PROGRESS_ESCAPE_LATCHED = True
    observation = _observation(turn=999)
    observation["select"]["context"] = 3
    assert submission_main._submission_no_progress_escape(observation) is None
