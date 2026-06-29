"""Drift guard: the submission archetype-heuristic scoring must stay byte-for-byte
behaviorally identical to poke_agent's.

submission/archetype_heuristics.py is a hand-maintained mirror of the scoring logic
in poke_agent/archetype_heuristics.py (the submission package uses flat imports and a
trimmed opponent-prediction path, so it cannot import poke_agent directly). This test
fails the moment a scorer, classifier, or dispatch branch diverges between the two,
turning a silent inference regression into a loud failure. build_submission.sh runs it
before packaging so a drifted submission can never ship.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBMISSION = ROOT / "submission"

import poke_agent.archetype_heuristics as train


def _load_submission_heuristics():
    # submission/archetype_heuristics.py uses flat imports (rewards, archetype_signatures_data).
    if str(SUBMISSION) not in sys.path:
        sys.path.insert(0, str(SUBMISSION))
    spec = importlib.util.spec_from_file_location(
        "submission_archetype_heuristics", SUBMISSION / "archetype_heuristics.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sub = _load_submission_heuristics()


# Representative decks for each archetype (signature lines only — enough to classify).
DECKS = {
    train.ARCHETYPE_DRAGAPULT: [train.DREEPY, train.DREEPY, train.DRAKLOAK, train.DRAGAPULT_EX, train.DRAGAPULT_EX],
    train.ARCHETYPE_LUCARIO: [train.RIOLU, train.MEGA_LUCARIO_EX, train.MEGA_LUCARIO_EX, train.SOLROCK, train.LUNATONE],
    train.ARCHETYPE_ABOMASNOW: [train.SNOVER, train.SNOVER, train.MEGA_ABOMASNOW_EX, train.MEGA_ABOMASNOW_EX, train.KYOGRE],
    train.ARCHETYPE_IONO: [train.IONO_VOLTORB, train.IONO_TADBULB, train.IONO_BELLIBOLT_EX, train.IONO_BELLIBOLT_EX, train.IONO_WATTREL, train.IONO_WATTREL],
    train.ARCHETYPE_STARMIE: [train.STARYU, train.STARYU, train.STARMIE, train.STARMIE, train.STARMIE, train.STARMIE],
    train.ARCHETYPE_CRUSTLE: [train.MEGA_KANGASKHAN_EX, train.MEGA_KANGASKHAN_EX, train.CRUSTLE_CARD],
}

# Option type fan-out covering every dispatch branch the scorers care about.
OPTIONS = [
    {"type": train.OPTION_ATTACK, "attackId": train.MEGA_BRAVE_ATTACK},
    {"type": train.OPTION_ATTACK, "attackId": train.PHANTOM_DIVE_ATTACK},
    {"type": train.OPTION_ATTACK, "attackId": train.HAMMER_LANCHE_ATTACK},
    {"type": train.OPTION_ATTACK, "attackId": 999},
    {"type": train.OPTION_PLAY, "index": 0},
    {"type": train.OPTION_EVOLVE, "index": 0},
    {"type": train.OPTION_ABILITY},
    {"type": train.OPTION_ATTACH},
    {"type": train.OPTION_END},
]

PHASES = ("early", "mid", "late")
CONTEXTS = ("default", "mirror", "aggro")
OPPONENTS = (train.ARCHETYPE_UNKNOWN, train.ARCHETYPE_DRAGAPULT, train.ARCHETYPE_LUCARIO)


def _obs(options, hand):
    return {
        "current": {
            "turn": 5,
            "players": [
                {"prize": [0] * 4, "active": [{"id": 999, "hp": 1, "maxHp": 1}], "bench": [], "hand": hand},
                {"prize": [0] * 4, "active": [{"id": 500, "hp": 1, "maxHp": 2}], "bench": [{"id": 501, "hp": 1, "maxHp": 2}]},
            ],
        },
        "select": {"option": options},
    }


def test_classify_archetype_parity():
    for archetype, deck in DECKS.items():
        assert train.classify_archetype(deck) == sub.classify_archetype(deck) == archetype
    assert train.classify_archetype([1, 2, 3]) == sub.classify_archetype([1, 2, 3])


def test_starmie_variant_parity():
    variants = [
        [train.STARYU, train.STARMIE, train.STARMIE, train.SNORUNT, train.FROSLASS, train.FROSLASS],
        [train.STARYU, train.MEGA_STARMIE_EX, train.MEGA_STARMIE_EX, train.STARMIE],
        [train.STARYU, train.STARMIE, train.DUSKULL, train.DUSKULL],
    ]
    for deck in variants:
        assert train.starmie_variant(deck) == sub.starmie_variant(deck)


def test_score_action_parity_across_archetypes():
    hand = [{"id": train.MEGA_LUCARIO_EX}]
    mismatches = []
    for archetype, deck in DECKS.items():
        variant = train.starmie_variant(deck) if archetype == train.ARCHETYPE_STARMIE else ""
        for opt in OPTIONS:
            obs = _obs([opt], hand)
            for phase in PHASES:
                for context in CONTEXTS:
                    for opp in OPPONENTS:
                        t = train.score_action(
                            [0], obs, 0, archetype=archetype, phase=phase,
                            context=context, opponent_archetype=opp, starmie_variant_key=variant,
                        )
                        s = sub.score_action(
                            [0], obs, 0, archetype=archetype, phase=phase,
                            context=context, opponent_archetype=opp, starmie_variant_key=variant,
                        )
                        if t != pytest.approx(s):
                            mismatches.append((archetype, opt, phase, context, opp, t, s))
    assert not mismatches, f"submission/poke_agent scorer drift: {mismatches[:5]}"
