from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poke_bot import archaludon_ex_heuristics as guide
from poke_bot import deck_guides


ROOT = Path(__file__).resolve().parents[1]
REPRESENTATIVE = json.loads(
    (
        ROOT
        / "data"
        / "training_mixes"
        / "specialist_representatives.v1.json"
    ).read_text(encoding="utf-8")
)["decks"]["archaludon-ex"]["card_ids"]


def _opening_obs() -> dict:
    return {
        "current": {
            "yourIndex": 0,
            "players": [
                {
                    "active": [],
                    "bench": [],
                    "hand": [{"id": guide.DURALUDON_SCR}, {"id": guide.DUNSPARCE_JTG}],
                    "deck": [{"id": guide.BASIC_METAL_ENERGY}] * 20,
                    "discard": [],
                    "prize": [{}, {}, {}, {}, {}, {}],
                },
                {
                    "active": [],
                    "bench": [],
                    "hand": [],
                    "deck": [],
                    "discard": [],
                    "prize": [{}, {}, {}, {}, {}, {}],
                },
            ],
        },
        "select": {
            "context": guide.CTX_SETUP_ACTIVE,
            "minCount": 1,
            "maxCount": 1,
            "option": [
                {"type": guide.OPT_CARD, "area": 2, "index": 0},
                {"type": guide.OPT_CARD, "area": 2, "index": 1},
            ],
        },
    }


def test_archaludon_guide_is_sparse_causal_and_complete_stage_only() -> None:
    obs = _opening_obs()
    combos = [(0,), (1,)]

    audit = guide.guide_audit(
        obs,
        combos,
        deck=REPRESENTATIVE,
        force_enabled=True,
    )

    assert audit is not None
    assert audit["specialist_id"] == "archaludon-ex"
    assert audit["runtime_authority"] == "none"
    assert audit["scores"][0] > audit["scores"][1]
    assert guide.guide_scores(
        obs,
        [(0,)],
        deck=REPRESENTATIVE,
        force_enabled=True,
    ) is None


def test_archaludon_guide_masks_wrong_deck_and_has_exact_runtime_bypass() -> None:
    obs = _opening_obs()
    combos = [(0,), (1,)]

    assert guide.guide_scores(
        obs,
        combos,
        deck=[guide.BASIC_METAL_ENERGY] * 60,
        force_enabled=True,
    ) is None
    assert guide.prior_logit_bias(obs, combos, scale=999.0) == [0.0, 0.0]


def test_generic_registry_dispatches_archaludon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "archaludon-ex")
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE_TARGETS", "1")

    assert deck_guides.enabled()
    assert deck_guides.guide_version() == guide.GUIDE_VERSION
    assert "archaludon-ex" in deck_guides.supported_ids()
    assert deck_guides.guide_scores(
        _opening_obs(),
        [(0,), (1,)],
        deck=REPRESENTATIVE,
    ) is not None


def test_archaludon_expert_writeup_and_teacher_are_checksum_bound() -> None:
    writeup = ROOT / "docs/deck_guides/archaludon-ex-expert-brief.txt"
    teacher = ROOT / "poke_bot/archaludon_ex_heuristics.py"
    text = writeup.read_text(encoding="utf-8")

    assert len(text.split()) == 6_475
    assert len(set(part for part in text.split() if part.startswith("http"))) >= 27
    assert hashlib.sha256(writeup.read_bytes()).hexdigest() == (
        "a44a9d3cc0793ace0a1a7fe8a19dcb6bd8a49f92fd400a7a9c17d6d856b9dd01"
    )
    assert hashlib.sha256(teacher.read_bytes()).hexdigest() == (
        "4ec28e4f26c281f7488b20c31bbe17a6e4320cb54cb7dc5e1e4722e839aceca3"
    )
