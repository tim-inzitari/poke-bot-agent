from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.eval_vs_baselines import (
    _parse_args,
    _resolve_our_decks,
)


def _args(
    *,
    deck_suite: str,
    our_deck: list[Path] | None = None,
    our_archetype: str | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        deck_suite=deck_suite,
        our_deck=list(our_deck or []),
        our_archetype=our_archetype,
    )


def test_core_ladder_gate_resolves_every_pinned_family() -> None:
    decks, contract = _resolve_our_decks(_args(deck_suite="core-ladder"))

    ids = [row["deck_id"] for row in decks]
    assert len(decks) == 17
    assert len(set(ids)) == len(ids)
    assert "alakazam" in ids
    assert "lucario" in ids
    assert all(len(row["cards"]) == 60 for row in decks)
    assert contract["suite"] == "core-ladder"
    assert contract["deck_agnostic"] is True
    assert len(contract["contract"]["representatives"]) == len(decks)


def test_explicit_decks_are_repeatable_and_conflict_with_core_suite(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    first.write_text("\n".join(str(i) for i in range(1, 61)) + "\n")
    second.write_text("\n".join(str(i) for i in range(61, 121)) + "\n")

    parsed = _parse_args(
        [
            "--checkpoint",
            str(tmp_path / "candidate.pt"),
            "--our-deck",
            str(first),
            "--our-deck",
            str(second),
        ]
    )
    decks, contract = _resolve_our_decks(parsed)
    assert [row["deck_id"] for row in decks] == ["first", "second"]
    assert contract == {"suite": "explicit", "deck_agnostic": True}

    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_our_decks(
            _args(deck_suite="core-ladder", our_deck=[first])
        )


def test_single_pinned_archetype_uses_exact_ladder_representative() -> None:
    decks, contract = _resolve_our_decks(
        _args(deck_suite="primary", our_archetype="AlAkAzAm")
    )

    assert [row["deck_id"] for row in decks] == ["alakazam"]
    assert len(decks[0]["cards"]) == 60
    assert decks[0]["source"] == "pinned_top_ladder_modal_representative"
    assert contract["suite"] == "pinned-archetype"
    assert contract["archetype"] == "alakazam"


def test_single_pinned_archetype_rejects_conflicting_explicit_deck(
    tmp_path: Path,
) -> None:
    deck = tmp_path / "deck.csv"
    deck.write_text("\n".join(str(i) for i in range(1, 61)) + "\n")
    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_our_decks(
            _args(
                deck_suite="primary",
                our_deck=[deck],
                our_archetype="alakazam",
            )
        )
