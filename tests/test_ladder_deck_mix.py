from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest

from poke_bot.ladder_deck_mix import (
    LadderDeckMixError,
    canonical_payload_digest,
    largest_remainder_quotas,
    load_ladder_deck_mix,
    load_ladder_deck_representatives,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "data" / "training_mixes" / "top_ladder.v1.json"
EXPECTED_DIGEST = (
    "sha256:2c570b3ae2cb4b5e15254596bf1eb128eb65ae77d34f3731c020c7039187ce71"
)
REPRESENTATIVES = (
    ROOT / "data" / "training_mixes" / "top_ladder_representatives.v1.json"
)
EXPECTED_REPRESENTATIVE_DIGEST = (
    "sha256:23314d3090f5f0b40bbb651fdc8171dd0dceecebe950f91c0ba0eafda613bcef"
)


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _write_with_fresh_digest(path: Path, payload: dict) -> None:
    payload["artifact_sha256"] = canonical_payload_digest(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _catalog_for(mix) -> dict[str, list[int]]:
    catalog: dict[str, list[int]] = {}
    for bucket in mix.decks:
        cards = [group[0] for group in bucket.signature_groups]
        cards.extend([9000 + bucket.source_rank] * (60 - len(cards)))
        catalog[bucket.deck_id] = cards
    return catalog


def test_official_artifact_preserves_counts_weights_and_provenance() -> None:
    mix = load_ladder_deck_mix(ARTIFACT)

    assert mix.artifact_sha256 == EXPECTED_DIGEST
    assert mix.source["dataset_slug"] == (
        "kaggle/pokemon-tcg-ai-battle-episodes-2026-07-12"
    )
    assert mix.source["episodes_processed"] == 5050
    assert mix.source["decisive_games"] == 5035
    assert mix.coverage["total_seat_appearances"] == 10070
    assert mix.coverage["recognized_seat_appearances"] == 9661
    assert mix.coverage["excluded_seat_appearances"] == 409
    assert len(mix.decks) == 17
    assert [(deck.deck_id, deck.observed_count) for deck in mix.decks[:6]] == [
        ("alakazam", 3595),
        ("crustle", 2347),
        ("marnie-s-grimmsnarl-ex", 946),
        ("garchomp", 567),
        ("cornerstone-ogerpon", 489),
        ("rockets-mewtwo", 451),
    ]
    assert sum(deck.observed_count for deck in mix.decks) == 9661
    assert sum(deck.train_weight for deck in mix.decks) == pytest.approx(1.0)


def test_artifact_digest_rejects_unpinned_edits(tmp_path: Path) -> None:
    payload = _payload()
    payload["decks"][0]["observed_count"] += 1
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(LadderDeckMixError, match="artifact digest mismatch"):
        load_ladder_deck_mix(path)


def test_derived_weight_validation_rejects_rehashed_bad_policy(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["decks"][0]["train_weight"] -= 0.01
    payload["decks"][1]["train_weight"] += 0.01
    path = tmp_path / "bad-weight.json"
    _write_with_fresh_digest(path, payload)

    with pytest.raises(LadderDeckMixError, match="training weight mismatch"):
        load_ladder_deck_mix(path)


def test_largest_remainder_is_exact_stable_and_within_one_game() -> None:
    weights = {"b": 0.25, "a": 0.25, "c": 0.5}
    assert largest_remainder_quotas(weights, 3) == {"b": 1, "a": 1, "c": 1}

    mix = load_ladder_deck_mix(ARTIFACT)
    quotas = mix.quotas(2048)
    weights = mix.weights()
    assert sum(quotas.values()) == 2048
    assert quotas["alakazam"] == 698
    assert quotas["crustle"] == 460
    for deck_id, count in quotas.items():
        assert abs(count - 2048 * weights[deck_id]) < 1.0


def test_schedule_is_deterministic_and_retains_exact_quotas() -> None:
    mix = load_ladder_deck_mix(ARTIFACT)

    first = mix.schedule_ids(2048, seed=17, iteration=4, stream="self_play_our")
    again = mix.schedule_ids(2048, seed=17, iteration=4, stream="self_play_our")
    next_iteration = mix.schedule_ids(
        2048, seed=17, iteration=5, stream="self_play_our"
    )

    assert first == again
    assert first != next_iteration
    assert Counter(first) == Counter(mix.quotas(2048))


def test_observed_basis_and_schedule_provenance() -> None:
    mix = load_ladder_deck_mix(ARTIFACT)
    quotas = mix.quotas(1000, basis="observed")
    provenance = mix.schedule_provenance(
        1000,
        seed=9,
        iteration=2,
        stream="public_our",
        basis="observed",
    )

    assert sum(quotas.values()) == 1000
    assert quotas["alakazam"] == 372
    assert provenance["artifact_sha256"] == EXPECTED_DIGEST
    assert provenance["quotas"] == quotas
    assert provenance["stream"] == "public_our"


def test_catalog_binding_requires_every_60_card_representative() -> None:
    mix = load_ladder_deck_mix(ARTIFACT)
    catalog = _catalog_for(mix)

    bound = mix.bind_catalog(catalog)

    assert len(bound) == 17
    assert all(len(deck.card_ids) == 60 for deck in bound)
    assert all(deck.canonical_multiset_sha256.startswith("sha256:") for deck in bound)

    catalog.pop("alakazam")
    with pytest.raises(LadderDeckMixError, match="missing representative"):
        mix.bind_catalog(catalog)


def test_catalog_binding_checks_shape_and_signature() -> None:
    mix = load_ladder_deck_mix(ARTIFACT)
    catalog = _catalog_for(mix)
    catalog["hammer-pult"] = [9999] * 60
    with pytest.raises(LadderDeckMixError, match="misses signature group"):
        mix.bind_catalog(catalog)

    catalog = _catalog_for(mix)
    catalog["crustle"] = catalog["crustle"][:-1]
    with pytest.raises(LadderDeckMixError, match="has 59 cards"):
        mix.bind_catalog(catalog)


def test_modal_representatives_bind_exactly_to_the_source_mix() -> None:
    mix = load_ladder_deck_mix(ARTIFACT)
    representatives = load_ladder_deck_representatives(REPRESENTATIVES)

    assert representatives.artifact_sha256 == EXPECTED_REPRESENTATIVE_DIGEST
    bound = representatives.bind(mix)
    assert [entry.bucket.deck_id for entry in bound] == [
        entry.deck_id for entry in mix.decks
    ]
    assert all(len(entry.card_ids) == 60 for entry in bound)
    contract = representatives.contract(mix)
    assert contract["mix_artifact_sha256"] == EXPECTED_DIGEST
    assert contract["representatives_artifact_sha256"] == (
        EXPECTED_REPRESENTATIVE_DIGEST
    )
    assert contract["representatives"][0]["modal_seat_count"] == 1635


def test_ladder_collection_schedule_uses_exact_quotas_and_cross_family_games() -> None:
    from scripts import train_pure_rl

    mix = load_ladder_deck_mix(ARTIFACT)
    representatives = load_ladder_deck_representatives(REPRESENTATIVES)
    decks = [
        (entry.bucket.deck_id, list(entry.card_ids))
        for entry in representatives.bind(mix)
    ]
    jobs, public = train_pure_rl._build_collect_jobs(
        n_games=128,
        ckpt=Path("/tmp/champion.pt"),
        digest="sha256:test",
        model_generation=1,
        decks=decks,
        specs=[],
        seed=23,
        game_timeout_s=10,
        mode="core",
        self_play_frac=1.0,
        ladder_mix=mix,
        iteration=4,
    )

    assert not public
    assert len(jobs) == 128
    assert Counter(job["archetype"] for job in jobs) == Counter(mix.quotas(128))
    assert Counter(job["opp_archetype"] for job in jobs) == Counter(
        mix.quotas(128)
    )
    assert all(job["archetype"] != job["opp_archetype"] for job in jobs)
    assert all(
        job["target_provenance"]["deck_mix"]["artifact_sha256"]
        == EXPECTED_DIGEST
        for job in jobs
    )


def test_iteration_one_production_quotas_have_an_exact_derangement() -> None:
    """Regression for the unattended iteration-1 restart loop."""
    from scripts import train_pure_rl

    mix = load_ladder_deck_mix(ARTIFACT)
    n_self = round(2048 * 0.85)
    # Production passes ``seed=args.seed + it * 100_000`` into collection.
    ours = mix.schedule_ids(
        n_self,
        seed=100_000,
        iteration=1,
        stream="self_play_our",
    )
    raw_opponents = mix.schedule_ids(
        n_self,
        seed=100_000,
        iteration=1,
        stream="self_play_opp",
    )

    opponents = train_pure_rl._derange_deck_schedule(ours, raw_opponents)

    assert opponents == train_pure_rl._derange_deck_schedule(ours, raw_opponents)
    assert Counter(opponents) == Counter(raw_opponents) == Counter(mix.quotas(n_self))
    assert all(our != opponent for our, opponent in zip(ours, opponents))


def test_derangement_has_minimum_mirrors_when_quota_is_infeasible() -> None:
    from scripts import train_pure_rl

    ours = ("a", "b", "a", "a", "a")
    raw_opponents = ("a", "a", "b", "a", "a")

    opponents = train_pure_rl._derange_deck_schedule(ours, raw_opponents)

    assert Counter(opponents) == Counter(raw_opponents)
    # Four of five tokens are family a, so 2 * 4 - 5 = 3 mirrors are
    # mathematically unavoidable.
    assert sum(our == opponent for our, opponent in zip(ours, opponents)) == 3


def test_measurement_decks_are_a_strict_ordered_subset() -> None:
    from scripts import train_pure_rl

    decks, _mix, _representatives, _contract = train_pure_rl._core_ladder_decks()
    selected = train_pure_rl._select_measurement_decks(
        decks, "lucario,alakazam,starmie,crustle"
    )

    assert [name for name, _cards in selected] == [
        "lucario",
        "alakazam",
        "starmie",
        "crustle",
    ]
    assert all(len(cards) == 60 for _name, cards in selected)


def test_measurement_decks_fail_closed_on_unknown_or_duplicate_ids() -> None:
    from scripts import train_pure_rl

    decks = [("lucario", list(range(60))), ("alakazam", list(range(60, 120)))]
    with pytest.raises(ValueError, match="outside the active training pool"):
        train_pure_rl._select_measurement_decks(decks, "lucario,missing")
    with pytest.raises(ValueError, match="duplicate"):
        train_pure_rl._select_measurement_decks(decks, "lucario,lucario")
