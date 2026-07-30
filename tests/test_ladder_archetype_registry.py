from __future__ import annotations

import json
from pathlib import Path

import torch

from poke_bot import archetypes, config
from poke_bot.ladder_deck_mix import canonical_payload_digest
from poke_bot.ladder_replay import canonical_deck_sha256
from poke_bot.model import build_model
from scripts.train_privileged_belief_shards import _expand_aux_head


def test_every_pinned_core_ladder_deck_has_an_auxiliary_class() -> None:
    payload = json.loads(
        Path("data/training_mixes/top_ladder.v1.json").read_text(encoding="utf-8")
    )
    mix_ids = [str(row["deck_id"]) for row in payload["decks"]]
    assert mix_ids == list(archetypes.CORE_LADDER_ARCHETYPE_IDS)
    assert set(mix_ids).issubset(archetypes.archetype_ids())
    assert tuple(archetypes.archetype_ids()[:5]) == archetypes.LEGACY_AUX_ARCHETYPE_IDS
    assert archetypes.archetype_ids()[-6:] == [
        "dudunsparce",
        "hops-trevenant",
        "walrein",
        "thwackey",
        "team-rockets-spidops",
        "teal-mask-ogerpon-ex",
    ]


def test_post_snapshot_archetypes_are_additive_and_signature_distinct() -> None:
    assert archetypes.classify_deck([879] + [1] * 59) == "hops-trevenant"
    assert archetypes.classify_deck([943] + [1] * 59) == "walrein"
    assert archetypes.classify_deck([306] + [1] * 59) == "dudunsparce"
    assert (
        archetypes.classify_deck([306, 646, 647, 648] + [1] * 56)
        == "marnie-s-grimmsnarl-ex"
    )
    assert (
        archetypes.classify_deck([400] * 4 + [401] * 4 + [431] + [1] * 51)
        == "team-rockets-spidops"
    )
    # The shared Spidops engine must not steal Rocket's Mewtwo lists.
    assert (
        archetypes.classify_deck([400] * 4 + [401] * 4 + [431] * 2 + [1] * 50)
        == "unknown"
    )
    assert (
        archetypes.classify_deck(
            [89] * 4 + [90] * 4 + [93] * 4 + [1245] * 4 + [1] * 44
        )
        == "thwackey"
    )
    assert (
        archetypes.classify_deck(
            [89] * 2 + [90] * 2 + [93] * 2 + [1245] * 2 + [1] * 52
        )
        == "unknown"
    )
    assert (
        archetypes.classify_deck(
            [63] * 2
            + [96] * 3
            + [108]
            + [272]
            + [756] * 3
            + [1071] * 3
            + [1116] * 4
            + [1250] * 4
            + [1] * 39
        )
        == "teal-mask-ogerpon-ex"
    )
    # A generic Teal Mask engine is not enough to capture another Ogerpon
    # archetype; Slop Box requires its exact Raging Bolt/toolbox markers.
    assert (
        archetypes.classify_deck([96] * 4 + [1116] * 4 + [1] * 52)
        == "unknown"
    )
    # Dragapult+Dudunsparce remains its older, separate family.
    assert (
        archetypes.classify_deck([119, 306] + [1] * 58)
        == "dragapult-dudunsparce"
    )


def test_teal_mask_ogerpon_representative_is_exact_and_self_checksumming() -> None:
    path = Path("data/training_mixes/specialist_representatives.v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["decks"]["teal-mask-ogerpon-ex"]

    assert payload["artifact_sha256"] == canonical_payload_digest(payload)
    assert len(row["card_ids"]) == 60
    assert row["source_deck_id"] == "slop-box"
    assert row["competitive_family_alias"] == "raging-bolt-ogerpon"
    assert row["source_archetype_id"] == 151
    public = json.loads(
        Path(
            "data/training_mixes/"
            "teal-mask-ogerpon-ex-public-full32.v1.json"
        ).read_text(encoding="utf-8")
    )["source_deck_rows"][0]["card_ids"]
    assert row["card_ids"] == public
    assert archetypes.classify_deck(row["card_ids"]) == (
        "teal-mask-ogerpon-ex"
    )


def test_archaludon_representative_has_collision_safe_exact_identity() -> None:
    path = Path("data/training_mixes/specialist_representatives.v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = payload["decks"]["archaludon-ex"]

    assert payload["artifact_sha256"] == canonical_payload_digest(payload)
    assert len(row["card_ids"]) == 60
    assert tuple(row["card_ids"]) == (
        archetypes.ARCHALUDON_EX_MODAL_REPRESENTATIVE
    )
    assert row["source_deck_id"] == "archaludon-ex"
    assert row["distinct_lists"] == 6
    assert row["labeled_seat_count"] == 113
    assert row["modal_seat_count"] == 81
    assert row["canonical_multiset_sha256"] == (
        "sha256:43b3939c786067e0621e4acd7c3af958bfa97c501e45e62bc21ab1fa0b4bb1e3"
    )
    assert row["cards_sha256"] == (
        "sha256:01590bfd928dc5028af6516e365c55a693ad6a1ccb5426580d5d16e7127e5e34"
    )
    assert archetypes.classify_deck(row["card_ids"]) == "archaludon-ex"
    assert archetypes.classify_deck(reversed(row["card_ids"])) == (
        "archaludon-ex"
    )

    one_card_mutation = list(row["card_ids"])
    one_card_mutation[-1] = 1
    assert archetypes.classify_deck(one_card_mutation) != "archaludon-ex"


def test_teal_mask_public_full_history_catalog_is_exact_and_complete() -> None:
    payload = json.loads(
        Path(
            "data/training_mixes/"
            "teal-mask-ogerpon-ex-public-full32.v1.json"
        ).read_text(encoding="utf-8")
    )
    row = payload["source_deck_rows"][0]

    assert payload["schema"] == "poke_bot.public_deck_archetype_catalog/v1"
    assert payload["source_archetype"] == {
        "id": 151,
        "name": "Teal Mask Ogerpon ex",
    }
    assert payload["source_window"]["days"] == 32
    assert len(payload["observed_by_day"]) == 32
    assert sum(payload["observed_by_day"].values()) == 1_135
    assert payload["observed_acting_seat_games"] == 1_135
    assert len(row["card_ids"]) == 60
    assert canonical_deck_sha256(row["card_ids"]) == (
        payload["deck_fingerprints"][0]
    )
    assert archetypes.classify_deck(row["card_ids"]) == (
        "teal-mask-ogerpon-ex"
    )


def test_teal_mask_signature_rejects_all_1032_non_target_ingest_rows() -> None:
    audit = json.loads(
        Path(
            "data/training_mixes/"
            "teal-mask-ogerpon-ex-public-signature-audit.v1.json"
        ).read_text(encoding="utf-8")
    )
    signature_ids = [int(value) for value in audit["signature_card_ids"]]
    target_rows = 0
    non_target_rows = 0

    for row in audit["signature_count_groups"]:
        cards = [
            card_id
            for card_id, count in zip(signature_ids, row["counts"])
            for _ in range(int(count))
        ]
        cards.extend([1] * (60 - len(cards)))
        if int(row["target_rows"]):
            assert archetypes.is_teal_mask_ogerpon_box_signature(
                json.loads(
                    Path(
                        "data/training_mixes/"
                        "teal-mask-ogerpon-ex-public-full32.v1.json"
                    ).read_text(encoding="utf-8")
                )["source_deck_rows"][0]["card_ids"]
            )
            target_rows += int(row["target_rows"])
        else:
            assert not archetypes.is_teal_mask_ogerpon_box_signature(cards)
            non_target_rows += int(row["rows"])

    assert target_rows == audit["target_rows"] == 1
    assert non_target_rows == audit["non_target_rows"] == 1_032
    assert audit["source_deck_rows"] == target_rows + non_target_rows
    assert audit["source_deck_rows_sha256"] == (
        "sha256:6158acb521fd1316de990f3320a50e7dc9896928ec27fd1106ad54b66a0ddd05"
    )


def test_teal_mask_signature_rejects_both_public_mega_kangaskhan_rows() -> None:
    audit = json.loads(
        Path(
            "data/training_mixes/"
            "teal-mask-ogerpon-ex-public-signature-audit.v1.json"
        ).read_text(encoding="utf-8")
    )
    collisions = audit["mega_kangaskhan_collision_rows"]

    assert len(collisions) == 2
    for row in collisions:
        assert row["archetype_id"] == 67
        assert canonical_deck_sha256(row["card_ids"]) == (
            row["canonical_deck_sha256"]
        )
        assert not archetypes.is_teal_mask_ogerpon_box_signature(
            row["card_ids"]
        )
        assert archetypes.classify_deck(row["card_ids"]) != (
            "teal-mask-ogerpon-ex"
        )


def test_aux_expansion_preserves_legacy_rows_and_moves_unknown() -> None:
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=0,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        decision_context="stateless",
        dropout=0.0,
    )
    legacy_classes = len(archetypes.LEGACY_AUX_ARCHETYPE_IDS) + 1
    model = build_model(cfg, aux_archetype_classes=legacy_classes)
    old_weight = model.aux_head[-1].weight.detach().clone()
    old_bias = model.aux_head[-1].bias.detach().clone()
    assert _expand_aux_head(model)
    new = model.aux_head[-1]
    assert new.out_features == len(archetypes.archetype_ids()) + 1
    for old_i, name in enumerate(archetypes.LEGACY_AUX_ARCHETYPE_IDS):
        new_i = archetypes.archetype_ids().index(name)
        assert torch.equal(new.weight[new_i], old_weight[old_i])
        assert torch.equal(new.bias[new_i], old_bias[old_i])
    assert torch.equal(new.weight[-1], old_weight[-1])
    assert torch.equal(new.bias[-1], old_bias[-1])


def test_aux_expansion_preserves_current_core_rows_and_moves_unknown() -> None:
    cfg = config.ModelConfig(
        d_model=16,
        spatial_layers=1,
        temporal_layers=0,
        option_decoder_layers=1,
        n_heads=4,
        ff_dim=32,
        max_context=8,
        decision_context="stateless",
        dropout=0.0,
    )
    old_ids = list(archetypes.PINNED_CORE_AUX_ARCHETYPE_IDS)
    model = build_model(cfg, aux_archetype_classes=len(old_ids) + 1)
    old_weight = model.aux_head[-1].weight.detach().clone()
    old_bias = model.aux_head[-1].bias.detach().clone()
    assert _expand_aux_head(model)
    new = model.aux_head[-1]
    for old_i, name in enumerate(old_ids):
        new_i = archetypes.archetype_ids().index(name)
        assert torch.equal(new.weight[new_i], old_weight[old_i])
        assert torch.equal(new.bias[new_i], old_bias[old_i])
    assert torch.equal(new.weight[-1], old_weight[-1])
    assert torch.equal(new.bias[-1], old_bias[-1])
