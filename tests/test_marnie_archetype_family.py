from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

import pytest
import torch

import poke_bot.archetype_family_activation as family_activation
from poke_bot.archetype_family_activation import (
    FamilyActivationError,
    boundary_decision,
    boundary_pause_hook,
    materialize_activation_ready_pause,
    sha256,
    validate_atomic_migration,
    validate_iteration9_upload_trigger,
    validate_package_deck_contract,
)
from poke_bot.archetype_family_study import (
    activation_gate,
    antithetic_spsa,
    compile_development_round,
    compile_post_activation_monitor,
    family_shadow_plan,
    selected_loss_vector,
    paired_panel_schedule,
    rollback_required,
    select_after_rounds,
    validate_same_parent_shadow,
)
from poke_bot.archetype_loss_contract import (
    ArchetypeLossContractError,
    MaskedObjective,
    guide_gradient_allowed,
    macro_list_loss,
    validate_loss_contract,
)
from poke_bot.specialist_archetype_family import (
    ArchetypeFamilyError,
    FamilyDeckMix,
    canonical_counts,
    cluster_variants,
    digest_json,
    family_probabilities,
    hamilton_quotas,
    multiset_digest,
    macro_replay_rows,
    ordered_digest,
    schedule_variants,
    singular_package_variant,
    split_clusters,
    swap_distance,
    validate_manifest,
)
from poke_bot.archetypes import classify_deck
from poke_bot.deck_pool import read_deck
from poke_bot.dataset import BootstrapDataset, GameSequence
from scripts.build_marnie_archetype_family import _observe_package_replay
from scripts.activate_marnie_archetype_family import activate
from scripts.run_marnie_archetype_family_study import (
    SHADOW_GUIDE_LOSS_WEIGHT,
    _cpu_only_panel_spawn_environment,
    _validated_shadow_guide_loss_weight,
)
from scripts.train_pure_rl import (
    _build_collect_jobs,
    _macro_resample_family_sequences,
)


def _cards(offset: int = 0) -> list[int]:
    # Marnie line plus unrelated positive card IDs. Three replacements create
    # a distinct >2-swap cluster while preserving classification.
    return [646, 647, 648] + [1000 + index + offset * 100 for index in range(57)]


def _manifest(cluster_count: int = 13) -> dict:
    rows = []
    for index in range(cluster_count):
        cards = _cards(index)
        rows.append(
            {
                "family_id": "marnie-s-grimmsnarl-ex",
                "variant_id": f"v{index}",
                "card_ids": cards,
                "card_counts": [[card, count] for card, count in canonical_counts(cards)],
                "ordered_digest": ordered_digest(cards),
                "multiset_digest": multiset_digest(cards),
                "provenance": {"source": f"episode-{index}"},
                "legality": {"legal": True},
                "classification": {"archetype_id": "marnie-s-grimmsnarl-ex"},
                "cluster_id": f"c{index}",
                "split": "train" if index < 7 else ("dev" if index < 10 else "locked"),
                "training_weight": 0.0,
                "capability_mask": {
                    "core_setup_continuity": True,
                    "resource_attack_readiness": True,
                    "long_horizon_prize_pressure": True,
                },
                "package": index == 0,
                "measurement": index == 0,
            }
        )
    payload = {
        "schema": "poke_bot.specialist_archetype_families/v1",
        "family_id": "marnie-s-grimmsnarl-ex",
        "variants": rows,
    }
    payload["artifact_sha256"] = digest_json(payload)
    return payload


def test_manifest_digests_dedup_clusters_and_split_leakage() -> None:
    payload = _manifest()
    assert validate_manifest(payload, require_activation_ready=True)
    assert swap_distance(payload["variants"][0]["card_ids"], payload["variants"][1]["card_ids"]) == 57
    assert len(set(cluster_variants(payload["variants"]).values())) == 13
    splits = split_clusters([f"c{i}" for i in range(13)], package_cluster_id="c0", seed="fixed")
    assert splits["c0"] == "train"
    assert set(splits.values()) == {"train", "dev", "locked"}

    duplicate = copy.deepcopy(payload)
    duplicate["variants"][1]["card_ids"] = list(duplicate["variants"][0]["card_ids"])
    duplicate["variants"][1]["card_counts"] = copy.deepcopy(duplicate["variants"][0]["card_counts"])
    duplicate["variants"][1]["ordered_digest"] = duplicate["variants"][0]["ordered_digest"]
    duplicate["variants"][1]["multiset_digest"] = duplicate["variants"][0]["multiset_digest"]
    duplicate.pop("artifact_sha256")
    duplicate["artifact_sha256"] = digest_json(duplicate)
    with pytest.raises(ArchetypeFamilyError, match="duplicate canonical"):
        validate_manifest(duplicate)


def test_repository_observed_tournament_catalog_has_required_cluster_floor() -> None:
    paths = sorted(Path("decks/competitive/the_rest").glob("*grimmsnarl-froslass.csv"))
    rows = []
    for path in paths:
        cards = [int(value) for value in read_deck(path)]
        assert len(cards) == 60
        assert classify_deck(cards) == "marnie-s-grimmsnarl-ex"
        rows.append({"variant_id": path.stem, "card_ids": cards, "multiset_digest": multiset_digest(cards)})
    assert len(rows) == 15
    assert len({row["multiset_digest"] for row in rows}) == 15
    assert len(set(cluster_variants(rows).values())) == 14


def test_package_variant_requires_checksum_exact_observed_replay(tmp_path: Path) -> None:
    cards = _cards()
    shard = tmp_path / "iter_00005.jsonl"
    shard.write_text(
        json.dumps(
            {
                "episode_id": "observed-package-game",
                "seat": 0,
                "archetype": "marnie-s-grimmsnarl-ex",
                "deck": cards,
                "source": "pure_rl",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = tmp_path / "collection.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "poke_bot.completed_collection/v1",
                "shard": {"path": str(shard), "sha256": sha256(shard)},
            }
        ),
        encoding="utf-8",
    )
    observed = _observe_package_replay(receipt, cards)
    assert observed["episode_id"] == "observed-package-game"
    assert observed["source_kind"] == "sealed_pure_rl_replay"
    assert observed["collection_receipt_sha256"] == sha256(receipt)

    shard.write_text(shard.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="digest mismatch"):
        _observe_package_replay(receipt, cards)


def test_deterministic_hamilton_seats_derangement_and_package_cap() -> None:
    payload = _manifest()
    probs = family_probabilities(payload)
    assert probs["v0"] == pytest.approx(0.20)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert sum(hamilton_quotas(1024, probs).values()) == 1024
    first = schedule_variants(payload, games=1024, checksum_seed="abc")
    second = schedule_variants(payload, games=1024, checksum_seed="abc")
    assert first == second
    assert all(row["variant_id"] != row["opponent_variant_id"] for row in first)
    package_count = sum(row["variant_id"] == "v0" for row in first)
    assert package_count in {204, 205}
    for variant in probs:
        seats = [row["seat"] for row in first if row["variant_id"] == variant]
        assert abs(seats.count(0) - seats.count(1)) <= 1


def test_family_collection_keeps_logical_archetype_and_binds_variants() -> None:
    payload = _manifest()
    mix = FamilyDeckMix.from_manifest(payload)
    variants = {
        str(row["variant_id"]): row
        for row in payload["variants"]
        if row["split"] == "train"
    }
    decks = [
        (variant_id, list(variants[variant_id]["card_ids"]))
        for variant_id in sorted(variants)
    ]
    self_jobs, public_jobs = _build_collect_jobs(
        n_games=64,
        ckpt=Path("/tmp/model.pt"),
        digest="sha256:" + "a" * 64,
        model_generation=10,
        decks=decks,
        specs=[],
        seed=99,
        game_timeout_s=30,
        mode="specialist",
        ladder_mix=mix,
        family_manifest=payload,
        iteration=10,
    )
    assert not public_jobs
    assert len(self_jobs) == 64
    for job in self_jobs:
        provenance = job["target_provenance"]
        assert job["archetype"] == "marnie-s-grimmsnarl-ex"
        assert job["opp_archetype"] == "marnie-s-grimmsnarl-ex"
        assert provenance["family_id"] == "marnie-s-grimmsnarl-ex"
        assert provenance["manifest_digest"] == payload["artifact_sha256"]
        assert provenance["variant_id"] in variants
        assert provenance["opponent_variant_id"] in variants
        assert provenance["variant_id"] != provenance["opponent_variant_id"]
        assert (
            provenance["deck_mix"]["scheduled_our_deck_id"]
            == provenance["variant_id"]
        )
        assert (
            provenance["deck_mix"]["scheduled_opp_deck_id"]
            == provenance["opponent_variant_id"]
        )


def test_family_macro_replay_is_deterministic_and_rejects_eval_split() -> None:
    payload = _manifest()
    train_rows = [row for row in payload["variants"] if row["split"] == "train"]
    sequences = []
    for index, row in enumerate(train_rows):
        provenance = (
            {}
            if row["package"]
            else {
                "family_id": payload["family_id"],
                "variant_id": row["variant_id"],
                "manifest_digest": payload["artifact_sha256"],
                "ordered_digest": row["ordered_digest"],
                "multiset_digest": row["multiset_digest"],
            }
        )
        sequences.append(
            GameSequence(
                episode_id=f"train-{index}",
                seat=index % 2,
                archetype=payload["family_id"],
                opp_archetype=payload["family_id"],
                deck=list(row["card_ids"]),
                value=1.0,
                decisions=[],
                target_provenance=provenance,
            )
        )
    sequences.append(copy.deepcopy(sequences[0]))
    dataset = BootstrapDataset(sequences=sequences)
    first, receipt = _macro_resample_family_sequences(
        dataset, manifest=payload, checksum_seed="fixed"
    )
    second, second_receipt = _macro_resample_family_sequences(
        dataset, manifest=payload, checksum_seed="fixed"
    )
    assert receipt == second_receipt
    assert [row.episode_id for row in first.sequences] == [
        row.episode_id for row in second.sequences
    ]
    assert receipt["legacy_package_sequences"] == 2
    assert receipt["selected_sequences"] == len(sequences)
    assert receipt["development_or_locked_rows"] == 0

    dev = payload["variants"][7]
    eval_sequence = GameSequence(
        episode_id="dev-leak",
        seat=0,
        archetype=payload["family_id"],
        opp_archetype=payload["family_id"],
        deck=list(dev["card_ids"]),
        value=0.0,
        decisions=[],
        target_provenance={
            "family_id": payload["family_id"],
            "variant_id": dev["variant_id"],
            "manifest_digest": payload["artifact_sha256"],
            "ordered_digest": dev["ordered_digest"],
            "multiset_digest": dev["multiset_digest"],
        },
    )
    with pytest.raises(ArchetypeFamilyError, match="development/locked"):
        _macro_resample_family_sequences(
            BootstrapDataset(sequences=[eval_sequence]),
            manifest=payload,
            checksum_seed="fixed",
        )


def test_loss_contract_masking_macro_average_and_guide_authority() -> None:
    contract = json.loads(
        open("config/archetype_loss_contracts/marnie-s-grimmsnarl-ex.v1.json", encoding="utf-8").read()
    )
    validate_loss_contract(contract)
    assert guide_gradient_allowed(contract, "action_q")
    assert not guide_gradient_allowed(contract, "available_but_unauthorized")
    bad = copy.deepcopy(contract)
    bad["residual_objectives"]["core_setup_continuity"]["weight"] = 0.051
    with pytest.raises(ArchetypeLossContractError):
        validate_loss_contract(bad)

    values = torch.tensor([1.0, 1.0, 3.0], requires_grad=True)
    loss = macro_list_loss(
        [MaskedObjective("x", values, torch.ones(3, dtype=torch.bool), 1.0, "cap")],
        variant_ids=["common", "common", "rare"],
        family_applicable=torch.ones(3, dtype=torch.bool),
        capabilities={"cap": torch.ones(3, dtype=torch.bool)},
    )
    assert loss.item() == pytest.approx(2.0)  # equal-list macro, not row mean
    loss.backward()
    assert torch.isfinite(values.grad).all()

    masked = torch.tensor([2.0], requires_grad=True)
    zero = macro_list_loss(
        [MaskedObjective("x", masked, torch.zeros(1, dtype=torch.bool), 1.0, "cap")],
        variant_ids=["one"],
        family_applicable=torch.ones(1, dtype=torch.bool),
        capabilities={"cap": torch.ones(1, dtype=torch.bool)},
    )
    zero.backward()
    assert zero.item() == 0.0 and masked.grad.item() == 0.0


def test_family_shadow_masks_unavailable_guide_without_changing_production() -> None:
    assert SHADOW_GUIDE_LOSS_WEIGHT == 0.0
    assert (
        _validated_shadow_guide_loss_weight({"shadow_guide_loss_weight": 0.0})
        == 0.0
    )
    with pytest.raises(RuntimeError, match="exactly masked"):
        _validated_shadow_guide_loss_weight({"shadow_guide_loss_weight": 0.05})
    with pytest.raises(RuntimeError, match="exactly masked"):
        _validated_shadow_guide_loss_weight({})


def test_boundary_defers_started_collection_and_atomic_allowlist() -> None:
    assert boundary_decision(trigger_valid=False, study_passed=True, committed_iteration=9, next_collection_started=False, already_paused_for_commit=False)["action"] == "continue_unchanged"
    assert boundary_decision(trigger_valid=True, study_passed=False, committed_iteration=9, next_collection_started=False, already_paused_for_commit=False) == {"action": "pause_for_required_evidence", "target_iteration": 10}
    assert boundary_decision(trigger_valid=True, study_passed=True, committed_iteration=9, next_collection_started=True, already_paused_for_commit=False) == {"action": "defer", "target_iteration": 11}
    assert boundary_decision(trigger_valid=True, study_passed=True, committed_iteration=10, next_collection_started=False, already_paused_for_commit=False)["action"] == "pause_for_atomic_activation"
    validate_atomic_migration(
        {"runtime_root": "/old"},
        {
            "runtime_root": "/new",
            "family_manifest": "m",
            "selected_loss_vector": "l",
        },
    )
    with pytest.raises(FamilyActivationError):
        validate_atomic_migration({}, {"family_manifest": "m", "selected_loss_vector": "l", "router_format": 7})


def test_passed_post_activation_monitor_is_reused_at_later_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    family_root = tmp_path / "family"
    family_root.mkdir()
    request = family_root / "activation-request.json"
    request.write_text("{}\n", encoding="utf-8")
    migration = family_root / "migration-receipt.json"
    migration.write_text("{}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    pause = run_dir / "family_activation" / "monitor_pause_after_iter_00010.json"
    pause.parent.mkdir(parents=True)
    pause.write_text('{"committed_iteration": 10}\n', encoding="utf-8")
    monitor = family_root / "post-activation-monitor.json"
    monitor.write_text(
        json.dumps(
            {
                "rollback_required": False,
                "pause_receipt": str(pause.resolve()),
                "pause_receipt_sha256": sha256(pause),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        family_activation, "validate_migration_receipt", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        family_activation,
        "validate_post_activation_monitor",
        lambda *args, **kwargs: {},
    )
    decision = family_activation._post_activation_boundary(
        request_path=request,
        run_dir=run_dir,
        committed_iteration=11,
        learner_digest="sha256:" + "a" * 64,
        next_collection_started=False,
    )
    assert decision == {
        "action": "continue_unchanged",
        "target_iteration": None,
        "reason": "post_activation_monitor_passed",
        "monitor_receipt": str(monitor),
    }


def test_replay_macro_weighting_legacy_admission_and_singular_package() -> None:
    payload = _manifest()
    package = singular_package_variant(payload)
    rows = []
    for variant in payload["variants"]:
        if variant["split"] != "train":
            continue
        rows.append({"variant_id": variant["variant_id"], "payload": variant["variant_id"]})
    rows.append({"multiset_digest": package["multiset_digest"], "payload": "legacy"})
    selected = macro_replay_rows(payload, rows, total=1000, checksum_seed="sealed")
    assert selected == macro_replay_rows(payload, rows, total=1000, checksum_seed="sealed")
    assert sum(row["variant_id"] == package["variant_id"] for row in selected) == 200
    assert all(row["family_id"] == payload["family_id"] for row in selected)
    with pytest.raises(ArchetypeFamilyError, match="development/locked"):
        macro_replay_rows(payload, [{"variant_id": "v8"}], total=1, checksum_seed="bad")


def test_explicit_five_way_masks_are_permutation_invariant() -> None:
    values = torch.tensor([1.0, 8.0, 3.0], requires_grad=True)
    objective = MaskedObjective(
        "masked", values, torch.ones(3, dtype=torch.bool), 1.0, "cap",
        row_applicable=torch.tensor([True, False, True]),
        target_observable=torch.tensor([True, True, True]),
        label_valid=torch.tensor([True, True, True]),
    )
    loss = macro_list_loss(
        [objective], variant_ids=["a", "a", "b"],
        family_applicable=torch.ones(3, dtype=torch.bool),
        capabilities={"cap": torch.ones(3, dtype=torch.bool)},
    )
    assert loss.item() == pytest.approx(2.0)
    perm = torch.tensor([2, 0, 1])
    permuted = MaskedObjective(
        "masked", values[perm], torch.ones(3, dtype=torch.bool), 1.0, "cap",
        row_applicable=torch.tensor([True, True, False]),
        target_observable=torch.ones(3, dtype=torch.bool),
        label_valid=torch.ones(3, dtype=torch.bool),
    )
    assert macro_list_loss(
        [permuted], variant_ids=["b", "a", "a"],
        family_applicable=torch.ones(3, dtype=torch.bool),
        capabilities={"cap": torch.ones(3, dtype=torch.bool)},
    ).item() == pytest.approx(loss.item())


def test_spsa_same_parent_panels_full_gate_and_rollback() -> None:
    first = antithetic_spsa({"residual": 0.0125}, round_index=1, seed_book="one")
    second = antithetic_spsa({"residual": 0.0125}, round_index=2, seed_book="two")
    assert first["magnitude"] == 0.20 and second["magnitude"] == 0.10
    with pytest.raises(Exception, match="exactly two"):
        antithetic_spsa({"residual": 0.0125}, round_index=3, seed_book="three")
    common = {
        "parent_checkpoint_sha256": "sha256:p", "sealed_rows_sha256": "sha256:r",
        "split_sha256": "sha256:s", "batch_order_sha256": "sha256:b",
        "optimizer_settings_sha256": "sha256:o", "seed_book_sha256": "sha256:k",
        "update_count": 100, "served": False, "promoted": False, "replay_eligible": False,
    }
    validate_same_parent_shadow({"plus": {**common, "loss_vector_sha256": "sha256:+"}, "minus": {**common, "loss_vector_sha256": "sha256:-"}})
    assert len(paired_panel_schedule(["a", "b", "c"], [str(i) for i in range(17)], pairs_per_cell=20, seed="dev")) == 1020
    metrics = {
        "macro_win_rate_improvement_lb95": .01, "current_package_delta_lb95": -.01,
        "cvar20_delta_lb95": -.01, "all_list_delta_lb90_ge_minus_003": True,
        "invalid_crash_increase": .001, "complete_required_label_coverage": True,
        "finite_gradients": True, "auxiliary_to_core_gradient_norm": .5,
        "total_core_gradient_cosine": .8, "mean_policy_kl": .02, "p99_policy_kl": .1,
        "greedy_action_flip_rate": .05, "development_paired_units": 1020,
        "locked_paired_units": 4284, "package_guard_pairs": 1020, "replay_eligible": False,
    }
    assert activation_gate(metrics)["passed"] is True
    metrics["mean_policy_kl"] = .021
    assert activation_gate(metrics)["passed"] is False
    assert select_after_rounds([{"round": 1, "status": "inconclusive"}, {"round": 2, "status": "failed"}])["activate"] is False
    assert rollback_required({"probability_regression_worse_than_002": .99, "current_package_delta_lb95": 0, "invalid_game_check": True, "causal_integrity_check": True, "latency_check": True})


def test_exact_family_shadow_panels_and_monitor_contract() -> None:
    plan = family_shadow_plan(
        _manifest(), [f"opponent-{index}" for index in range(17)], seed_book="fixed"
    )
    assert plan["panel_units"] == {
        "development": 1020,
        "locked": 4284,
        "package": 1020,
    }
    assert len(plan["selected_cluster_ids"]["development"]) == 3
    assert len(plan["selected_cluster_ids"]["locked"]) == 3

    def treatment_rows(panel, left, right, left_score, right_score):
        rows = []
        for schedule in panel:
            for treatment, score, latency in (
                (left, left_score, 0.010),
                (right, right_score, 0.011),
            ):
                rows.append(
                    {
                        **schedule,
                        "treatment": treatment,
                        "score": score,
                        "invalid": False,
                        "decision_latency_seconds": latency,
                        "causal_integrity": True,
                        "training_eligible": False,
                        "replay_eligible": False,
                    }
                )
        return rows

    development = treatment_rows(
        plan["panels"]["development"], "plus", "minus", 1.0, 0.0
    )
    assert compile_development_round(development, round_index=1)[
        "selected_direction"
    ] == "plus"
    locked = treatment_rows(
        plan["panels"]["locked"], "candidate", "parent", 1.0, 0.0
    )
    package = treatment_rows(
        plan["panels"]["package"], "candidate", "parent", 1.0, 0.0
    )
    monitor = compile_post_activation_monitor(
        locked_rows=locked, package_rows=package
    )
    assert monitor["rollback_required"] is False
    assert monitor["locked_paired_units"] == 4284
    assert monitor["current_package_pairs"] == 1020


def test_future_package_switch_is_owner_exact_and_fail_closed() -> None:
    pending = json.load(open("config/package_deck_contracts/marnie-s-grimmsnarl-ex.pending.v1.json", encoding="utf-8"))
    result = validate_package_deck_contract(pending, legality=lambda _: True, classify=lambda _: "marnie-s-grimmsnarl-ex")
    assert result == {"status": "pending_owner_exact_list", "authorized": False}
    authorized = copy.deepcopy(pending)
    authorized["authorized_exact_card_ids"] = _cards()
    authorized["validation_evidence"] = {
        "engine_smoke": True, "exact_formal_evaluation": True,
        "paired_old_package_comparison": True, "clean_boundary": True,
    }
    assert validate_package_deck_contract(authorized, legality=lambda _: True, classify=lambda _: "marnie-s-grimmsnarl-ex")["authorized"] is True


def test_upload_trigger_and_boundary_pause_are_exact_and_idempotent(tmp_path) -> None:
    def write(name: str, value) -> Path:
        path = tmp_path / name
        if isinstance(value, (bytes, bytearray)):
            path.write_bytes(value)
        else:
            path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
        return path

    commit = write("commit.json", {"iteration": 9})
    checkpoint = write("iter_00009.pt", b"checkpoint")
    bundle = write("bundle.zip", b"bundle")
    deck = write("deck.csv", b"deck")
    uploaded = write("uploaded.zip", b"uploaded")
    auth_path = tmp_path / "auth.json"
    attempt_path = tmp_path / "attempt.json"
    auth = {
        "schema": "poke_bot.kaggle_submission_authorization/v1",
        "consumed_before_upload": True, "remaining_uses": 0,
        "submission_file_checksum": sha256(uploaded),
        "frozen_checkpoint_checksum": sha256(checkpoint),
    }
    auth_path.write_text(json.dumps(auth), encoding="utf-8")
    attempt = {
        "schema": "poke_bot.kaggle_submission_attempt/v1", "returncode": 0,
        "authorization_consumed": str(auth_path),
        "identity": {"file": str(uploaded), "file_sha256": sha256(uploaded), "competition": "comp", "message": "label"},
    }
    attempt_path.write_text(json.dumps(attempt), encoding="utf-8")
    bindings = {
        "commit": {"path": str(commit), "sha256": sha256(commit)},
        "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
        "package_bundle": {"path": str(bundle), "sha256": sha256(bundle)},
        "representative_deck": {"path": str(deck), "sha256": sha256(deck)},
        "competition": "comp", "submission_label": "label",
        "uploaded_file": {"path": str(uploaded), "sha256": sha256(uploaded)},
    }
    trigger = {
        "schema": "poke_bot.marnie_family_iteration9_upload_trigger/v1",
        "specialist_id": "marnie-s-grimmsnarl-ex", "iteration": 9, "bindings": bindings,
        "attempt": {"path": str(attempt_path), "sha256": sha256(attempt_path)},
        "consumed_authorization": {"path": str(auth_path), "sha256": sha256(auth_path)},
    }
    assert validate_iteration9_upload_trigger(trigger)["valid"]
    trigger_path = write("trigger.json", trigger)
    pretrigger = boundary_pause_hook(
        request_path=None, trigger_path=tmp_path / "future-trigger.json",
        run_dir=tmp_path / "pretrigger-run", committed_iteration=8,
        learner_digest="sha256:learner", next_collection_started=False,
    )
    assert pretrigger["action"] == "continue_unchanged"
    learner_digest = sha256(checkpoint)
    missing = boundary_pause_hook(
        request_path=None, trigger_path=trigger_path,
        run_dir=tmp_path / "missing-run", committed_iteration=9,
        learner_digest=learner_digest, next_collection_started=False,
    )
    assert missing["action"] == "pause_for_required_evidence"
    missing_receipt = json.loads(Path(missing["pause_receipt"]).read_text(encoding="utf-8"))
    assert missing_receipt["owner_revision"] == 130
    assert missing_receipt["activation_evidence_complete"] is False
    assert missing_receipt["restart_prevent_status"] == 75
    assert boundary_pause_hook(
        request_path=None, trigger_path=trigger_path,
        run_dir=tmp_path / "missing-run", committed_iteration=9,
        learner_digest=learner_digest, next_collection_started=False,
    )["action"] == "already_paused"
    manifest_path = write("manifest.json", _manifest())
    loss_path = Path("config/archetype_loss_contracts/marnie-s-grimmsnarl-ex.v1.json").resolve()
    passing_metrics = {
        "macro_win_rate_improvement_lb95": .01,
        "current_package_delta_lb95": -.01,
        "cvar20_delta_lb95": -.01,
        "all_list_delta_lb90_ge_minus_003": True,
        "invalid_crash_increase": .001,
        "complete_required_label_coverage": True,
        "finite_gradients": True,
        "auxiliary_to_core_gradient_norm": .5,
        "total_core_gradient_cosine": .8,
        "mean_policy_kl": .02,
        "p99_policy_kl": .1,
        "greedy_action_flip_rate": .05,
        "development_paired_units": 1020,
        "locked_paired_units": 4284,
        "package_guard_pairs": 1020,
        "replay_eligible": False,
    }
    passing_gate = activation_gate(passing_metrics)
    study_path = write(
        "study.json",
        {
            "schema": "poke_bot.marnie_archetype_family_shadow_study/v1",
            "passed": True,
            "training_eligible": False,
            "replay_eligible": False,
            "selection": {"activate": True, "round": 1, "direction": "plus"},
            "activation_gate": passing_gate,
            "same_parent_validation": {"valid": True},
        },
    )
    registry = write("registry.json", {})
    candidate_registry = write(
        "candidate-registry.json",
        {"family_manifest": "manifest", "selected_loss_vector": "vector"},
    )
    selector = write("selector.env", b"selector")
    vector = write(
        "vector.json",
        selected_loss_vector(
            weights={
                "core_setup_continuity": 0.0125,
                "resource_attack_readiness": 0.0125,
                "long_horizon_prize_pressure": 0.0125,
            },
            manifest_sha256=sha256(manifest_path),
            loss_contract_sha256=sha256(loss_path),
            study_sha256=sha256(study_path),
            selected_round=1,
            selected_direction="plus",
        ),
    )
    request = {
        "schema": "poke_bot.marnie_family_activation_request/v1",
        "trigger": {"path": str(trigger_path), "sha256": sha256(trigger_path)},
        "manifest": {"path": str(manifest_path), "sha256": sha256(manifest_path)},
        "loss_contract": {"path": str(loss_path), "sha256": sha256(loss_path)},
        "study": {"path": str(study_path), "sha256": sha256(study_path)},
        "bindings": {
            "learner_sha256": learner_digest,
            "checkpoint": {"path": str(checkpoint), "sha256": sha256(checkpoint)},
            "registry": {"path": str(registry), "sha256": sha256(registry)},
            "selector": {"path": str(selector), "sha256": sha256(selector)},
            "selected_loss_vector": {"path": str(vector), "sha256": sha256(vector)},
            "candidate_registry": {
                "path": str(candidate_registry),
                "sha256": sha256(candidate_registry),
            },
        },
        "sealed_pre_activation": {
            name: "sha256:" + name
            for name in (
                "registry", "selector", "learner", "checkpoint", "optimizer",
                "scaler", "rng", "design_contract", "manifest", "loss_vector",
            )
        },
    }
    request_path = write("request.json", request)
    first = boundary_pause_hook(
        request_path=request_path, run_dir=tmp_path / "run", committed_iteration=9,
        learner_digest=learner_digest, next_collection_started=False,
    )
    assert first["action"] == "pause_for_atomic_activation"
    second = boundary_pause_hook(
        request_path=request_path, run_dir=tmp_path / "run", committed_iteration=9,
        learner_digest=learner_digest, next_collection_started=False,
    )
    assert second["action"] == "already_paused"
    failed_study = copy.deepcopy(request)
    failed_study_path = write(
        "failed-study.json",
        {"schema": "poke_bot.marnie_archetype_family_shadow_study/v1", "passed": False},
    )
    failed_study["study"] = {
        "path": str(failed_study_path),
        "sha256": sha256(failed_study_path),
    }
    failed_request_path = write("failed-request.json", failed_study)
    failed = boundary_pause_hook(
        request_path=failed_request_path, trigger_path=trigger_path,
        run_dir=tmp_path / "failed-run", committed_iteration=9,
        learner_digest=learner_digest, next_collection_started=False,
    )
    assert failed["action"] == "pause_for_required_evidence"
    assert "shadow study" in json.loads(
        Path(failed["pause_receipt"]).read_text(encoding="utf-8")
    )["pause_reason"]
    bad = copy.deepcopy(trigger)
    bad["bindings"]["submission_label"] = "other"
    with pytest.raises(FamilyActivationError, match="label"):
        validate_iteration9_upload_trigger(bad)
    invalid_trigger_path = write("invalid-trigger.json", bad)
    invalid_boundary = boundary_pause_hook(
        request_path=None, trigger_path=invalid_trigger_path,
        run_dir=tmp_path / "invalid-trigger-run", committed_iteration=9,
        learner_digest=learner_digest, next_collection_started=False,
    )
    assert invalid_boundary["action"] == "pause_for_required_evidence"
    assert json.loads(
        Path(invalid_boundary["pause_receipt"]).read_text(encoding="utf-8")
    )["next_collection_started"] is False


def test_trainer_watches_trigger_even_when_activation_request_is_missing() -> None:
    source = Path("scripts/train_pure_rl.py").read_text(encoding="utf-8")
    assert "POKEBOT_MARNIE_ITERATION9_UPLOAD_TRIGGER" in source
    assert '"pause_for_required_evidence"' in source
    assert "raise SystemExit(75)" in source


def test_panel_spawn_environment_hides_cuda_and_restores_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("POKEBOT_WORKER_CPU_ONLY", "parent-value")

    with _cpu_only_panel_spawn_environment():
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert os.environ["POKEBOT_WORKER_CPU_ONLY"] == "1"

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["POKEBOT_WORKER_CPU_ONLY"] == "parent-value"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    monkeypatch.delenv("POKEBOT_WORKER_CPU_ONLY")
    with _cpu_only_panel_spawn_environment():
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert os.environ["POKEBOT_WORKER_CPU_ONLY"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert "POKEBOT_WORKER_CPU_ONLY" not in os.environ
