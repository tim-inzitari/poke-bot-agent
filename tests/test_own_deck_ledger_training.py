"""Focused offline-only checks for the dormant r258 training path."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from poke_bot import dataset, features
from poke_bot import own_deck_successor as successor
from poke_bot import own_deck_training_contract as contract
from poke_bot.dataset import (
    DecisionSample,
    GameSequence,
    OwnDeckSidecarJoinError,
    PolicyStage,
)
from poke_bot.own_deck_ledger import OwnDeckLedger
from poke_bot.own_deck_rollout_store import (
    OWN_DECK_ROLLOUT_SIDECAR_SCHEMA,
    OWN_DECK_ROLLOUT_SIDECAR_VERSION,
)
from poke_bot.own_deck_supervision import (
    build_own_deck_supervision_targets,
    terminal_conversion_target_mask,
    terminal_conversion_target_vector,
    visible_tutor_completion_target_mask,
    visible_tutor_completion_target_vector,
)
from poke_bot.train import (
    _AwrValueGamePlan,
    _capture_value_only_awr_plans,
    _masked_own_deck_typed_option_loss,
    _own_deck_supervision_for_stage,
    _validate_own_deck_snapshot,
)


def _card(card_id: int, serial: int) -> dict[str, int]:
    return {"id": card_id, "serial": serial}


def _observation(*, known_prize: bool) -> dict:
    own = {
        "hand": [],
        "active": [],
        "bench": [],
        "discard": [],
        "prize": [_card(1, 10) if known_prize else None] + [None] * 5,
        "deckCount": 54,
    }
    return {
        "current": {
            "yourIndex": 0,
            "players": [own, {"hand": [], "deckCount": 54, "prize": [None] * 6}],
            "looking": [],
            "stadium": [],
        },
        "select": {"deck": [], "option": []},
    }


def _sv(words: int = 1) -> features.SparseVector:
    vector = features.SparseVector()
    for word in range(words):
        vector.word_start()
        vector.add(word, 1.0)
    return vector


def _snapshot() -> object:
    return OwnDeckLedger([1] * 60).observe(_observation(known_prize=False))


def test_convert_record_reconstructs_ledger_before_context_trim(monkeypatch) -> None:
    captured: list[object] = []

    def fake_featurize(
        step,
        _deck,
        *,
        verify_info_set=True,
        ledger_snapshot=None,
        own_deck_supervision=None,
    ) -> DecisionSample:
        del verify_info_set, own_deck_supervision
        captured.append(ledger_snapshot)
        return DecisionSample(
            board=_sv(features.NUM_BOARD_TOKENS),
            options=_sv(),
            action=[],
            action_combo_index=0,
            action_combos=[[]],
            env_step=int(step["env_step"]),
            ledger_snapshot=ledger_snapshot,
        )

    monkeypatch.setattr(dataset, "featurize_step", fake_featurize)
    monkeypatch.setattr(
        dataset,
        "build_own_deck_supervision_targets",
        lambda steps: [{} for _ in steps],
    )
    monkeypatch.setattr(
        "poke_bot.blackwell_heads.attach_blackwell_strategy_labels",
        lambda _steps: None,
    )
    monkeypatch.setattr(
        "poke_bot.strategic_heads.attach_expanded_strategic_labels",
        lambda _steps, **_kwargs: {"schema": "test"},
    )

    record = {
        "episode_id": "episode-1",
        "seat": 0,
        "deck": [1] * 60,
        "value": 0.0,
        "steps": [
            {"env_step": 3, "observation": _observation(known_prize=True)},
            {"env_step": 4, "observation": _observation(known_prize=False)},
        ],
    }
    sequence, reason, details = dataset.convert_record(
        record,
        max_context=1,
        verify_info_set=False,
        own_deck_ledger_enabled=True,
    )

    assert reason is None
    assert details["decisions_truncated"] == 1
    assert sequence is not None
    assert [decision.env_step for decision in sequence.decisions] == [4]
    snapshot = captured[0]
    assert snapshot is sequence.decisions[0].ledger_snapshot
    # The current record masks the prize, so this surviving fact can only come
    # from causal reconstruction over the full original match before trimming.
    assert dict(snapshot.known_prize_slots) == {0: 1}
    assert snapshot.revision == 2


def test_snapshot_validation_rejects_fail_closed_and_fingerprint_drift() -> None:
    snapshot = _snapshot()
    assert _validate_own_deck_snapshot(snapshot) is snapshot

    with pytest.raises(ValueError, match="integrity-ok"):
        _validate_own_deck_snapshot(replace(snapshot, integrity_ok=False))
    with pytest.raises(ValueError, match="fingerprint"):
        _validate_own_deck_snapshot(
            replace(snapshot, fingerprint="sha256:" + "0" * 64)
        )


def test_own_deck_loss_weights_are_neutral_by_default_and_bounded() -> None:
    logits = torch.zeros(1, 1, 6)
    selected = torch.tensor([0])
    # selected target: own win + prize closeout, with full typed masks.
    targets = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0]])
    masks = torch.ones_like(targets, dtype=torch.bool)
    neutral, rows = _masked_own_deck_typed_option_loss(
        logits,
        selected_indices=selected,
        targets=targets,
        masks=masks,
        categorical_slice=slice(0, 4),
        class_weights=(1.0, 1.0, 1.0, 1.0),
        positive_weight=1.0,
    )
    weighted, weighted_rows = _masked_own_deck_typed_option_loss(
        logits,
        selected_indices=selected,
        targets=targets,
        masks=masks,
        categorical_slice=slice(0, 4),
        class_weights=(1.0, 2.0, 1.0, 1.0),
        positive_weight=3.0,
    )

    assert rows == weighted_rows == 1
    assert float(weighted) > float(neutral)
    with pytest.raises(ValueError, match="32.0"):
        _masked_own_deck_typed_option_loss(
            logits,
            selected_indices=selected,
            targets=targets,
            masks=masks,
            categorical_slice=slice(0, 4),
            class_weights=(1.0, 33.0, 1.0, 1.0),
            positive_weight=1.0,
        )


class _LedgerAwareValueModel(torch.nn.Module):
    """Minimal value-only model that records the offline ledger residual path."""

    own_deck_ledger_enabled = True
    decision_context = "history"

    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.value_head = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.value_head.weight.fill_(1.0)
        self.seen_snapshots = None
        self.offline_training_path = None
        self.seen_history_residual = None

    def encode_board(self, boards):
        return torch.zeros(len(boards), 2, device=self.anchor.device)

    def own_deck_ledger_residuals(
        self,
        snapshots,
        *,
        batch_size,
        device,
        dtype,
        offline_training_path=False,
    ):
        self.seen_snapshots = list(snapshots or ())
        self.offline_training_path = offline_training_path
        return torch.ones(batch_size, 2, device=device, dtype=dtype)

    def history_tokens(self, spatial, previous_actions=None, ledger_residuals=None):
        del previous_actions
        self.seen_history_residual = ledger_residuals
        return spatial + ledger_residuals

    def temporal_encode(self, cls, **_kwargs):
        return cls, None


def test_value_only_awr_baseline_uses_offline_ledger_residuals() -> None:
    snapshot = _snapshot()
    decision = DecisionSample(
        board=_sv(features.NUM_BOARD_TOKENS),
        options=_sv(),
        action=[],
        action_combo_index=0,
        action_combos=[[]],
        env_step=7,
        action_token=None,
        ledger_snapshot=snapshot,
    )
    sequence = GameSequence(
        episode_id="episode-awr",
        seat=0,
        archetype="alakazam",
        opp_archetype="other",
        deck=[1] * 60,
        value=0.0,
        decisions=[decision],
    )
    plan = _AwrValueGamePlan(
        sequence=sequence,
        row_keys_by_decision=(((1, 0, 0),),),
    )
    model = _LedgerAwareValueModel()

    values = _capture_value_only_awr_plans(
        model,
        [plan],
        pack_temporal_games=False,
    )

    assert model.offline_training_path is True
    assert model.seen_snapshots == [snapshot]
    assert model.seen_history_residual is not None
    assert values[(1, 0, 0)] == pytest.approx(float(torch.tanh(torch.tensor(2.0))))


def test_multiselect_tutor_supervision_stays_on_selected_menu_stages() -> None:
    """A visible two-card tutor never trains completion at the later STOP."""

    select_row = (0.0, 1.0, 0.5, 0.5, 0.0, 1.0, 0.0, 1.0)
    stop_row = (0.0,) * 8
    stages = [
        PolicyStage(
            options=_sv(),
            action_combos=[[0], [1]],
            target_index=0,
            ledger_option_features=(select_row, stop_row),
        ),
        PolicyStage(
            options=_sv(),
            action_combos=[[0, 1], [0, 2]],
            target_index=0,
            ledger_option_features=(select_row, stop_row),
        ),
        PolicyStage(
            options=_sv(),
            action_combos=[[0, 1], [0, 1, 2]],
            target_index=0,
            selected_is_stop=True,
            # A factorized STOP can retain the prior card-bearing projection.
            ledger_option_features=(select_row, stop_row),
        ),
    ]
    observation = {
        "current": {"yourIndex": 0},
        "select": {
            "option": [
                {"area": "Deck", "playerIndex": 0, "index": 0},
                {"area": "Deck", "playerIndex": 0, "index": 1},
                {"area": "Hand", "playerIndex": 0, "index": 0},
            ],
            "deck": [_card(101, 1001), _card(102, 1002)],
        },
    }
    labels = {
        "schema": "poke_bot.own_deck_supervision/v1",
        "version": 1,
        "visible_tutor_completion": {
            "selected_card_ids": [101, 102],
            "selected_card_serials": [1001, 1002],
        },
    }

    mapped = dataset._selected_visible_tutor_stage_indices(
        observation,
        [0, 1],
        stages,
        labels,
    )
    assert mapped == (0, 1)
    # The sidecar-only reconstruction path has no raw observation, so it must
    # still reject a STOP stage whose inherited row looks card-bearing.
    assert dataset._tutor_stage_indices_from_option_features(stages) == (0, 1)

    decision = DecisionSample(
        board=_sv(features.NUM_BOARD_TOKENS),
        options=_sv(),
        action=[0, 1],
        action_combo_index=0,
        action_combos=stages[0].action_combos,
        env_step=7,
        policy_stages=stages,
        own_deck_supervision=labels,
        own_deck_supervision_stage_indices={"visible_tutor_completion": mapped},
    )
    assert (
        _own_deck_supervision_for_stage(
            decision,
            family="visible_tutor_completion",
            stage_index=0,
            stage_count=len(stages),
        )
        is labels
    )
    assert (
        _own_deck_supervision_for_stage(
            decision,
            family="visible_tutor_completion",
            stage_index=1,
            stage_count=len(stages),
        )
        is labels
    )
    assert _own_deck_supervision_for_stage(
        decision,
        family="visible_tutor_completion",
        stage_index=2,
        stage_count=len(stages),
    ) is None


def _sidecar_fixture(
    *,
    observation: dict | None = None,
) -> tuple[GameSequence, dict, str, dict[str, str]]:
    """One manually-featurized row with the exact r259 sidecar ABI."""

    deck = [1] * 60
    observed = deepcopy(observation) if observation is not None else _observation(
        known_prize=False
    )
    snapshot = OwnDeckLedger(deck).observe(observed)
    public_observation_fingerprint = "sha256:" + "c" * 64
    stage = PolicyStage(options=_sv(), action_combos=[[]], target_index=0)
    decision = DecisionSample(
        board=_sv(features.NUM_BOARD_TOKENS),
        options=_sv(),
        action=[],
        action_combo_index=0,
        action_combos=[[]],
        env_step=7,
        policy_stages=[stage],
        observation_fingerprint=public_observation_fingerprint,
    )
    sequence = GameSequence(
        episode_id="episode-sidecar",
        seat=0,
        archetype="alakazam",
        opp_archetype="other",
        deck=deck,
        value=0.0,
        decisions=[decision],
    )
    labels = build_own_deck_supervision_targets(
        [{"env_step": 7, "observation": observed, "action": []}]
    )[0]
    terminal = dict(labels["terminal_conversion"])
    tutor = dict(labels["visible_tutor_completion"])
    source_sha = "sha256:" + "a" * 64
    meta_sha = "sha256:" + "b" * 64
    option_features = snapshot.option_features(observed, [[]])
    row = {
        "schema": OWN_DECK_ROLLOUT_SIDECAR_SCHEMA,
        "version": OWN_DECK_ROLLOUT_SIDECAR_VERSION,
        "episode_id": sequence.episode_id,
        "seat": sequence.seat,
        "env_step": decision.env_step,
        "source_date": "2026-07-22",
        "source_manifest_sha256": source_sha,
        "deck_fingerprint": snapshot.deck_fingerprint,
        "observation_fingerprint": public_observation_fingerprint,
        "ledger_observation_fingerprint": snapshot.observation_fingerprint,
        "board_feature_fingerprint": dataset._sparse_board_feature_fingerprint(
            decision.board
        ),
        "ledger_snapshot": snapshot.to_dict(),
        "policy_stage_option_features": [
            {
                "stage_index": 0,
                "action_combos_fingerprint": dataset._action_combos_fingerprint([[]]),
                "candidate_count": 1,
                "selected_index": 0,
                "ledger_option_features": [list(option_features[0])],
            }
        ],
        "supervision": {
            "schema": labels["schema"],
            "version": labels["version"],
            "target_only": True,
            "terminal_conversion": {
                "labels": terminal,
                "vector": list(terminal_conversion_target_vector(terminal)),
                "mask": list(terminal_conversion_target_mask(terminal)),
            },
            "visible_tutor_completion": {
                "labels": tutor,
                "vector": list(visible_tutor_completion_target_vector(tutor)),
                "mask": list(visible_tutor_completion_target_mask(tutor)),
            },
        },
        "training_eligibility": {"active_r241": False},
    }
    return sequence, row, source_sha, {"2026-07-22": meta_sha}


def _attack_ko_prize_observations() -> tuple[dict, dict]:
    """A public immediate attack transition: one prize to zero and a KO."""

    before = _observation(known_prize=False)
    opponent = before["current"]["players"][1]
    opponent["active"] = [_card(77, 7700)]
    opponent["bench"] = []
    opponent["discard"] = []
    after = deepcopy(before)
    after["current"]["result"] = 0
    after["current"]["players"][0]["prize"] = []
    after_opponent = after["current"]["players"][1]
    after_opponent["active"] = []
    after_opponent["discard"] = [_card(77, 7700)]
    return before, after


def test_sidecar_join_is_one_to_one_and_attaches_only_after_validation() -> None:
    sequence, row, source_sha, meta_sha256s = _sidecar_fixture()

    untrusted, direct_row, source_sha, meta_sha256s = _sidecar_fixture()
    with pytest.raises(OwnDeckSidecarJoinError, match="test-only"):
        dataset.attach_own_deck_sidecar(
            [untrusted],
            expected_source_manifest_sha256=source_sha,
            daily_meta_sha256s=meta_sha256s,
            sidecar_rows=[direct_row],
        )

    provenance = dataset.attach_own_deck_sidecar(
        [sequence],
        expected_source_manifest_sha256=source_sha,
        daily_meta_sha256s=meta_sha256s,
        sidecar_rows=[row],
        test_only_sidecar_rows=True,
    )

    decision = sequence.decisions[0]
    assert provenance["one_to_one_coverage"] is True
    assert provenance["joined_decision_count"] == 1
    assert provenance["observation_fingerprint_parity_count"] == 1
    assert provenance["canonical_record_key_coverage"] is True
    assert decision.ledger_snapshot is not None
    assert decision.own_deck_supervision is not None
    assert decision.policy_stages[0].ledger_option_features == ((0.0,) * 8,)
    assert decision.own_deck_supervision_stage_indices == {
        "terminal_conversion": (0,)
    }

    second, malformed, source_sha, meta_sha256s = _sidecar_fixture()
    malformed = deepcopy(malformed)
    malformed["board_feature_fingerprint"] = "sha256:" + "d" * 64
    with pytest.raises(OwnDeckSidecarJoinError, match="board fingerprint"):
        dataset.attach_own_deck_sidecar(
            [second],
            expected_source_manifest_sha256=source_sha,
            daily_meta_sha256s=meta_sha256s,
            sidecar_rows=[malformed],
            test_only_sidecar_rows=True,
        )
    # Validation is all-or-nothing: the bad candidate was never partially
    # supplied with a ledger or target-only labels.
    assert second.decisions[0].ledger_snapshot is None
    assert second.decisions[0].own_deck_supervision is None

    third, changed_observation, source_sha, meta_sha256s = _sidecar_fixture()
    changed_observation = deepcopy(changed_observation)
    changed_observation["observation_fingerprint"] = "sha256:" + "e" * 64
    with pytest.raises(OwnDeckSidecarJoinError, match="one-to-one"):
        dataset.attach_own_deck_sidecar(
            [third],
            expected_source_manifest_sha256=source_sha,
            daily_meta_sha256s=meta_sha256s,
            sidecar_rows=[changed_observation],
            test_only_sidecar_rows=True,
        )
    assert third.decisions[0].ledger_snapshot is None


def test_sidecar_authoritative_ko_prize_labels_can_replace_local_masked_oracle() -> None:
    """r241's missing transition is unavailable, not a factual zero label."""

    before, after = _attack_ko_prize_observations()
    sequence, row, source_sha, meta_sha256s = _sidecar_fixture(observation=before)
    decision = sequence.decisions[0]
    local_labels = build_own_deck_supervision_targets(
        [{"env_step": 7, "observation": before, "action": []}]
    )[0]
    authoritative_labels = build_own_deck_supervision_targets(
        [
            {
                "env_step": 7,
                "observation": before,
                "action": [],
                "transition_after": {"observation": after},
            }
        ]
    )[0]
    local_terminal = local_labels["terminal_conversion"]
    authoritative_terminal = authoritative_labels["terminal_conversion"]
    assert all(
        local_terminal[name]["mask"] is False
        for name in ("terminal_class", "prize_closeout", "opponent_knockout")
    )
    assert authoritative_terminal["terminal_class"] == {"value": 1, "mask": True}
    assert authoritative_terminal["prize_closeout"] == {"value": 1.0, "mask": True}
    assert authoritative_terminal["opponent_knockout"] == {"value": 1.0, "mask": True}

    # Simulate the raw reconstruction oracle: its ledger/options are exact,
    # while its r241 record lacks `transition_after` and therefore masks facts.
    decision.ledger_snapshot = OwnDeckLedger(sequence.deck).observe(before)
    decision.policy_stages[0].ledger_option_features = tuple(
        tuple(values)
        for values in row["policy_stage_option_features"][0][
            "ledger_option_features"
        ]
    )
    decision.own_deck_supervision = local_labels
    decision.own_deck_supervision_stage_indices = {"terminal_conversion": (0,)}
    row = deepcopy(row)
    row["supervision"]["terminal_conversion"] = {
        "labels": authoritative_terminal,
        "vector": list(terminal_conversion_target_vector(authoritative_terminal)),
        "mask": list(terminal_conversion_target_mask(authoritative_terminal)),
    }

    dataset.attach_own_deck_sidecar(
        [sequence],
        expected_source_manifest_sha256=source_sha,
        daily_meta_sha256s=meta_sha256s,
        sidecar_rows=[row],
        test_only_sidecar_rows=True,
    )
    attached = decision.own_deck_supervision
    assert attached is not None
    assert attached["terminal_conversion"]["terminal_class"] == {
        "value": 1,
        "mask": True,
    }
    assert attached["terminal_conversion"]["prize_closeout"] == {
        "value": 1.0,
        "mask": True,
    }
    assert attached["terminal_conversion"]["opponent_knockout"] == {
        "value": 1.0,
        "mask": True,
    }

    # Conversely, a local row with an actual immediate public transition is a
    # factual oracle, so a contradictory r259 value must fail closed.
    conflicting_sequence, conflicting_row, source_sha, meta_sha256s = _sidecar_fixture(
        observation=before
    )
    conflicting_decision = conflicting_sequence.decisions[0]
    conflicting_labels = deepcopy(authoritative_labels)
    conflicting_labels["terminal_conversion"]["prize_closeout"] = {
        "value": 0.0,
        "mask": True,
    }
    conflicting_decision.ledger_snapshot = OwnDeckLedger(
        conflicting_sequence.deck
    ).observe(before)
    conflicting_decision.policy_stages[0].ledger_option_features = tuple(
        tuple(values)
        for values in conflicting_row["policy_stage_option_features"][0][
            "ledger_option_features"
        ]
    )
    conflicting_decision.own_deck_supervision = conflicting_labels
    conflicting_decision.own_deck_supervision_stage_indices = {
        "terminal_conversion": (0,)
    }
    conflicting_row = deepcopy(conflicting_row)
    conflicting_row["supervision"]["terminal_conversion"] = {
        "labels": authoritative_terminal,
        "vector": list(terminal_conversion_target_vector(authoritative_terminal)),
        "mask": list(terminal_conversion_target_mask(authoritative_terminal)),
    }
    with pytest.raises(OwnDeckSidecarJoinError, match="observed supervision"):
        dataset.attach_own_deck_sidecar(
            [conflicting_sequence],
            expected_source_manifest_sha256=source_sha,
            daily_meta_sha256s=meta_sha256s,
            sidecar_rows=[conflicting_row],
            test_only_sidecar_rows=True,
        )


def _sha256(char: str) -> str:
    return "sha256:" + char * 64


def _contract_join_fixture() -> tuple[
    successor.OwnDeckSuccessorManifest,
    tuple[contract.DailySidecarMeta, ...],
    dict[str, object],
    str,
    str,
    str,
]:
    """Minimal complete r259 daily identity set for writer/contract parity."""

    manifest = successor.load_canonical_manifest()
    labels = contract.SupervisionLabelCounts(
        terminal_class_counts=(1, 0, 0, 0),
        terminal_scalars=(
            ("prize_closeout", contract.BinaryCounts(0, 1)),
            ("opponent_knockout", contract.BinaryCounts(0, 1)),
        ),
        tutor_terminal_class_counts=(0, 0, 0, 0),
        tutor_scalars=(
            ("selected_from_visible_deck", contract.BinaryCounts(0, 0)),
            ("selected_target_observed_after_action", contract.BinaryCounts(0, 0)),
            ("same_actor_followup", contract.BinaryCounts(0, 0)),
        ),
    )
    daily = tuple(
        contract.DailySidecarMeta(
            source_day=day,
            meta_sha256="sha256:" + f"{index + 1:064x}",
            shard_sha256="sha256:" + f"{index + 101:064x}",
            record_count=1,
            source_manifest_sha256=manifest.elmo_side_store.source_manifest_sha256,
            sidecar_build_code_sha256=_sha256("a"),
            sidecar_build_code_identities=(("own_deck_rollout_store.py", _sha256("b")),),
            source_snapshot_tree_sha256=_sha256("c"),
            image_id=manifest.elmo_side_store.container_image_id,
            classifier_sha256=_sha256("9"),
            label_counts=labels,
        )
        for index, day in enumerate(
            contract.expected_sidecar_days(manifest.elmo_side_store)
        )
    )
    daily_map = {item.source_day: item.meta_sha256 for item in daily}
    source_manifest_sha256 = manifest.elmo_side_store.source_manifest_sha256
    provenance: dict[str, object] = {
        "schema": dataset.OWN_DECK_SIDECAR_JOIN_SCHEMA,
        "source_manifest_sha256": source_manifest_sha256,
        "daily_meta_sha256s": daily_map,
        "sidecar_meta_identity": dataset._sidecar_meta_identity(
            source_manifest_sha256=source_manifest_sha256,
            daily_meta_sha256s=daily_map,
        ),
        "record_key": list(manifest.elmo_side_store.record_key),
        "sidecar_record_count": len(daily),
        "joined_decision_count": len(daily),
        "unmatched_record_count": 0,
        "duplicate_key_count": 0,
        "observation_fingerprint_parity_count": len(daily),
        "raw_reconstruction_parity_count": len(daily),
        "one_to_one_coverage": True,
        "canonical_record_key_coverage": True,
        "active_r241_training_eligible": False,
    }
    return (
        manifest,
        daily,
        provenance,
        _sha256("d"),
        _sha256("e"),
        _sha256("f"),
    )


def test_sealed_join_receipt_is_atomic_0444_and_accepted_by_next_train(
    tmp_path: Path,
) -> None:
    (
        manifest,
        daily,
        provenance,
        code_sha256,
        model_sha256,
        migration_receipt_sha256,
    ) = _contract_join_fixture()
    sidecar_dataset_sha256 = contract.sidecar_dataset_sha256(daily)
    target = tmp_path / "own-deck-sidecar-join.json"

    assert dataset.write_own_deck_sidecar_join_receipt(
        target,
        provenance,
        manifest_sha256=manifest.identity.sha256,
        code_sha256=code_sha256,
        sidecar_dataset_sha256=sidecar_dataset_sha256,
        model_sha256=model_sha256,
        migration_receipt_sha256=migration_receipt_sha256,
    ) == target
    receipt = dataset.validate_own_deck_sidecar_join_receipt(target)
    assert target.stat().st_mode & 0o777 == 0o444
    assert receipt["schema"] == contract.SIDE_STORE_JOIN_RECEIPT_SCHEMA
    assert receipt["join_provenance_schema"] == dataset.OWN_DECK_SIDECAR_JOIN_SCHEMA
    assert receipt["record_key"] == [
        "episode_id",
        "seat",
        "env_step",
        "observation_fingerprint",
    ]

    validated = contract._validate_join_receipt(
        target,
        manifest=manifest,
        source=contract.ContentIdentity(
            role="expert_source_manifest",
            identity="r241-exact20",
            sha256=manifest.elmo_side_store.source_manifest_sha256,
            path=manifest.elmo_side_store.source_manifest,
        ),
        model=contract.ContentIdentity(
            role="migrated_successor_checkpoint",
            identity="successor.pt",
            sha256=model_sha256,
        ),
        code=contract.ContentIdentity(
            role="successor_code",
            identity="r258-source",
            sha256=code_sha256,
        ),
        daily=daily,
        dataset_sha256=sidecar_dataset_sha256,
        migration_receipt_sha256=migration_receipt_sha256,
    )
    assert validated["receipt_sha256"] == receipt["receipt_sha256"]
    with pytest.raises(OwnDeckSidecarJoinError, match="already exists"):
        dataset.write_own_deck_sidecar_join_receipt(
            target,
            provenance,
            manifest_sha256=manifest.identity.sha256,
            code_sha256=code_sha256,
            sidecar_dataset_sha256=sidecar_dataset_sha256,
            model_sha256=model_sha256,
            migration_receipt_sha256=migration_receipt_sha256,
        )
