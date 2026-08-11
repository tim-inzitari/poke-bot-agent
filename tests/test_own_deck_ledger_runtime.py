"""Runtime transport tests for the dormant own-deck ledger successor."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from poke_bot import agent as agent_module
from poke_bot import batched_infer, config, features
from poke_bot.agent import PolicyAgent
from poke_bot.batched_infer import (
    LeafPacket,
    RemoteLeafClient,
    featurize_packets,
    forward_featurized,
    forward_leaf_batch,
)
from poke_bot.features import SparseVector
from poke_bot.model import build_model
from poke_bot.own_deck_ledger import OPTION_FEATURE_DIM, OwnDeckLedger


@dataclass(frozen=True)
class _Snapshot:
    serial: int

    def option_features(self, _obs, candidates):
        # The real core's contract is an 8-wide row per legal candidate.
        return tuple(
            (
                float(self.serial),
                float(len(candidate)),
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            )
            for candidate in candidates
        )


class _Ledger:
    def __init__(self, serial: int = 0) -> None:
        self.serial = serial
        self.observations: list[object] = []
        self.reset_calls = 0

    def observe(self, observation):
        self.serial += 1
        self.observations.append(observation)
        return _Snapshot(self.serial)

    def reset(self) -> None:
        self.serial = 0
        self.observations.clear()
        self.reset_calls += 1

    def fork(self):
        clone = _Ledger(self.serial)
        clone.observations = list(self.observations)
        clone.reset_calls = self.reset_calls
        return clone


def _install_lightweight_history_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(agent_module.features, "assert_info_set", lambda _obs: None)
    monkeypatch.setattr(
        agent_module.features,
        "build_board_tokens",
        lambda obs, _deck: ("board", id(obs)),
    )
    monkeypatch.setattr(
        agent_module.features,
        "build_option_tokens",
        lambda _obs, candidates: tuple(tuple(row) for row in candidates),
    )


def _enabled_agent() -> tuple[PolicyAgent, _Ledger]:
    # The actual core accepts this full starting multiset; immediately replace
    # it with a probe so the test isolates lifecycle wiring rather than ledger
    # parsing rules (covered by the core tests).
    policy = PolicyAgent(
        model=None,
        deck=[1] * 60,
        own_deck_ledger_enabled=True,
    )
    ledger = _Ledger()
    policy.own_deck_ledger = ledger
    return policy, ledger


def test_default_off_preserves_short_deck_and_no_ledger_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_history_features(monkeypatch)
    monkeypatch.setenv("POKEBOT_OWN_DECK_LEDGER_ENABLED", "1")
    policy = PolicyAgent(model=None, deck=[1])

    board = policy._append_decision_history({"select": {"option": []}})

    assert board[0] == "board"
    assert policy.own_deck_ledger_enabled is False
    assert policy.own_deck_ledger is None
    assert policy.ledger_history == []


def test_enabled_successor_requires_direct_policy_and_exact_deck() -> None:
    with pytest.raises(ValueError, match="exact 60-card"):
        PolicyAgent(
            model=None,
            deck=[1],
            own_deck_ledger_enabled=True,
        )
    with pytest.raises(ValueError, match="direct-policy-only"):
        PolicyAgent(
            model=None,
            deck=[1] * 60,
            own_deck_ledger_enabled=True,
            use_mcts=True,
            oracle_mode=True,
        )
    with pytest.raises(ValueError, match="recursive turn planning"):
        PolicyAgent(
            model=None,
            deck=[1] * 60,
            own_deck_ledger_enabled=True,
            use_recursive_turn_planner=True,
        )

    # Dataclass fields are mutable, so route entrypoints recheck the same
    # direct-policy boundary rather than relying only on construction-time
    # validation.
    policy, _ledger = _enabled_agent()
    policy.use_mcts = True
    with pytest.raises(RuntimeError, match="direct-policy-only"):
        policy.mcts_select({"select": {"option": []}})
    policy.use_mcts = False
    policy.use_recursive_turn_planner = True
    with pytest.raises(RuntimeError, match="direct-policy-only"):
        policy.rtp_select({"select": {"option": []}})


def test_observe_before_history_reset_and_transactional_search_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_history_features(monkeypatch)
    policy, ledger = _enabled_agent()
    observation = {"select": {"option": []}}

    policy._append_decision_history(observation)
    assert ledger.observations == [observation]
    assert policy.ledger_history == [_Snapshot(1)]
    assert len(policy.ledger_history) == len(policy.board_history)

    policy.reset_game()
    assert ledger.reset_calls == 1
    assert policy.ledger_history == []
    assert policy.board_history == []

    # Force a trusted search failure after it has appended one observation.
    # The fallback must restore the forked ledger and append exactly one fresh
    # snapshot, rather than retaining the failed speculative observation.
    policy.use_mcts = True
    policy.oracle_mode = True
    calls: list[str] = []

    def _flaky_call(self, obs):
        self._append_decision_history(obs)
        calls.append("call")
        if len(calls) == 1:
            raise RuntimeError("search failed")
        return [0]

    monkeypatch.setattr(PolicyAgent, "__call__", _flaky_call)
    assert policy.trusted_search_or_greedy_select(observation, search=True) == [0]
    assert calls == ["call", "call"]
    assert len(policy.board_history) == 1
    assert policy.ledger_history == [_Snapshot(1)]
    assert isinstance(policy.own_deck_ledger, _Ledger)
    assert policy.own_deck_ledger.serial == 1


def test_enabled_go_first_propagates_ledger_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_history_features(monkeypatch)
    policy, _ledger = _enabled_agent()

    class _BrokenLedger:
        def observe(self, _obs):
            raise ValueError("bad public ledger input")

    policy.own_deck_ledger = _BrokenLedger()
    with pytest.raises(ValueError, match="bad public ledger input"):
        policy._record_go_first({"select": {"option": []}}, [0])

    # The public entrypoint must not turn the same successor contract failure
    # into the legacy random-legal fallback when ``strict_runtime`` is false.
    with pytest.raises(RuntimeError, match="policy runtime failed closed"):
        policy({"select": {"option": []}})


def test_factorized_stages_freeze_snapshot_and_recompute_option_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_history_features(monkeypatch)
    policy, ledger = _enabled_agent()
    observation = {"select": {"option": [{"id": 1}, {"id": 2}]}}
    board = policy._append_decision_history(observation)
    snapshot = policy.ledger_history[-1]

    monkeypatch.setattr(
        agent_module.cg_env,
        "to_observation",
        lambda _obs: SimpleNamespace(current=SimpleNamespace(yourIndex=0)),
    )

    def _stages(_obs, prefix):
        if prefix == []:
            return [[0], [1]]
        if prefix == [1]:
            return [[1], [1, 0]]
        return [list(prefix)]

    monkeypatch.setattr(
        agent_module.features,
        "factorized_action_candidates",
        _stages,
    )
    seen: list[LeafPacket] = []

    def _backend(packets):
        packet = packets[0]
        seen.append(packet)
        candidates = packet.action_combos_override or []
        return [
            LeafPacket(
                obs=packet.obs,
                your_deck=packet.your_deck,
                root_seat=packet.root_seat,
                priors=[0.1, 0.9],
                combos=[list(row) for row in candidates],
            )
        ]

    policy.leaf_backend = _backend
    assert policy._factorized_greedy_prepared(observation, board) == [1, 0]

    assert len(ledger.observations) == 1
    assert len(seen) == 2
    assert all(packet.ledger_snapshot is snapshot for packet in seen)
    assert all(packet.history_ledger_snapshots == [snapshot] for packet in seen)
    assert seen[0].ledger_option_features == snapshot.option_features(
        observation, [[0], [1]]
    )
    assert seen[1].ledger_option_features == snapshot.option_features(
        observation, [[1], [1, 0]]
    )


class _HistoryModel:
    decision_context = "history"
    own_deck_ledger_option_feature_dim = 8

    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls: list[dict[str, object]] = []

    def parameters(self):
        yield self.weight

    def forward_history_batch(self, _histories, _options, **kwargs):
        self.calls.append(kwargs)
        batch_size = len(_histories)
        return {
            "policy_logits": torch.tensor([[0.0, 1.0]] * batch_size),
            "value": torch.tensor([0.25] * batch_size),
        }


class _RuntimeHistoryModel(_HistoryModel):
    own_deck_ledger_runtime_enabled = True


class _LegacySignatureHistoryModel(_HistoryModel):
    """Pre-ledger signature used to verify only dormant paths may retry."""

    def forward_history_batch(
        self,
        histories,
        options,
        *,
        n_options=None,
        previous_action_histories=None,
        matchup_routes=None,
    ):
        self.calls.append(
            {
                "n_options": n_options,
                "previous_action_histories": previous_action_histories,
                "matchup_routes": matchup_routes,
            }
        )
        batch_size = len(histories)
        return {
            "policy_logits": torch.tensor([[0.0, 1.0]] * batch_size),
            "value": torch.tensor([0.25] * batch_size),
        }


class _EnabledLegacySignatureHistoryModel(_LegacySignatureHistoryModel):
    own_deck_ledger_enabled = True


class _EnabledLegacySignatureLocalModel:
    """A stale local model signature behind an explicitly enabled successor."""

    decision_context = "stateless"
    kv_cache_enabled = False
    max_context = 4
    own_deck_ledger_enabled = True

    def __init__(self) -> None:
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def parameters(self):
        yield self.weight

    def eval(self):
        return self

    def forward(
        self,
        _board,
        _options,
        *,
        append_cache=False,
        n_options=None,
        matchup_routes=None,
    ):
        raise AssertionError("ledger-enabled forwarding must not silently retry")


class _LedgerCapabilityModel:
    """Minimal local model surface used to validate agent/model binding."""

    decision_context = "stateless"
    kv_cache_enabled = False
    max_context = 4

    def __init__(self, *, enabled: bool, runtime_enabled: bool) -> None:
        self.own_deck_ledger_enabled = bool(enabled)
        self.own_deck_ledger_runtime_enabled = bool(runtime_enabled)
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def parameters(self):
        yield self.weight

    def eval(self):
        return self


def _install_lightweight_leaf_features(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(batched_infer.features, "assert_info_set", lambda _obs: None)
    monkeypatch.setattr(
        batched_infer.features,
        "build_board_tokens",
        lambda obs, _deck: ("board", obs),
    )
    monkeypatch.setattr(
        batched_infer.features,
        "build_option_tokens",
        lambda _obs, combos: tuple(tuple(row) for row in combos),
    )
    monkeypatch.setattr(
        batched_infer.cg_env,
        "to_observation",
        lambda _obs: SimpleNamespace(current=SimpleNamespace(yourIndex=0)),
    )


def test_local_and_remote_leaf_transport_preserve_ledger_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_leaf_features(monkeypatch)
    first = _Snapshot(1)
    second = _Snapshot(2)
    packet = LeafPacket(
        obs={"select": {"option": []}},
        your_deck=[1] * 60,
        root_seat=0,
        history_boards=["old", "current"],
        history_previous_actions=[None, None],
        history_ledger_snapshots=[first, second],
        ledger_snapshot=second,
        ledger_option_features=((1.0,) * 8, (2.0,) * 8),
        action_combos_override=[[0], [1]],
    )

    leaves = featurize_packets([packet])
    assert leaves.ledger_snapshots == [second]
    assert leaves.ledger_histories == [[first, second]]
    assert leaves.ledger_option_features == [packet.ledger_option_features]

    model = _HistoryModel()
    local = forward_leaf_batch(model, [packet])[0]
    assert model.calls[0]["ledger_histories"] == [[first, second]]
    assert model.calls[0]["ledger_option_features"] == [packet.ledger_option_features]
    assert local.ledger_snapshot is second
    assert local.history_ledger_snapshots == [first, second]
    assert local.ledger_option_features == packet.ledger_option_features

    # The client sends the same featurized snapshots to a remote leaf and
    # reconstructs a response without dropping packet-local ledger data.
    outbound = queue.Queue()
    inbound = queue.Queue()
    alive = threading.Event()
    alive.set()
    inbound.put(
        {
            "generation": 0,
            "rid": 1,
            "ok": True,
            "values": [(0.25, [0.25, 0.75])],
            "version": 0,
            "checkpoint_digest": "",
        }
    )
    remote = RemoteLeafClient(
        0,
        outbound,
        inbound,
        generation=0,
        alive_evt=alive,
        timeout_s=0.2,
    )
    returned = remote([packet])[0]
    sent = outbound.get_nowait()["leaves"]
    assert sent.ledger_snapshots == [second]
    assert sent.ledger_histories == [[first, second]]
    assert sent.ledger_option_features == [packet.ledger_option_features]
    assert returned.ledger_snapshot is second
    assert returned.history_ledger_snapshots == [first, second]
    assert returned.ledger_option_features == packet.ledger_option_features


def test_legacy_leaf_model_receives_no_new_keywords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_leaf_features(monkeypatch)
    packet = LeafPacket(
        obs={"select": {"option": []}},
        your_deck=[1] * 60,
        root_seat=0,
        action_combos_override=[[0], [1]],
    )
    model = _HistoryModel()

    forward_leaf_batch(model, [packet])

    assert "ledger_histories" not in model.calls[0]
    assert "ledger_option_features" not in model.calls[0]


def test_unsupported_ledger_keywords_retry_only_for_dormant_legacy_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_leaf_features(monkeypatch)
    snapshot = _Snapshot(6)
    packet = LeafPacket(
        obs={"select": {"option": []}},
        your_deck=[1] * 60,
        root_seat=0,
        history_boards=["current"],
        history_ledger_snapshots=[snapshot],
        ledger_snapshot=snapshot,
        ledger_option_features=((6.0,) * 8, (6.0,) * 8),
        action_combos_override=[[0], [1]],
    )

    # Old models with no capability marker retain their historical no-keyword
    # path even if a mixed packet incidentally carries a dormant side-store.
    dormant = _LegacySignatureHistoryModel()
    forward_leaf_batch(dormant, [packet])
    assert len(dormant.calls) == 1

    # Once the successor architecture is explicit, a stale model signature is
    # a contract failure rather than permission to discard public ledger input.
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        forward_leaf_batch(_EnabledLegacySignatureHistoryModel(), [packet])


def test_enabled_policy_does_not_retry_a_stale_local_model_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_history_features(monkeypatch)
    policy = PolicyAgent(
        model=_EnabledLegacySignatureLocalModel(),
        deck=[1] * 60,
    )
    policy.own_deck_ledger = _Ledger()
    observation = {"select": {"option": [{"id": 1}, {"id": 2}]}}
    board = policy._append_decision_history(observation)
    monkeypatch.setattr(
        agent_module.cg_env,
        "to_observation",
        lambda _obs: SimpleNamespace(current=SimpleNamespace(yourIndex=0)),
    )
    monkeypatch.setattr(
        agent_module.features,
        "factorized_action_candidates",
        lambda _obs, prefix: [[0], [1]] if not prefix else [list(prefix)],
    )

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        policy._factorized_greedy_prepared(observation, board)


def test_local_successor_model_and_policy_ledger_must_bind_together() -> None:
    """No local runtime-enabled model may silently receive neutral inputs."""

    with pytest.raises(ValueError, match="runtime_enabled model requires an enabled"):
        PolicyAgent(
            model=_LedgerCapabilityModel(enabled=True, runtime_enabled=True),
            deck=[1] * 60,
            own_deck_ledger_enabled=False,
        )

    with pytest.raises(ValueError, match="physical own-deck ledger capability"):
        PolicyAgent(
            model=_LedgerCapabilityModel(enabled=False, runtime_enabled=False),
            deck=[1] * 60,
            own_deck_ledger_enabled=True,
        )

    # The inferred capability remains the convenient safe path for an actual
    # local successor model, and it creates the match-local side-store.
    policy = PolicyAgent(
        model=_LedgerCapabilityModel(enabled=True, runtime_enabled=True),
        deck=[1] * 60,
    )
    assert policy.own_deck_ledger_enabled is True
    assert policy.own_deck_ledger is not None


def test_mixed_successor_and_legacy_batch_keeps_per_row_neutral_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_leaf_features(monkeypatch)
    snapshot = _Snapshot(3)
    successor = LeafPacket(
        obs={"select": {"option": []}},
        your_deck=[1] * 60,
        root_seat=0,
        history_boards=["current"],
        history_ledger_snapshots=[snapshot],
        ledger_snapshot=snapshot,
        ledger_option_features=((3.0,) * 8, (3.0,) * 8),
        action_combos_override=[[0], [1]],
    )
    legacy = LeafPacket(
        obs={"select": {"option": []}},
        your_deck=[1] * 60,
        root_seat=0,
        history_boards=["current"],
        action_combos_override=[[0], [1]],
    )
    model = _HistoryModel()

    forward_leaf_batch(model, [successor, legacy])

    assert model.calls[0]["ledger_histories"] == [[snapshot], [None]]
    assert model.calls[0]["ledger_option_features"] == [
        successor.ledger_option_features,
        None,
    ]


def test_malformed_ledger_transport_is_neutral_per_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_leaf_features(monkeypatch)
    snapshot = _Snapshot(4)
    packet = LeafPacket(
        obs={"select": {"option": []}},
        your_deck=[1] * 60,
        root_seat=0,
        history_boards=["old", "current"],
        # A stale side-store cannot be assigned to both temporal boards.  The
        # transport retains the independently supplied current snapshot only.
        history_ledger_snapshots=[snapshot],
        ledger_snapshot=snapshot,
        action_combos_override=[[0], [1]],
    )
    leaves = featurize_packets([packet])
    assert leaves.ledger_histories == [[None, snapshot]]

    model = _HistoryModel()
    forward_featurized(
        model,
        boards=["left", "right"],
        opts=["left-options", "right-options"],
        n_opts=[2, 2],
        seats=[0, 0],
        root_seats=[0, 0],
        histories=[["left"], ["right"]],
        # A missing second row must not shift the first row onto it.
        ledger_histories=[[snapshot]],
        ledger_option_features=[((4.0,) * 8,)],
    )
    assert model.calls[0]["ledger_histories"] == [[snapshot], [None]]
    assert model.calls[0]["ledger_option_features"] == [None, None]


def test_runtime_enabled_model_rejects_missing_ledger_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_lightweight_leaf_features(monkeypatch)
    snapshot = _Snapshot(5)
    successor = LeafPacket(
        obs={"select": {"option": []}},
        your_deck=[1] * 60,
        root_seat=0,
        history_boards=["current"],
        history_ledger_snapshots=[snapshot],
        ledger_snapshot=snapshot,
        ledger_option_features=((5.0,) * 8, (5.0,) * 8),
        action_combos_override=[[0], [1]],
    )
    legacy = LeafPacket(
        obs={"select": {"option": []}},
        your_deck=[1] * 60,
        root_seat=0,
        history_boards=["current"],
        action_combos_override=[[0], [1]],
    )

    with pytest.raises(ValueError, match="requires every history snapshot"):
        forward_leaf_batch(_RuntimeHistoryModel(), [successor, legacy])


def _native_board() -> SparseVector:
    board = SparseVector()
    for _ in range(features.NUM_BOARD_TOKENS):
        board.word_start()
    return board


def _native_options() -> SparseVector:
    options = SparseVector()
    for index in range(2):
        options.word_start()
        options.add(index + 1, 1.0)
    return options


def _real_snapshot():
    ledger = OwnDeckLedger([1] * 60)
    snapshot = ledger.observe(
        {
            "current": {
                "yourIndex": 0,
                "looking": [],
                "players": [
                    {
                        "hand": [{"id": 1, "serial": 101}],
                        "active": [],
                        "bench": [],
                        "discard": [],
                        "prize": [None] * 6,
                        "deckCount": 53,
                    },
                    {
                        "hand": [],
                        "active": [],
                        "bench": [],
                        "discard": [],
                        "prize": [],
                    },
                ],
            },
            "select": {"deck": [], "option": []},
        }
    )
    assert snapshot.integrity_ok is True
    return snapshot


def _runtime_ledger_model(monkeypatch: pytest.MonkeyPatch):
    """Tiny real successor model for the batched serialization boundary."""

    monkeypatch.setattr(features, "card_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "attack_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "encoder_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "decoder_vocab_size", lambda: 64)
    monkeypatch.setattr(features, "decoder_binding_offset", lambda: 64)
    model = build_model(
        config.ModelConfig(
            d_model=16,
            spatial_layers=1,
            temporal_layers=1,
            option_decoder_layers=1,
            n_heads=4,
            ff_dim=32,
            max_context=8,
            temporal_pos="rope",
            decision_context="history",
            kv_cache=False,
            dropout=0.0,
            own_deck_ledger_enabled=True,
            own_deck_ledger_runtime_enabled=True,
        ),
        device=torch.device("cpu"),
        aux_archetype_classes=3,
        encoder_vocab=64,
        decoder_vocab=64,
        belief_card_vocab=64,
    )
    model.eval()
    assert model.own_deck_ledger_adapter is not None
    with torch.no_grad():
        # The physical adapter starts zero-safe; make valid snapshots visibly
        # observable so this transport test distinguishes pass-through from a
        # fail-closed neutral mapping without enabling any unrelated route.
        model.own_deck_ledger_adapter.output.bias.zero_()
        model.own_deck_ledger_adapter.output.bias[0] = 0.5
    return model


def _batched_ledger_result(model, board, options, snapshot):
    zero_menu = ((0.0,) * OPTION_FEATURE_DIM, (0.0,) * OPTION_FEATURE_DIM)
    return forward_featurized(
        model,
        [board],
        [options],
        [2],
        [0],
        [0],
        histories=[[board]],
        ledger_histories=[[snapshot]],
        ledger_option_features=[zero_menu],
    )[0]


def test_batched_serialized_ledger_tampering_is_neutral_but_typed_snapshot_flows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The remote/local batch boundary never turns a forged dict into input."""

    torch.manual_seed(23)
    model = _runtime_ledger_model(monkeypatch)
    snapshot = _real_snapshot()
    board, options = _native_board(), _native_options()
    with torch.no_grad():
        baseline_out = model.forward_history_batch([[board]], [options], n_options=[2])
    baseline = (
        float(baseline_out["value"][0]),
        batched_infer._policy_probs(baseline_out["policy_logits"][0]),
    )

    typed = _batched_ledger_result(model, board, options, snapshot)
    canonical = _batched_ledger_result(model, board, options, snapshot.to_dict())
    assert typed[0] == pytest.approx(canonical[0], abs=1e-7)
    assert typed[1] == pytest.approx(canonical[1], abs=1e-7)
    assert (
        abs(typed[0] - baseline[0]) > 1e-6
        or any(abs(left - right) > 1e-6 for left, right in zip(typed[1], baseline[1]))
    )

    tampered_fingerprint = snapshot.to_dict()
    tampered_fingerprint["fingerprint"] = "sha256:" + "0" * 64
    tampered_counts = snapshot.to_dict()
    tampered_counts["starting_counts"] = [[1, 59]]
    for forged in (tampered_fingerprint, tampered_counts):
        neutral = _batched_ledger_result(model, board, options, forged)
        assert neutral[0] == pytest.approx(baseline[0], abs=1e-7)
        assert neutral[1] == pytest.approx(baseline[1], abs=1e-7)
