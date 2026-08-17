import os
import queue
import threading
import time

import pytest

from scripts import unattended_monitor
from poke_bot import batched_infer
from poke_bot.batched_infer import (
    FeaturizedLeaves,
    LeafPacket,
    RemoteLeafClient,
    RemoteLeafTimeout,
)
from poke_bot.eval_metrics import FieldReport
from poke_bot.mcts import GameClock
from poke_bot.train import process_with_oom_splitting


def _fake_features(_packets):
    return FeaturizedLeaves(
        boards=[object()],
        opts=[object()],
        combos=[[[0]]],
        n_opts=[1],
        seats=[0],
        root_seats=[0],
    )


def test_remote_request_times_out_instead_of_hanging(monkeypatch) -> None:
    monkeypatch.setattr(batched_infer, "featurize_packets", _fake_features)
    alive = threading.Event()
    alive.set()
    client = RemoteLeafClient(
        0,
        queue.Queue(),
        queue.Queue(),
        generation=1,
        alive_evt=alive,
        timeout_s=0.01,
    )
    with pytest.raises(RemoteLeafTimeout):
        client([LeafPacket(obs=object(), your_deck=[1] * 60, root_seat=0)])


def test_remote_leaf_client_keeps_home_queue_when_depths_tie() -> None:
    queues = [queue.Queue() for _ in range(4)]
    alive = [threading.Event() for _ in queues]
    for event in alive:
        event.set()
    client = RemoteLeafClient(
        9,
        queues[2],
        queue.Queue(),
        req_qs=queues,
        leaf_devices=[1, 1, 1, 1],
        alive_evts=alive,
        home_server_idx=2,
    )

    selected, selected_alive = client._select_req_target()

    assert selected is queues[2]
    assert selected_alive is alive[2]


def test_move_deadline_bounds_remote_rpc_and_game_clock_reserves_watchdog(
    monkeypatch,
) -> None:
    monkeypatch.setattr(batched_infer, "featurize_packets", _fake_features)
    alive = threading.Event()
    alive.set()
    client = RemoteLeafClient(
        0,
        queue.Queue(),
        queue.Queue(),
        generation=1,
        alive_evt=alive,
        timeout_s=30.0,
    )
    client.set_deadline(time.monotonic() - 0.01)
    with pytest.raises(RemoteLeafTimeout, match="move deadline"):
        client([LeafPacket(obs=object(), your_deck=[1] * 60, root_seat=0)])

    clock = GameClock(
        total_s=180.0,
        reserve_s=30.0,
        expected_search_decisions=30,
    )
    assert clock.next_move_budget(30.0) == pytest.approx(5.0)
    clock.consume(4.0)
    assert clock.remaining_s == pytest.approx(176.0)
    assert clock.decisions_used == 1

    long_core_clock = GameClock(
        total_s=1200.0,
        reserve_s=120.0,
        expected_search_decisions=128,
    )
    for _ in range(75):
        long_core_clock.consume(4.0)
    assert long_core_clock.next_move_budget(8.0) == pytest.approx(8.0)


def test_remote_response_generation_prevents_stale_slot_routing(monkeypatch) -> None:
    monkeypatch.setattr(batched_infer, "featurize_packets", _fake_features)
    alive = threading.Event()
    alive.set()
    req = queue.Queue()
    resp = queue.Queue()
    resp.put(
        {
            "generation": 1,
            "rid": 1,
            "ok": True,
            "values": [(0.0, [1.0])],
            "version": 3,
            "checkpoint_digest": "sha256:old",
        }
    )
    resp.put(
        {
            "generation": 2,
            "rid": 1,
            "ok": True,
            "values": [(0.25, [1.0])],
            "version": 4,
            "checkpoint_digest": "sha256:new",
            "queue_wait_ms": 1.5,
            "batch_size": 12,
            "inference_ms": 2.5,
        }
    )
    client = RemoteLeafClient(
        0,
        req,
        resp,
        generation=2,
        alive_evt=alive,
        expected_digest="sha256:new",
        expected_version=4,
        timeout_s=0.1,
    )
    out = client([LeafPacket(obs=object(), your_deck=[1] * 60, root_seat=0)])
    assert out[0].value == 0.25
    sent = req.get_nowait()
    assert sent["generation"] == 2
    assert sent["enqueued_ns"] > 0
    telemetry = client.telemetry_since(0)
    assert telemetry["remote_requests"] == 1
    assert telemetry["remote_leaves"] == 1
    assert telemetry["queue_wait_ms_mean"] == 1.5
    assert telemetry["queue_wait_ms_p95"] == 1.5
    assert telemetry["inference_batch_size_mean"] == 12


def test_remote_response_reads_live_checkpoint_expectation_proxy(monkeypatch) -> None:
    monkeypatch.setattr(batched_infer, "featurize_packets", _fake_features)
    alive = threading.Event()
    alive.set()
    req = queue.Queue()
    resp = queue.Queue()
    expected = {"digest": "sha256:first", "version": 7, "pinned": []}
    for rid, digest, version, value in (
        (1, "sha256:first", 7, 0.1),
        (2, "sha256:second", 8, 0.2),
    ):
        resp.put(
            {
                "generation": 3,
                "rid": rid,
                "ok": True,
                "values": [(value, [1.0])],
                "version": version,
                "checkpoint_digest": digest,
            }
        )
    client = RemoteLeafClient(
        0,
        req,
        resp,
        generation=3,
        alive_evt=alive,
        expected_digest=expected,
        expected_version=expected,
        timeout_s=0.1,
    )
    packet = LeafPacket(obs=object(), your_deck=[1] * 60, root_seat=0)
    assert client([packet])[0].value == pytest.approx(0.1)

    expected["digest"] = "sha256:second"
    expected["version"] = 8
    assert client([packet])[0].value == pytest.approx(0.2)


def test_expected_field_completeness_and_greedy_isolation() -> None:
    report = FieldReport(
        gate_threshold=0.0,
        expected_opponents={"a", "b"},
        min_games_per_opponent=2,
    )
    for seat in (0, 1):
        report.merge_game(
            "a",
            our_seat=seat,
            winner=seat,
            is_mirror=False,
            mcts_on=True,
            pair_id="a0",
        )
    report.merge_game(
        "a",
        our_seat=0,
        winner=1,
        is_mirror=False,
        mcts_on=False,
        pair_id="greedy",
    )
    first = report.summary()
    assert first["all_pass"] is False
    assert first["missing_opponents"] == ["b"]
    assert first["pooled_mcts"]["games"] == 2
    assert first["greedy_ablation"]["games"] == 1
    assert report.get("a").games == 2

    for seat in (0, 1):
        report.merge_game(
            "b",
            our_seat=seat,
            winner=seat,
            is_mirror=False,
            mcts_on=True,
            pair_id="b0",
        )
    final = report.summary()
    assert final["valid"] is True
    assert final["all_pass"] is True
    assert final["pooled_mcts"]["games"] == 4
    assert final["paired_mcts"]["complete_pairs"] == 2


def test_draw_aware_interval_preserves_draw_outcomes() -> None:
    report = FieldReport(
        gate_threshold=0.49,
        expected_opponents={"drawish"},
        min_games_per_opponent=4,
    )
    for game in range(4):
        report.merge_game(
            "drawish",
            our_seat=game % 2,
            winner=2,
            is_mirror=False,
            mcts_on=True,
            pair_id=None,
        )
    summary = report.summary()
    interval = summary["matchups"][0]["draw_aware_score_interval"]
    assert interval["center"] == 0.5
    assert interval["lower"] == 0.5
    assert interval["upper"] == 0.5
    assert summary["all_pass"] is True


def test_oom_split_processes_both_halves_completely() -> None:
    seen = []

    def process(items):
        if len(items) > 1:
            raise RuntimeError("CUDA out of memory")
        seen.extend(items)
        return items[0]

    completed = process_with_oom_splitting(
        [0, 1, 2, 3],
        process,
        is_oom=lambda exc: "out of memory" in str(exc).lower(),
    )
    assert seen == [0, 1, 2, 3]
    assert completed == [0, 1, 2, 3]


def test_unattended_monitor_avoids_zero_count_false_alarms() -> None:
    fail_closed = unattended_monitor.FATAL_PATTERNS["fail_closed"]
    zero_target = unattended_monitor.FATAL_PATTERNS["zero_target"]
    assert fail_closed.search("fail_closed_games=0/24") is None
    assert zero_target.search("zero_target_games=0") is None
    assert fail_closed.search("fail_closed_games=1/24") is not None
    assert zero_target.search("zero_target_games=2") is not None
    assert unattended_monitor.FATAL_PATTERNS["game_timeout"].search(
        "game_timeouts=0"
    ) is None
    assert unattended_monitor.FATAL_PATTERNS["game_timeout"].search(
        "game_timeouts=1"
    ) is not None
    assert unattended_monitor.FATAL_PATTERNS["trust_failure"].search(
        "trust_failures=0"
    ) is None
    assert unattended_monitor.FATAL_PATTERNS["trust_failure"].search(
        "trust_failures=2"
    ) is not None

    process = unattended_monitor._process_snapshot(os.getpid(), False)
    assert process["process_count"] == 1
    assert process["rss_mb"] > 0


def test_unattended_monitor_ignores_remote_pin_digest_soft_drops() -> None:
    digest = unattended_monitor.FATAL_PATTERNS["digest_mismatch"]
    remote_soft_drop = (
        "[remote-farm] WARN dropped endpoint(s) after reload failure: "
        "elmo:8765: reload failed host=elmo "
        "remote_path=/workspace/checkpoint/iter_00001.pt: "
        "{'type': 'reload_ok', 'ok': False, 'error': "
        "\"leaf[1] reload failed: {'type': 'pin', 'ok': False, 'version': 1, "
        "'checkpoint_digest': 'sha256:abc', 'error': "
        "'ValueError: pin digest mismatch: expected sha256:old, got sha256:new'}\"}"
    )
    assert digest.search(remote_soft_drop) is None
    assert digest.search(
        "reload digest mismatch: expected sha256:a, got sha256:b"
    ) is not None
    assert digest.search(
        "initial checkpoint digest mismatch: expected sha256:a, got sha256:b"
    ) is not None
    assert digest.search(
        "leaf response checkpoint digest mismatch: expected sha256:a, got sha256:b"
    ) is not None
    assert digest.search("FAIL-CLOSED digest mismatch on leaf") is not None
