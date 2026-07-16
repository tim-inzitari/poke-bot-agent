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


def test_leaf_version_mismatch_adopts_when_digest_matches(monkeypatch) -> None:
    """Reload bumps server version; client must adopt instead of fail-closing."""
    monkeypatch.setattr(batched_infer, "featurize_packets", _fake_features)
    alive = threading.Event()
    alive.set()
    req = queue.Queue()
    resp = queue.Queue()
    shared = {"digest": "sha256:same", "version": 0}
    resp.put(
        {
            "generation": 1,
            "rid": 1,
            "ok": True,
            "values": [(0.5, [1.0])],
            "version": 1,
            "checkpoint_digest": "sha256:same",
        }
    )
    client = RemoteLeafClient(
        0,
        req,
        resp,
        generation=1,
        alive_evt=alive,
        expected_digest=shared,
        expected_version=shared,
        timeout_s=0.1,
    )
    out = client([LeafPacket(obs=object(), your_deck=[1] * 60, root_seat=0)])
    assert out[0].value == 0.5
    assert shared["version"] == 1


def test_leaf_version_bump_n_to_n_plus_1_adopts(monkeypatch) -> None:
    """expected 0→1 then 1→2 without expected-N-got-N+1 fail-closed."""
    monkeypatch.setattr(batched_infer, "featurize_packets", _fake_features)
    alive = threading.Event()
    alive.set()
    req = queue.Queue()
    resp = queue.Queue()
    shared = {"digest": "sha256:same", "version": 0, "pinned": []}
    for ver in (1, 2):
        resp.put(
            {
                "generation": 1,
                "rid": ver,
                "ok": True,
                "values": [(0.1 * ver, [1.0])],
                "version": ver,
                "checkpoint_digest": "sha256:same",
            }
        )
    client = RemoteLeafClient(
        0,
        req,
        resp,
        generation=1,
        alive_evt=alive,
        expected_digest=shared,
        expected_version=shared,
        timeout_s=0.1,
    )
    assert client([LeafPacket(obs=object(), your_deck=[1] * 60, root_seat=0)])[0].value == 0.1
    assert shared["version"] == 1
    assert client([LeafPacket(obs=object(), your_deck=[1] * 60, root_seat=0)])[0].value == 0.2
    assert shared["version"] == 2


def test_leaf_accepts_pinned_secondary_digest(monkeypatch) -> None:
    monkeypatch.setattr(batched_infer, "featurize_packets", _fake_features)
    alive = threading.Event()
    alive.set()
    req = queue.Queue()
    resp = queue.Queue()
    shared = {
        "digest": "sha256:primary",
        "version": 3,
        "pinned": ["sha256:secondary"],
    }
    resp.put(
        {
            "generation": 1,
            "rid": 1,
            "ok": True,
            "values": [(0.7, [1.0])],
            "version": 3,
            "checkpoint_digest": "sha256:secondary",
        }
    )
    client = RemoteLeafClient(
        0,
        req,
        resp,
        generation=1,
        alive_evt=alive,
        expected_digest=shared,
        expected_version=shared,
        timeout_s=0.1,
    )
    out = client([LeafPacket(obs=object(), your_deck=[1] * 60, root_seat=0)])
    assert out[0].value == 0.7


def test_leaf_version_mismatch_hard_fails_when_digest_differs(monkeypatch) -> None:
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
            "values": [(0.5, [1.0])],
            "version": 1,
            "checkpoint_digest": "sha256:other",
        }
    )
    client = RemoteLeafClient(
        0,
        req,
        resp,
        generation=1,
        alive_evt=alive,
        expected_digest="sha256:mine",
        expected_version=0,
        timeout_s=0.1,
    )
    with pytest.raises(batched_infer.RemoteLeafError, match="digest mismatch"):
        client([LeafPacket(obs=object(), your_deck=[1] * 60, root_seat=0)])


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
    # Per-game PolicyAgent / timeout lines must NOT kill overnight runs.
    assert fail_closed.search(
        "[PolicyAgent] FAIL-CLOSED (no search/leaf): TimeoutError: "
        "promotion game exceeded 600s"
    ) is None
    assert "game_timeout" not in unattended_monitor.FATAL_PATTERNS
    assert "trust_failure" not in unattended_monitor.FATAL_PATTERNS
    assert "insufficient_sims" not in unattended_monitor.FATAL_PATTERNS
    assert unattended_monitor.FATAL_PATTERNS["fatal_gate"].search(
        "FATAL HEALTH GATE: iter soft"
    ) is None
    assert unattended_monitor.FATAL_PATTERNS["fatal_gate"].search(
        "ABORT: leaf server dead"
    ) is not None

    process = unattended_monitor._process_snapshot(os.getpid(), False)
    assert process["process_count"] == 1
    assert process["rss_mb"] > 0
