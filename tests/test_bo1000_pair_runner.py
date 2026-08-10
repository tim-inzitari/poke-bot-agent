from __future__ import annotations

import hashlib
import json
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from poke_bot.recursive_turn_planner.bo1000_evaluation import (
    CONTROL_ARM,
    MCTS_ARM,
    R207_FROZEN_BUNDLE_SHA256,
    R207_FROZEN_CHECKPOINT_SHA256,
    build_bo1000_schedule,
)
from poke_bot.recursive_turn_planner.bo1000_pair_runner import (
    DEFAULT_EVALUATION_ID,
    BO1000PairRunnerError,
    HostLocalBO1000PairController,
    HostLocalSubprocessGameRunner,
    R207_R195_MATCHUP_TREE_SHA256,
    build_bo1000_pair_envelope,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode('utf-8')).hexdigest()}"


CHECKPOINT = R207_FROZEN_CHECKPOINT_SHA256
BUNDLE = R207_FROZEN_BUNDLE_SHA256


_CHILD_RECEIPT = r"""
import json
from pathlib import Path
import os
import sys

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
marker = Path(sys.argv[2])
marker.parent.mkdir(parents=True, exist_ok=True)
with marker.open("a", encoding="utf-8") as stream:
    stream.write(f"{os.getpid()}:{request['game']['game_index']}\n")
game = request["game"]
receipt = {
    "game_nonce_sha256": game["game_nonce_sha256"],
    "pair_id": game["pair_id"],
    "game_index": game["game_index"],
    "mcts_seat": game["mcts_seat"],
    "no_rtp_seat": game["no_rtp_seat"],
    "pair_rng_snapshot_sha256": request["rng"]["pair_rng_snapshot_sha256"],
    "deck_order_rng_sha256": request["rng"]["deck_order_rng_sha256"],
    "checkpoint_sha256": request["frozen_model"]["checkpoint_sha256"],
    "bundle_sha256": request["frozen_model"]["bundle_sha256"],
    "terminal_status": "completed",
    "winner_seat": game["mcts_seat"],
    "illegal_action_count": 0,
    "forfeit_count": 0,
    "crash_count": 0,
    "timeout_count": 0,
    "mcts_turns": [],
}
print(json.dumps(receipt, sort_keys=True))
"""


_BAD_CHILD = 'print(\'{"not": "a receipt"}\')'


def _envelope(
    *,
    pair_index: int = 7,
    checkpoint: str = CHECKPOINT,
    pair_rng_snapshot: str | None = None,
    deck_order_rng: str | None = None,
    evaluation_id: str = DEFAULT_EVALUATION_ID,
    experimental_arm: str = MCTS_ARM,
    control_arm: str = CONTROL_ARM,
):
    schedule = build_bo1000_schedule(_digest("schedule-seed"))
    return build_bo1000_pair_envelope(
        schedule,
        pair_index=pair_index,
        checkpoint_sha256=checkpoint,
        bundle_sha256=BUNDLE,
        pair_rng_snapshot_sha256=pair_rng_snapshot or _digest(f"pair-rng:{pair_index}"),
        deck_order_rng_sha256=deck_order_rng or _digest(f"deck-rng:{pair_index}"),
        execution_host="pair-test-host",
        evaluation_id=evaluation_id,
        experimental_arm=experimental_arm,
        control_arm=control_arm,
    )


def test_pair_envelope_is_exactly_one_seat_swapped_pair_with_stable_nonce() -> None:
    envelope = _envelope()

    assert envelope.pair_index == 7
    assert (
        envelope.evaluation_id
        == "alakazam-r207-simulator-backed-chance-aware-inter-turn-mcts-bo1000"
    )
    assert [
        (game.game_index, game.mcts_seat, game.no_rtp_seat)
        for game in envelope.game_specs
    ] == [
        (0, 0, 1),
        (1, 1, 0),
    ]
    assert len({game.pair_nonce_sha256 for game in envelope.game_specs}) == 1
    assert envelope.idempotency_key_sha256 == _envelope().idempotency_key_sha256
    assert (
        envelope.idempotency_key_sha256
        != _envelope(
            pair_rng_snapshot=_digest("different-pair-rng")
        ).idempotency_key_sha256
    )
    assert (
        envelope.idempotency_key_sha256
        != _envelope(
            deck_order_rng=_digest("different-deck-rng")
        ).idempotency_key_sha256
    )
    assert envelope.as_payload()["execution"] == {
        "scope": "host_local_fresh_process",
        "host": "pair-test-host",
        "remote_dispatch_authorized": False,
        "fresh_process_required_per_game": True,
    }
    assert envelope.as_payload()["policy_transforms"] == {
        "frozen_base_policy": "r195_identity",
        "guide2vec_enabled": False,
        "guide_logit_transform_enabled": False,
        "guide_linear_transform_enabled": False,
    }
    assert envelope.as_payload()["matchup_adapter"] == {
        "matchup_tree_sha256": R207_R195_MATCHUP_TREE_SHA256,
        "enabled": True,
        "trained": True,
        "frozen": True,
        "same_exact_runtime_required_for_both_arms": True,
    }


def test_pair_envelope_rejects_non_r207_identity_or_arm_labels() -> None:
    with pytest.raises(BO1000PairRunnerError, match="exact r207 evaluation"):
        _envelope(evaluation_id="alakazam-r205-chance-aware-inter-turn-mcts-bo1000")
    with pytest.raises(BO1000PairRunnerError, match="exact r207 arm identities"):
        _envelope(experimental_arm="legacy-experimental")
    with pytest.raises(BO1000PairRunnerError, match="exact r207 arm identities"):
        _envelope(control_arm="legacy-control")
    with pytest.raises(BO1000PairRunnerError, match="pair_id must bind"):
        replace(_envelope(), pair_id="r207-pair-000007-tampered")
    with pytest.raises(BO1000PairRunnerError, match="original r195 no-RTP checkpoint"):
        _envelope(checkpoint=_digest("later-guide-adjusted-checkpoint"))


def test_pair_runner_strips_inherited_guide_transforms_and_forbids_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    envelope = _envelope()
    controller = HostLocalBO1000PairController(
        tmp_path / "out", execution_host="pair-test-host"
    )
    marker = tmp_path / "child-env.json"
    child = r"""
import json, os, sys
from pathlib import Path
request = json.loads(Path(sys.argv[1]).read_text())
Path(sys.argv[2]).write_text(json.dumps({
    "guide": os.environ.get("POKEBOT_CURRENT_DECK_GUIDE"),
    "guide2vec": os.environ["BO1000_GUIDE2VEC_ENABLED"],
    "guide_logit": os.environ["BO1000_GUIDE_LOGIT_TRANSFORM_ENABLED"],
    "guide_linear": os.environ["BO1000_GUIDE_LINEAR_TRANSFORM_ENABLED"],
    "base_transform": os.environ["BO1000_BASE_POLICY_TRANSFORM"],
    "matchup_adapter": os.environ["BO1000_MATCHUP_ADAPTER_ENABLED"],
    "matchup_tree": os.environ["BO1000_MATCHUP_TREE_SHA256"],
}))
game = request["game"]
print(json.dumps({
    "game_nonce_sha256": game["game_nonce_sha256"], "pair_id": game["pair_id"],
    "game_index": game["game_index"], "mcts_seat": game["mcts_seat"],
    "no_rtp_seat": game["no_rtp_seat"],
    "pair_rng_snapshot_sha256": request["rng"]["pair_rng_snapshot_sha256"],
    "deck_order_rng_sha256": request["rng"]["deck_order_rng_sha256"],
    "checkpoint_sha256": request["frozen_model"]["checkpoint_sha256"],
    "bundle_sha256": request["frozen_model"]["bundle_sha256"],
    "terminal_status": "completed", "winner_seat": game["mcts_seat"],
    "illegal_action_count": 0, "forfeit_count": 0, "crash_count": 0,
    "timeout_count": 0, "mcts_turns": [],
}))
"""
    monkeypatch.setenv("POKEBOT_CURRENT_DECK_GUIDE", "alakazam")
    runner = HostLocalSubprocessGameRunner(
        lambda request_path, _spec: [
            sys.executable,
            "-c",
            child,
            str(request_path),
            str(marker),
        ]
    )
    controller.run_pair(envelope, runner)
    observed = json.loads(marker.read_text())
    assert observed == {
        "guide": None,
        "guide2vec": "0",
        "guide_logit": "0",
        "guide_linear": "0",
        "base_transform": "frozen_r195_identity",
        "matchup_adapter": "1",
        "matchup_tree": R207_R195_MATCHUP_TREE_SHA256,
    }
    with pytest.raises(BO1000PairRunnerError, match="guide transforms"):
        HostLocalSubprocessGameRunner(
            lambda _path, _spec: [sys.executable],
            extra_env={"BO1000_GUIDE2VEC_ENABLED": "1"},
        )


def test_pair_runner_spawns_two_local_fresh_process_games_and_is_idempotent(
    tmp_path: Path,
) -> None:
    envelope = _envelope()
    controller = HostLocalBO1000PairController(
        tmp_path / "out", execution_host="pair-test-host"
    )
    marker = tmp_path / "child-pids.txt"
    calls: list[tuple[Path, int]] = []

    def command(request_path: Path, spec) -> list[str]:
        calls.append((request_path, spec.game_index))
        return [sys.executable, "-c", _CHILD_RECEIPT, str(request_path), str(marker)]

    runner = HostLocalSubprocessGameRunner(command)
    assert controller.status(envelope).status == "pending"
    result = controller.run_pair(envelope, runner)

    assert result.status.status == "complete"
    assert result.status.observed_game_receipts == 2
    assert [receipt.mcts_seat for receipt in result.game_receipts] == [0, 1]
    assert len(calls) == 2
    children = marker.read_text(encoding="utf-8").splitlines()
    assert len(children) == 2
    assert len({line.rsplit(":", 1)[0] for line in children}) == 2
    assert {line.rsplit(":", 1)[1] for line in children} == {"0", "1"}

    pair_dir = controller.pair_directory(envelope)
    envelope_payload = json.loads((pair_dir / "pair-envelope.json").read_text())
    assert envelope_payload["execution"]["scope"] == "host_local_fresh_process"
    assert envelope_payload["execution"]["remote_dispatch_authorized"] is False
    request_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((pair_dir / "game-requests").glob("*.json"))
    ]
    assert [payload["game"]["mcts_seat"] for payload in request_payloads] == [0, 1]
    assert all(
        payload["execution"]["host"] == "pair-test-host" for payload in request_payloads
    )
    assert all(
        payload["authority"]["remote_dispatch_authorized"] is False
        for payload in request_payloads
    )
    assert stat.S_IMODE((pair_dir / "pair-envelope.json").stat().st_mode) == 0o444
    assert stat.S_IMODE((pair_dir / "pair-receipt.json").stat().st_mode) == 0o444

    second = controller.run_pair(envelope, runner)
    assert second.status == result.status
    assert len(calls) == 2
    assert marker.read_text(encoding="utf-8").splitlines() == children


def test_bad_local_result_fails_closed_for_both_seats_and_stays_terminal(
    tmp_path: Path,
) -> None:
    envelope = _envelope()
    controller = HostLocalBO1000PairController(
        tmp_path / "out", execution_host="pair-test-host"
    )
    calls: list[int] = []

    def command(request_path: Path, spec) -> list[str]:
        del request_path
        calls.append(spec.game_index)
        return [sys.executable, "-c", _BAD_CHILD]

    runner = HostLocalSubprocessGameRunner(command)
    result = controller.run_pair(envelope, runner)

    assert result.status.status == "failed_closed"
    assert calls == [0, 1]
    assert all(
        receipt.terminal_status == "failed_closed" for receipt in result.game_receipts
    )
    assert [receipt.crash_count for receipt in result.game_receipts] == [1, 1]
    assert controller.run_pair(envelope, runner).status.status == "failed_closed"
    assert calls == [0, 1]


def test_mismatched_existing_envelope_fails_closed_instead_of_overwriting(
    tmp_path: Path,
) -> None:
    controller = HostLocalBO1000PairController(
        tmp_path / "out", execution_host="pair-test-host"
    )
    original = _envelope()
    controller.materialize_envelope(original)
    changed = _envelope()
    object.__setattr__(changed, "checkpoint_sha256", _digest("different-checkpoint"))

    with pytest.raises(BO1000PairRunnerError, match="differs|does not match"):
        controller.materialize_envelope(changed)
