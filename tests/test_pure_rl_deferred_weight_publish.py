"""Guards for candidate gating and serialized weight publication."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest


def _load_train_pure_rl():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    spec = importlib.util.spec_from_file_location("train_pure_rl_gate_runtime", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_candidate_gate_precedes_publish_and_next_collect() -> None:
    """A candidate reaches collection only after its safety-carry decision."""
    src = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    text = src.read_text(encoding="utf-8")
    start = text.index("candidate = CheckpointIdentity.from_path(candidate_path)")
    promotion_at = text.index(
        "promotion_report, promotion_rows = _promotion_eval(", start
    )
    decision_at = text.index(
        'promoted = bool(promotion_report.get("passed"))', start
    )
    publish_at = text.index("weight_gate_proof = _hard_gate_publish_weights(", start)
    heldout_at = text.index("heldout_rows, heldout_audit = _heldout_eval(", start)
    collection_publish_at = text.index(
        "collection_publish_proof = _hard_gate_publish_weights(", start
    )
    next_collect_at = text.index(
        "pending_collect = _kick_collect(\n                    next_it,", start
    )
    candidate_window = text[start:next_collect_at]
    assert promotion_at < decision_at < publish_at < next_collect_at
    assert heldout_at < collection_publish_at < next_collect_at
    assert "if promoted:" in candidate_window[: publish_at - start]
    assert "ckpt=learner_after.path" in candidate_window
    assert "digest=learner_after.digest" in candidate_window
    next_collect_window = text[next_collect_at : next_collect_at + 300]
    assert "learner_after.path," in next_collect_window
    assert "learner_after.digest," in next_collect_window
    assert "replace_existing=False" in text
    assert "shard.unlink()" not in text
    assert "WARN remote reload" not in candidate_window


def test_promotion_retries_only_exact_invalid_game_identities(monkeypatch) -> None:
    mod = _load_train_pure_rl()
    calls: list[dict[str, object]] = []

    class FakeWorkerPool:
        def __init__(self, *, num_workers: int) -> None:
            self.num_workers = int(num_workers)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def imap_unordered(self, _fn, jobs):
            wave = list(jobs)
            calls.append(
                {
                    "workers": self.num_workers,
                    "seeds": [int(job["seed"]) for job in wave],
                }
            )
            first_wave = len(calls) == 1
            for job in reversed(wave):
                failed = first_wave and int(job["seed"]) == 102
                yield {
                    "seed": int(job["seed"]),
                    "candidate_seat": int(job["candidate_seat"]),
                    "cluster_id": int(job["cluster_id"]),
                    "valid": not failed,
                    "winner": (
                        2 if failed else int(job["candidate_seat"])
                    ),
                    "error": "transient worker load failure" if failed else None,
                }

    monkeypatch.setattr("poke_bot.worker_pool.WorkerPool", FakeWorkerPool)
    monkeypatch.setattr(
        "poke_bot.remote_sim_jobs.remote_promotion_job", lambda job: job
    )
    candidate = SimpleNamespace(path="candidate.pt", digest="sha256:candidate")
    incumbent = SimpleNamespace(path="incumbent.pt", digest="sha256:incumbent")
    candidate.as_dict = lambda: {
        "path": candidate.path,
        "digest": candidate.digest,
    }
    incumbent.as_dict = lambda: {
        "path": incumbent.path,
        "digest": incumbent.digest,
    }

    report, rows = mod._promotion_eval(
        candidate=candidate,
        incumbent=incumbent,
        decks=[("alakazam", list(range(60)))],
        n_games=4,
        n_workers=4,
        threshold=0.0,
        confidence=0.90,
        bootstrap_resamples=100,
        seed=100,
        game_timeout_s=600,
        model_generation=1,
    )

    assert calls == [
        {"workers": 4, "seeds": [100, 101, 102, 103]},
        {"workers": 1, "seeds": [102]},
    ]
    assert [row["seed"] for row in rows] == [100, 101, 102, 103]
    assert all(row["valid"] for row in rows)
    assert report["passed"] is True
    recovery = report["transport_recovery"]
    assert recovery["valid_games_retried"] == 0
    assert recovery["waves"][0]["recovered_games"] == 1
    assert recovery["waves"][0]["remaining_invalid_games"] == 0


def test_failed_promotion_restores_runtime_compatible_behavior_identity() -> None:
    """Temporary evaluation must not roll back to a pre-adapter champion."""
    src = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    text = src.read_text(encoding="utf-8")
    start = text.index("heldout_rows, heldout_audit = _heldout_eval(")
    end = text.index("gate = aggregate_heldout_wr(", start)
    rollback = text[start:end]

    assert "if not promoted:" in rollback
    assert "ckpt=behavior_before.path" in rollback
    assert "digest=behavior_before.digest" in rollback
    assert "ckpt=incumbent_before.path" not in rollback


def test_boot_collection_uses_continuous_learner_not_rollback_champion() -> None:
    src = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    text = src.read_text(encoding="utf-8")
    start = text.index("leaf = _LeafFarm()")
    first_collect_match = re.search(
        r"pending_collect\s*=\s*_kick_collect\(\s*start_iteration,",
        text[start:],
    )
    assert first_collect_match is not None
    first_collect = start + first_collect_match.start()
    boot = text[start : first_collect + 300]
    assert "ckpt=learner_identity.path" in boot
    assert "digest=learner_identity.digest" in boot
    assert "learner_identity.path,\n            learner_identity.digest" in boot


def test_continuous_learner_selection_does_not_reset_to_heldout_champion() -> None:
    src = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    text = src.read_text(encoding="utf-8")
    start = text.index("carry_candidate, learner_carry_reason = (")
    end = text.index('promotion_report["continuous_learner"]', start)
    selection = text[start:end]

    assert "continuous_learner_carry_decision(" in selection
    assert "heldout_audit_ok=bool(heldout_audit.get" in selection
    assert "learner_after = candidate" in selection
    assert "learner_after = learner_before" in selection
    assert "_exact_regression_rollback_identity(" in selection
    assert "learner_after = heldout_champion_identity" not in selection
    assert "learner_after = prior_heldout_champion_identity" not in selection


def test_rejected_candidate_restores_exact_pre_eval_behavior_identity() -> None:
    src = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    text = src.read_text(encoding="utf-8")
    start = text.index("finally:\n                if not promoted:")
    end = text.index("            gate = aggregate_heldout_wr(", start)
    rollback = text[start:end]

    assert "ckpt=behavior_before.path" in rollback
    assert "digest=behavior_before.digest" in rollback
    assert "ckpt=incumbent_before.path" not in rollback
    assert "digest=incumbent_before.digest" not in rollback


def test_leaf_reload_deadline_is_global_and_preserves_old_identity(
    monkeypatch,
) -> None:
    mod = _load_train_pure_rl()
    leaf = mod._LeafFarm()
    control_messages: list[dict] = []

    class ControlQueue:
        def put(self, value, *, timeout):
            assert timeout == pytest.approx(5.0)
            control_messages.append(value)

    class FirstMissingStatus:
        def get(self, *, timeout):
            assert timeout == pytest.approx(240.0)
            raise TimeoutError("missing first acknowledgement")

    class MustNotWaitAgain:
        def get(self, *, timeout):
            raise AssertionError(
                f"second leaf received a serialized timeout: {timeout}"
            )

    leaf.ctrl_qs = [ControlQueue(), ControlQueue()]
    leaf.status_qs = [FirstMissingStatus(), MustNotWaitAgain()]
    leaf.digest = "sha256:" + "a" * 64
    leaf.version = 7
    leaf.remote_channel = {
        "expected_digest": leaf.digest,
        "expected_version": leaf.version,
    }
    clock = iter((0.0, 0.0, 0.0, 0.0, 240.1))
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(clock))

    with pytest.raises(RuntimeError, match="global reload deadline exceeded"):
        leaf.reload(Path("candidate.pt"), "sha256:" + "b" * 64)

    assert len(control_messages) == 2
    assert leaf.digest == "sha256:" + "a" * 64
    assert leaf.version == 7
    assert leaf.remote_channel["expected_digest"] == leaf.digest
    assert leaf.remote_channel["expected_version"] == leaf.version


def test_canonical_production_learner_has_one_safety_carry_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    assert not (root / "deploy/staging/train_pure_rl_v11.py").exists()
    src = root / "scripts/train_pure_rl.py"
    text = src.read_text(encoding="utf-8")
    start = text.index("carry_candidate, learner_carry_reason = (")
    end = text.index('promotion_report["continuous_learner"]', start)
    selection = text[start:end]

    assert "_continuous_learner_carry_decision(" in selection
    assert "heldout_audit_ok=bool(heldout_audit.get" in selection
    assert "learner_after = candidate" in selection
    assert "learner_after = learner_before" in selection
    assert "learner_after = prior_heldout_champion_identity" not in selection
    assert "continuous_learner_safety_carry" in text


def test_hard_gate_helper_and_real_iter_wiring() -> None:
    """Between-iter hard gate + tqdm iter must be wired (not hardcoded 0)."""
    src = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    text = src.read_text(encoding="utf-8")
    assert "def _hard_gate_publish_weights(" in text
    assert "class BetweenIterSyncError" in text
    assert "iteration=int(it)" in text
    # Progress bars inside _collect_wave must take the iteration kwarg.
    assert "iteration=int(iteration)" in text
    # Soft boot WARN path replaced by hard gate.
    assert "WARN remote reload/pin" not in text


def test_hard_gate_refreshes_stale_present_client_before_reload(tmp_path: Path) -> None:
    """A retained dead socket must be reconnected, not treated as alive."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    spec = importlib.util.spec_from_file_location("train_pure_rl_gate", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    digest = "sha256:" + "a" * 64
    calls: list[str] = []

    class StalePresentClient:
        host = "worker.local"
        port = 8765

        def __init__(self) -> None:
            self.connected = False
            self.published_digest = "sha256:" + "b" * 64

        def reconnect(self):
            calls.append("reconnect")
            self.connected = True
            return SimpleNamespace(checkpoint_digest=self.published_digest)

        def reload_checkpoint(self, _path, *, digest, version):
            calls.append("reload")
            assert self.connected
            self.published_digest = digest
            return {
                "ok": True,
                "checkpoint_digest": digest,
                "version": version,
            }

        def pin_checkpoint(self, _path, *, digest):
            calls.append("pin")
            assert self.connected
            return {"ok": True, "checkpoint_digest": digest}

    client = StalePresentClient()
    farm = SimpleNamespace(
        clients=[client],
        _reconnect_missing=lambda: calls.append("farm_reconnect_missing"),
    )
    leaf = SimpleNamespace(remote_channel=None)
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")

    proof = mod._hard_gate_publish_weights(
        leaf=leaf,
        remote_farm=farm,
        ckpt=checkpoint,
        digest=digest,
        version=11,
        required_endpoints=["worker.local:8765"],
    )

    assert proof["remote_ok"] is True
    assert calls == [
        "farm_reconnect_missing",
        "reconnect",
        "reload",
        "pin",
        "reconnect",
    ]


def test_hard_gate_skips_redundant_reload_from_exact_leaf_health(
    tmp_path: Path,
) -> None:
    """A fresh exact leaf-and-pin proof makes the publish idempotent."""
    mod = _load_train_pure_rl()
    digest = "sha256:" + "a" * 64
    calls: list[str] = []

    class ExactClient:
        host = "worker.local"
        port = 8765

        def __init__(self):
            self.reconnects = 0

        def reconnect(self):
            calls.append("reconnect")
            self.reconnects += 1
            # A first hello may land during leaf recycle and advertise no
            # primary identity. The following complete health proof is newer;
            # the final hello still has to confirm the exact digest.
            return SimpleNamespace(
                checkpoint_digest=digest if self.reconnects > 1 else None
            )

        def health(self):
            calls.append("health")
            return {
                "ok": True,
                "controller_healthy": True,
                "leaf_alive": True,
                "leaf_identity_ok": True,
                "checkpoint_digest": digest,
                "checkpoint_version": 60,
                "pinned_digests": [digest],
                "leaves": [
                    {"healthy": True, "checkpoint_digest": digest},
                    {"healthy": True, "checkpoint_digest": digest},
                ],
            }

        def reload_checkpoint(self, *_args, **_kwargs):
            raise AssertionError("exact health must bypass reload")

        def pin_checkpoint(self, *_args, **_kwargs):
            raise AssertionError("exact health must bypass pin")

    client = ExactClient()
    farm = SimpleNamespace(
        clients=[client],
        _reconnect_missing=lambda: calls.append("farm_reconnect_missing"),
    )
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")

    proof = mod._hard_gate_publish_weights(
        leaf=SimpleNamespace(remote_channel=None),
        remote_farm=farm,
        ckpt=checkpoint,
        digest=digest,
        version=11,
        required_endpoints=["worker.local:8765"],
    )

    assert proof["remote_ok"] is True
    assert proof["remote_endpoints"][0]["reload_skipped_exact_health"] is True
    assert calls == [
        "farm_reconnect_missing",
        "reconnect",
        "health",
        "reconnect",
    ]


def test_weight_publish_refuses_shadow_only_remote_when_runtime_is_required(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_train_pure_rl()
    digest = "sha256:" + "c" * 64

    class ShadowOnlyClient:
        host = "worker.local"
        port = 8765

        def reconnect(self):
            return SimpleNamespace(
                checkpoint_digest=digest,
                matchup_runtime=None,
            )

        def reload_checkpoint(self, _path, *, digest, version):
            return {
                "ok": True,
                "checkpoint_digest": digest,
                "version": version,
            }

        def pin_checkpoint(self, _path, *, digest):
            return {"ok": True, "checkpoint_digest": digest}

    farm = SimpleNamespace(
        clients=[ShadowOnlyClient()],
        _reconnect_missing=lambda: None,
    )
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")
    runtime_tree = tmp_path / "runtime-tree.json"
    runtime_tree.write_text(
        json.dumps(
            {
                "runtime_contract": {
                    "accepted_archetype_ids": ["hops-trevenant"]
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "1")
    monkeypatch.setenv(
        "POKEBOT_PUBLIC_MATCHUP_TREE_PATH", str(runtime_tree)
    )

    with pytest.raises(
        mod.BetweenIterSyncError,
        match="matchup runtime is not active",
    ):
        mod._hard_gate_publish_weights(
            leaf=SimpleNamespace(remote_channel=None),
            remote_farm=farm,
            ckpt=checkpoint,
            digest=digest,
            version=11,
            required_endpoints=["worker.local:8765"],
        )


def test_weight_publish_probes_pool_child_and_rotates_on_stale_tree(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_train_pure_rl()
    digest = "sha256:" + "c" * 64
    runtime_tree = tmp_path / "runtime-tree.json"
    runtime_tree.write_text(
        json.dumps(
            {
                "runtime_contract": {
                    "accepted_archetype_ids": [
                        "alakazam",
                        "hops-trevenant",
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    tree_digest = mod._sha256_file(runtime_tree)
    rotations: list[str] = []

    class StaleChildClient:
        host = "worker.local"
        port = 8765

        def reconnect(self):
            return SimpleNamespace(
                checkpoint_digest=digest,
                matchup_runtime={
                    "checkpoint_digest": digest,
                    "tree_digest": tree_digest,
                    "accepted_archetype_ids": [
                        "alakazam",
                        "hops-trevenant",
                    ],
                    "continuous_reevaluation": True,
                    "one_route_per_decision": True,
                    "unknown_route_exact_bypass": True,
                },
            )

        def reload_checkpoint(self, _path, *, digest, version):
            return {
                "ok": True,
                "checkpoint_digest": digest,
                "version": version,
            }

        def pin_checkpoint(self, _path, *, digest):
            return {"ok": True, "checkpoint_digest": digest}

        def submit_job(self, _job, *, kind):
            assert kind == "runtime_probe"
            return {
                "runtime_probe": {
                    "runtime_enabled": True,
                    "tree_digest": "sha256:" + "d" * 64,
                    "accepted_archetype_ids": [
                        "alakazam",
                        "hops-trevenant",
                        "walrein",
                    ],
                }
            }

        def request_rotation(self, reason):
            rotations.append(reason)
            return {"ok": True, "rotation_scheduled": True}

    monkeypatch.setenv("POKEBOT_MATCHUP_ADAPTER_RUNTIME", "1")
    monkeypatch.setenv(
        "POKEBOT_PUBLIC_MATCHUP_TREE_PATH", str(runtime_tree)
    )
    farm = SimpleNamespace(
        clients=[StaleChildClient()],
        _reconnect_missing=lambda: None,
    )
    checkpoint = tmp_path / "candidate.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(
        mod.BetweenIterSyncError,
        match="simulator-child matchup runtime differs",
    ):
        mod._hard_gate_publish_weights(
            leaf=SimpleNamespace(remote_channel=None),
            remote_farm=farm,
            ckpt=checkpoint,
            digest=digest,
            version=11,
            required_endpoints=["worker.local:8765"],
        )
    assert len(rotations) == 1


def test_tqdm_progress_desc_uses_real_iteration() -> None:
    """Bar desc must show the passed iteration (not stuck at 0)."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    spec = importlib.util.spec_from_file_location("train_pure_rl_iter_bar", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    prog = mod._TqdmProgress(
        stage="collect:self_play",
        iteration=3,
        total=10,
        remotes=30,
        inplace=False,
        mininterval=60.0,
    )
    try:
        assert prog.iteration == 3
        assert "iter=3" in prog._bar.desc
    finally:
        prog.close()
