"""Guards for candidate gating and serialized weight publication."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import SimpleNamespace


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
    assert "learner_after = prior_heldout_champion_identity" not in selection


def test_staged_production_learner_has_same_safety_carry_contract() -> None:
    src = (
        Path(__file__).resolve().parents[1]
        / "deploy/staging/train_pure_rl_v11.py"
    )
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
