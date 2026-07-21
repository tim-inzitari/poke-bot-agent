"""Guards for candidate gating and serialized weight publication."""

from __future__ import annotations

from pathlib import Path


def test_candidate_gate_precedes_publish_and_next_collect() -> None:
    """A rejected candidate must never reach leaves, remotes, or self-play."""
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
    next_collect_at = text.index("pending_collect = _kick_collect(next_it", start)
    candidate_window = text[start:next_collect_at]
    assert promotion_at < decision_at < publish_at < next_collect_at
    assert "if promoted:" in candidate_window[: publish_at - start]
    assert "replace_existing=False" in text
    assert "shard.unlink()" not in text
    assert "WARN remote reload" not in candidate_window


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
