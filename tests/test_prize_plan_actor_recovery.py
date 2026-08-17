from __future__ import annotations

import argparse
import hashlib
import inspect
from pathlib import Path

import pytest

from scripts.train_pure_rl import (
    _configured_prize_plan_h3_actor_provider,
    _validate_orphan_awr_provider,
    run_full_loop,
)


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _args(tmp_path: Path, *, enabled: bool) -> argparse.Namespace:
    if not enabled:
        return argparse.Namespace(
            prize_plan_h3_cache_receipt=None,
            prize_plan_h3_cache_receipt_sha256="",
            prize_plan_h3_activation_receipt=None,
            prize_plan_h3_activation_receipt_sha256="",
        )
    cache = tmp_path / "cache.json"
    activation = tmp_path / "activation.json"
    cache.write_text("{}\n", encoding="utf-8")
    activation.write_text("{}\n", encoding="utf-8")
    return argparse.Namespace(
        prize_plan_h3_cache_receipt=cache,
        prize_plan_h3_cache_receipt_sha256=_sha(cache),
        prize_plan_h3_activation_receipt=activation,
        prize_plan_h3_activation_receipt_sha256=_sha(activation),
    )


def test_legacy_design_remains_absent_and_rejects_h3_orphan(tmp_path: Path) -> None:
    assert _configured_prize_plan_h3_actor_provider(_args(tmp_path, enabled=False)) is None
    with pytest.raises(RuntimeError, match="unrequested Prize-plan H3"):
        _validate_orphan_awr_provider(
            extra={"awr_advantage_provider": {"actor_activation": True}},
            provenance={},
            expected=None,
        )


def test_h3_orphan_must_match_both_receipt_identities(tmp_path: Path) -> None:
    expected = _configured_prize_plan_h3_actor_provider(
        _args(tmp_path, enabled=True)
    )
    assert expected is not None
    binding = {
        "schema": "poke_bot.alakazam_prize_plan_v2_h3_actor_provider_binding/v1",
        "cache_receipt_sha256": expected["cache_receipt"]["digest"],
        "activation_receipt_sha256": expected["activation_receipt"]["digest"],
    }
    actual = {
        "actor_activation": True,
        "exact_legacy_baseline_computed_in_batch": True,
        "runtime_critic_calls": False,
        "provider_binding": binding,
    }
    _validate_orphan_awr_provider(
        extra={"awr_advantage_provider": actual},
        provenance={"prize_plan_h3_actor_provider": binding},
        expected=expected,
    )
    bad = {**binding, "activation_receipt_sha256": "sha256:" + "0" * 64}
    with pytest.raises(RuntimeError, match="disagrees"):
        _validate_orphan_awr_provider(
            extra={"awr_advantage_provider": {**actual, "provider_binding": bad}},
            provenance={"prize_plan_h3_actor_provider": bad},
            expected=expected,
        )


def test_configured_h3_rejects_declared_digest_drift(tmp_path: Path) -> None:
    args = _args(tmp_path, enabled=True)
    args.prize_plan_h3_cache_receipt_sha256 = "sha256:" + "f" * 64
    with pytest.raises(RuntimeError, match="configured receipt digest mismatch"):
        _configured_prize_plan_h3_actor_provider(args)


def test_full_loop_initializes_h3_provider_before_optimizer_use() -> None:
    source = inspect.getsource(run_full_loop)
    initialization = (
        "prize_plan_h3_actor_provider = "
        "_configured_prize_plan_h3_actor_provider(args)"
    )
    assert source.count(initialization) == 1
    assert source.index(initialization) < source.index("live_h3_mode = bool(")
