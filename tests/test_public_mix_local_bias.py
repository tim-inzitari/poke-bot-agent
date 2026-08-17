"""Public mix (baseline_jobs vs public/recent-self roster) local-bias routing.

Public mix finishes fast on local workers, so it defaults to local-only
dispatch (``PURE_RL_PUBLIC_MIX_LOCAL_ONLY=1``) — remotes stay free for the
much larger self-play wave. Self-play's own local/remote split
(``PURE_RL_REBALANCE_MIN_LOCAL_FRAC``) must stay untouched.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from poke_bot.remote_jobs import RemoteWorkerInfo


def _load_train_pure_rl():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_pure_rl.py"
    spec = importlib.util.spec_from_file_location("train_pure_rl_public_mix", path)
    assert spec is not None and spec.loader is not None
    sys.modules.pop("train_pure_rl_public_mix", None)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def mod():
    return _load_train_pure_rl()


def _fake_remote_farm(*, workers: int = 20, max_workers: int = 40) -> types.SimpleNamespace:
    info = RemoteWorkerInfo(
        endpoint="elmo:8765",
        workers=workers,
        leaf_servers=1,
        gpu_name="fake",
        device="cpu",
        checkpoint_digest=None,
        hostname="elmo",
        job_kinds=("self_play", "play"),
        capabilities=(
            "greedy_play_v1",
            "active_checkpoint_job_barrier_v1",
            "play_result_contract_v1",
            "portable_baseline_spec_v1",
        ),
        max_workers=max_workers,
        default_workers=workers,
    )
    client = types.SimpleNamespace(
        host="elmo",
        port=8765,
        endpoint="elmo:8765",
        info=info,
    )
    return types.SimpleNamespace(clients=[client], total_workers=workers)


def test_public_mix_local_only_default_true(mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PURE_RL_PUBLIC_MIX_LOCAL_ONLY", raising=False)
    assert mod._public_mix_local_only() is True


@pytest.mark.parametrize("raw", ["0", "false", "No", "off"])
def test_public_mix_local_only_env_disable(
    mod, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("PURE_RL_PUBLIC_MIX_LOCAL_ONLY", raw)
    assert mod._public_mix_local_only() is False


@pytest.mark.parametrize("raw", ["1", "true", "Yes", "on"])
def test_public_mix_local_only_env_enable(
    mod, monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    monkeypatch.setenv("PURE_RL_PUBLIC_MIX_LOCAL_ONLY", raw)
    assert mod._public_mix_local_only() is True


def test_public_mix_min_local_frac_default(mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC", raising=False)
    assert mod._public_mix_min_local_frac() == pytest.approx(0.95)


def test_public_mix_min_local_frac_clamped(mod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC", "1.50")
    assert mod._public_mix_min_local_frac() == pytest.approx(0.95)
    monkeypatch.setenv("PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC", "0.0")
    assert mod._public_mix_min_local_frac() == pytest.approx(0.05)


def test_public_mix_defaults_to_local_only_dispatch(
    mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """kind="play" (public mix) routes fully local by default: remote_cap=0."""
    monkeypatch.delenv("PURE_RL_PUBLIC_MIX_LOCAL_ONLY", raising=False)
    farm = _fake_remote_farm()
    local_slots, remote_cap, weight_bits = mod._remote_dispatch_slots(
        remote_farm=farm, scheduler=None, baseline_workers=32, kind="play"
    )
    assert remote_cap == 0
    assert local_slots == 32
    assert "local_only" in weight_bits


def test_self_play_dispatch_untouched_by_public_mix_default(
    mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """kind="self_play" must ignore PURE_RL_PUBLIC_MIX_LOCAL_ONLY entirely."""
    monkeypatch.delenv("PURE_RL_PUBLIC_MIX_LOCAL_ONLY", raising=False)
    monkeypatch.delenv("PURE_RL_REBALANCE_MIN_LOCAL_FRAC", raising=False)
    farm = _fake_remote_farm(workers=20, max_workers=40)
    local_slots, remote_cap, _weight_bits = mod._remote_dispatch_slots(
        remote_farm=farm, scheduler=None, baseline_workers=32, kind="self_play"
    )
    assert remote_cap > 0
    # min_local_frac=0.40 default -> local_slots >= remote_cap * 0.40/0.60.
    assert local_slots >= 32


def test_public_mix_local_only_disabled_falls_back_to_heavy_local_frac(
    mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabling local-only still biases public mix far harder than self-play."""
    monkeypatch.setenv("PURE_RL_PUBLIC_MIX_LOCAL_ONLY", "0")
    monkeypatch.setenv("PURE_RL_PUBLIC_MIX_MIN_LOCAL_FRAC", "0.95")
    farm = _fake_remote_farm(workers=20, max_workers=40)
    local_slots, remote_cap, _weight_bits = mod._remote_dispatch_slots(
        remote_farm=farm, scheduler=None, baseline_workers=32, kind="play"
    )
    assert remote_cap > 0
    # 0.95 local frac -> local_slots >= remote_cap * 0.95/0.05 (heavily local).
    assert local_slots >= remote_cap * 18


def test_formal_heldout_adds_remotes_without_inflating_local_pool(
    mod, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PURE_RL_PUBLIC_MIX_LOCAL_ONLY", raising=False)
    farm = _fake_remote_farm(workers=20, max_workers=40)
    local_slots, remote_cap, weight_bits = mod._remote_dispatch_slots(
        remote_farm=farm,
        scheduler=None,
        baseline_workers=32,
        kind="play",
        allow_remote_play=True,
    )
    assert local_slots == 32
    assert remote_cap > 0
    assert "formal_heldout_additive" in weight_bits


def test_formal_heldout_remote_capabilities_are_fail_closed(mod) -> None:
    farm = _fake_remote_farm()
    audit = mod._remote_heldout_capability_audit(
        farm, required_endpoints=["elmo:8765"]
    )
    assert audit["passed"] is True
    farm.clients[0].info.capabilities = ()
    audit = mod._remote_heldout_capability_audit(
        farm, required_endpoints=["elmo:8765"]
    )
    assert audit["passed"] is False
