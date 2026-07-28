from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from poke_bot import paths
from poke_bot.baselines_runtime import (
    BaselineSpec,
    baseline_spec_payload,
    load_baseline_agent,
    resolve_baseline_spec_payload,
)


def _install_library(root: Path, *, main: str = "def agent(obs): return [0]\n") -> Path:
    baseline = root / "official" / "example-agent"
    baseline.mkdir(parents=True)
    (baseline / "main.py").write_text(main, encoding="utf-8")
    (baseline / "deck.csv").write_text("1\n" * 60, encoding="utf-8")
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "agents": [
                    {
                        "id": "example",
                        "name": "Example",
                        "dir": "example-agent",
                        "group": "official",
                        "source": "test",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return baseline


def _point_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(paths, "BASELINES_DIR", root)
    monkeypatch.setattr(paths, "BASELINES_MANIFEST", root / "manifest.json")


def test_portable_payload_ignores_sender_absolute_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sender_root = tmp_path / "inzi" / "baselines"
    sender_path = _install_library(sender_root)
    _point_paths(monkeypatch, sender_root)
    payload = baseline_spec_payload(
        BaselineSpec(
            id="example",
            name="Example",
            dir_name="example-agent",
            group="official",
            source="test",
            path=sender_path,
        )
    )

    remote_root = tmp_path / "bert" / "different" / "baselines"
    remote_path = _install_library(remote_root)
    _point_paths(monkeypatch, remote_root)
    resolved = resolve_baseline_spec_payload(
        payload, require_content_identity=True
    )

    assert resolved.path == remote_path.resolve()
    assert resolved.path != Path(payload["path"])


def test_portable_payload_fails_closed_on_content_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sender_root = tmp_path / "sender"
    sender_path = _install_library(sender_root)
    _point_paths(monkeypatch, sender_root)
    payload = baseline_spec_payload(
        BaselineSpec(
            id="example",
            name="Example",
            dir_name="example-agent",
            group="official",
            source="test",
            path=sender_path,
        )
    )

    remote_root = tmp_path / "remote"
    _install_library(remote_root, main="def agent(obs): return [1]\n")
    _point_paths(monkeypatch, remote_root)
    with pytest.raises(ValueError, match="content mismatch"):
        resolve_baseline_spec_payload(payload, require_content_identity=True)


def test_portable_payload_rejects_traversal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "baselines"
    baseline = _install_library(root)
    _point_paths(monkeypatch, root)
    payload = baseline_spec_payload(
        BaselineSpec(
            id="example",
            name="Example",
            dir_name="example-agent",
            group="official",
            source="test",
            path=baseline,
        )
    )
    payload["dir_name"] = "../outside"
    with pytest.raises(ValueError, match="unsafe baseline dir_name"):
        resolve_baseline_spec_payload(payload, require_content_identity=True)


def test_baseline_agent_cannot_leak_matchup_runtime_into_active_specialist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "baselines"
    baseline = _install_library(
        root,
        main="""\
import os

def agent(obs):
    os.environ["CG_LIB_PATH"] = "/frozen/opponent/cg"
    os.environ["POKEBOT_MATCHUP_ADAPTER_RUNTIME"] = "1"
    os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = "/frozen/alakazam/tree.json"
    os.environ["POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE"] = "frozen-opponent"
    return [0]
""",
    )
    _point_paths(monkeypatch, root)
    active = {
        "CG_LIB_PATH": "/active/cg",
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
        "POKEBOT_PUBLIC_MATCHUP_TREE_PATH": "/active/trevenant/tree.json",
        "POKEBOT_MATCHUP_ADAPTER_ROUTER_MODE": "active-specialist",
    }
    for key, value in active.items():
        monkeypatch.setenv(key, value)

    fn, _ = load_baseline_agent(
        BaselineSpec(
            id="example",
            name="Example",
            dir_name="example-agent",
            group="official",
            source="test",
            path=baseline,
        )
    )
    assert fn({}) == [0]
    assert {key: os.environ.get(key) for key in active} == active


def test_baseline_environment_is_restored_when_agent_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "baselines"
    baseline = _install_library(
        root,
        main="""\
import os

def agent(obs):
    os.environ["POKEBOT_PUBLIC_MATCHUP_TREE_PATH"] = "/leaked/tree.json"
    raise RuntimeError("expected test failure")
""",
    )
    _point_paths(monkeypatch, root)
    monkeypatch.delenv("POKEBOT_PUBLIC_MATCHUP_TREE_PATH", raising=False)
    fn, _ = load_baseline_agent(
        BaselineSpec(
            id="example",
            name="Example",
            dir_name="example-agent",
            group="official",
            source="test",
            path=baseline,
        )
    )

    with pytest.raises(RuntimeError, match="expected test failure"):
        fn({})
    assert "POKEBOT_PUBLIC_MATCHUP_TREE_PATH" not in os.environ
