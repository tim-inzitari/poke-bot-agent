from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot import paths
from poke_bot.baselines_runtime import (
    BaselineSpec,
    baseline_spec_payload,
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
