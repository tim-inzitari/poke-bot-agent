from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from scripts.managed_runtime_registry import registry_from_managed_service


def test_resolves_exact_effective_registry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text("{}\n", encoding="utf-8")

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=f"{{ path=/python ; argv[]=/python launch.py --registry {registry} ; }}\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert registry_from_managed_service("pokebot-marnie.service") == registry.resolve()


def test_rejects_ambiguous_or_missing_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout="/python launch.py\n", stderr=""
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exactly one registry"):
        registry_from_managed_service("pokebot-marnie.service")


def test_rejects_unsafe_service_identity() -> None:
    with pytest.raises(RuntimeError, match="unsafe managed service"):
        registry_from_managed_service("ssh.service")
