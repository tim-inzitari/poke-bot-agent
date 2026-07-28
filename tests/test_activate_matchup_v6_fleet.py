from __future__ import annotations

import json
from pathlib import Path

import pytest

from poke_bot.dormant_adapter_compat import LOADER_RUNTIME_FILES
from scripts import activate_matchup_v6_fleet as subject


def _config(tmp_path: Path) -> dict[str, object]:
    source = tmp_path / "candidate"
    source.mkdir()
    registry = tmp_path / "registry.json"
    registry.write_text("{}\n", encoding="utf-8")
    return {
        "source_root": str(source),
        "registry": str(registry),
        "bert": {
            "host": "bert.local",
            "runtime_root": "/Users/test/workspace/poke-bot-agent",
            "service_label": "com.pokebot.remote-worker-8766",
            "endpoint": "bert.local:8766",
            "expected_workers": 16,
            "expected_leaves": 4,
        },
        "elmo": {
            "host": "elmo",
            "container": "poke-bot",
            "service": "worker",
            "image": "poke-bot:v6",
            "compose_files": ["/srv/host.yml", "/srv/production.yml"],
            "endpoint": "192.168.1.143:8765",
            "expected_workers": 36,
            "expected_leaves": 4,
        },
    }


def test_active_trainer_blocks_fleet_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "_service_is_active", lambda _service: True)
    with pytest.raises(RuntimeError, match="stopped trainer boundary"):
        subject.activate_fleet(
            config=_config(tmp_path),
            training_service="pokebot-training.service",
            receipt_path=tmp_path / "receipt.json",
        )


def test_activation_uses_relative_rsync_and_managed_services(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    source = Path(str(config["source_root"]))
    receipt_path = tmp_path / "receipt.json"
    expected = {relative: f"sha256:{index:064x}" for index, relative in enumerate(
        LOADER_RUNTIME_FILES, start=1
    )}
    commands: list[tuple[list[str], Path | None]] = []

    monkeypatch.setattr(subject, "_service_is_active", lambda _service: False)
    monkeypatch.setattr(subject, "_expected_contract", lambda _root: expected)
    monkeypatch.setattr(subject, "load_slot_registry", lambda _path: {"slots": []})
    monkeypatch.setattr(subject, "registry_digest", lambda _registry: "sha256:registry")
    monkeypatch.setattr(subject, "_remote_digests", lambda _host, _root: expected)
    monkeypatch.setattr(
        subject,
        "_wait_endpoint",
        lambda endpoint, workers, leaves: {
            "endpoint": endpoint,
            "workers": workers,
            "leaf_servers": leaves,
            "job_kinds": ["play"],
            "health": {"ok": True},
        },
    )

    def fake_run(
        argv: list[str],
        *,
        timeout: float = 180.0,
        cwd: Path | None = None,
    ) -> str:
        del timeout
        commands.append((argv, cwd))
        if argv[-2:] == ["id", "-u"]:
            return "501\n"
        return ""

    monkeypatch.setattr(subject, "_run", fake_run)

    def fake_validate(
        *,
        config: dict[str, object],
        receipt_path: Path,
    ) -> dict[str, object]:
        del config
        return json.loads(receipt_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(subject, "validate_fleet", fake_validate)
    receipt = subject.activate_fleet(
        config=config,
        training_service="pokebot-training.service",
        receipt_path=receipt_path,
    )

    rsync, rsync_cwd = commands[0]
    assert rsync[:2] == ["rsync", "-aR"]
    assert rsync[2:-1] == list(LOADER_RUNTIME_FILES)
    assert rsync_cwd == source.resolve()
    assert all(not value.startswith(str(source)) for value in rsync[2:-1])
    assert any(
        "launchctl" in " ".join(command) for command, _cwd in commands
    )
    assert any(
        command[:5] == ["ssh", "-o", "BatchMode=yes", "elmo", "sudo"]
        and "--force-recreate" in command
        for command, _cwd in commands
    )
    assert receipt["training_service_was_stopped"] is True
    assert receipt["managed_service_activation_only"] is True
