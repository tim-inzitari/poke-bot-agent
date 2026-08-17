from __future__ import annotations

import os
from pathlib import Path

import poke_bot
from poke_bot.remote_jobs import _BERT_HOSTS, _ELMO_HOSTS, _TRAIN_ROOT


ROOT = Path(__file__).resolve().parents[1]


def test_public_package_imports_and_reports_version() -> None:
    assert poke_bot.__version__ == "0.2.0"


def test_public_remote_defaults_have_no_private_fleet_identity() -> None:
    rendered = " ".join([*sorted(_BERT_HOSTS), *sorted(_ELMO_HOSTS)])
    assert "/home/" + "inzi" not in rendered
    assert "/Users/" + "tsinzitari" not in rendered
    assert "192.168.1." not in rendered
    assert _TRAIN_ROOT == Path.cwd()


def test_public_source_boundary_excludes_private_artifacts() -> None:
    for name in (
        ".r241-local-staging",
        ".cursor-loop",
        "state",
        "runtime",
        "evidence",
        "goals",
    ):
        assert not (ROOT / name).exists()
    assert (ROOT / "SOURCE_PROVENANCE.md").is_file()
    assert (ROOT / "THIRD_PARTY_NOTICES.md").is_file()


def test_environment_can_override_public_fleet_defaults(monkeypatch) -> None:
    monkeypatch.setenv("POKEBOT_TRAIN_ROOT", "/srv/example-agent")
    assert os.environ["POKEBOT_TRAIN_ROOT"] == "/srv/example-agent"
