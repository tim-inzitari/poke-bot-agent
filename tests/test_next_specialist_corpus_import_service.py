from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_import_service_is_generic_and_uses_one_selector_environment() -> None:
    service = (
        ROOT / "ops/systemd/pokebot-next-specialist-corpus-import.service"
    ).read_text(encoding="utf-8")
    timer = (
        ROOT / "ops/systemd/pokebot-next-specialist-corpus-import.timer"
    ).read_text(encoding="utf-8")
    assert "EnvironmentFile=%h/.config/pokebot/next_specialist_prestage.env" in service
    assert "team-rockets-spidops" not in service
    assert "OnSuccess=pokebot-next-specialist-prestage.service" in service
    assert "import_prestaged_specialist_corpus.py" in service
    assert "OnUnitInactiveSec=60s" in timer


def test_current_selector_names_one_exact_target() -> None:
    selector = (ROOT / "ops/next_specialist_prestage.env").read_text(
        encoding="utf-8"
    )
    assert selector.count("PRESTAGE_SPECIALIST_ID=") == 1
    assert selector.count("PRESTAGE_GUIDE_VERSION=") == 1
    assert selector.count("PRESTAGE_REMOTE_ROOT=") == 1
    assert selector.count("PRESTAGE_DESTINATION=") == 1
