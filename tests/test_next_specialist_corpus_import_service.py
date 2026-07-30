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
    assert "OnSuccess=pokebot-next-specialist-prestage.service" not in service
    assert "ExecCondition=" not in service
    assert (
        "ExecStart=/home/inzi/poke-bot-agent/scripts/"
        "poll_and_import_prestaged_specialist_corpus.sh"
        in service
    )
    wrapper = (
        ROOT / "scripts/poll_and_import_prestaged_specialist_corpus.sh"
    ).read_text(encoding="utf-8")
    assert "import_prestaged_specialist_corpus.py" in wrapper
    assert '--minimum-records "$PRESTAGE_MINIMUM_RECORDS"' in wrapper
    assert (
        '--superseded-ready-sha256="$PRESTAGE_SUPERSEDED_READY_SHA256"'
        in wrapper
    )
    assert (
        "/usr/bin/systemctl --user start "
        "pokebot-next-specialist-prestage.service"
        in wrapper
    )
    assert "if ! /usr/bin/ssh" in wrapper
    assert "OnUnitInactiveSec=60s" in timer


def test_current_selector_names_one_exact_target() -> None:
    selector = (ROOT / "ops/next_specialist_prestage.env").read_text(
        encoding="utf-8"
    )
    assert selector.count("PRESTAGE_SPECIALIST_ID=") == 1
    assert selector.count("PRESTAGE_GUIDE_VERSION=") == 1
    assert selector.count("PRESTAGE_REMOTE_ROOT=") == 1
    assert selector.count("PRESTAGE_DESTINATION=") == 1
    assert "PRESTAGE_SPECIALIST_ID=teal-mask-ogerpon-ex" in selector
    assert (
        "PRESTAGE_GUIDE_VERSION="
        "teal-mask-ogerpon-ex-slop-box-north-star-v3"
        in selector
    )
    assert (
        "PRESTAGE_REMOTE_ROOT=/mnt/Main/main/poke-bot-agent/archive/"
        "teal-mask-ogerpon-ex-guide-corpus-full-v4-slop-box"
        in selector
    )
    assert selector.count("PRESTAGE_MINIMUM_RECORDS=1442") == 1
    assert (
        "PRESTAGE_SUPERSEDED_READY_SHA256="
        "sha256:9a243f6cd35973630b594756adcf1e2054f17e104621e9adddc75be849288483"
        in selector
    )
