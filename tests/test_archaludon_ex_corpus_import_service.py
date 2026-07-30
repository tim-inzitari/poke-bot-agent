from pathlib import Path
import hashlib
import json

import pytest

from scripts import materialize_archaludon_ex_full_public_schema7_corpus as corpus


def test_archaludon_import_is_inactive_and_checksum_authorized() -> None:
    unit = Path(
        "ops/systemd/pokebot-archaludon-ex-corpus-import.service"
    ).read_text(encoding="utf-8")

    assert "ARCHALUDON_EX_GUIDE_CORPUS_READY.json" in unit
    assert "CURRENT_DECK_GUIDE_CORPUS_READY.json" not in unit.split(
        "ExecCondition=", 1
    )[1].splitlines()[0]
    assert "--specialist-id archaludon-ex" in unit
    assert "--guide-version archaludon-ex-north-star-v1" in unit
    assert "--minimum-records 16639" in unit
    assert (
        "--finalization-receipt-name "
        "ARCHALUDON_EX_GUIDE_CORPUS_READY.json" in unit
    )
    assert (
        "--finalization-receipt-schema "
        "poke_bot.archaludon_ex_guide_corpus_validation/v2" in unit
    )
    assert "archaludon-ex-guide-full-public-schema7-r56-v1" in unit
    assert "archaludon-ex-full-public-schema7-r56-v1" in unit
    assert (
        "ExecStartPost=/home/inzi/miniconda3/envs/poke-bot-agent/bin/python "
        "/home/inzi/poke-bot-agent/scripts/"
        "validate_future_specialist_strategic_curriculum.py" in unit
    )
    assert "--guide-ready-receipt /home/inzi/poke-bot-agent/data/bootstrap/" in unit
    assert "--training-implementation /home/inzi/poke-bot-agent/poke_bot/train.py" in unit
    assert "--superseded-pointer-sha256" not in unit
    assert "latest20" not in unit
    assert "OnSuccess=" not in unit
    assert "systemctl" not in unit
    assert "specialist_runtime.env" not in unit


def test_archaludon_import_poll_is_bounded_and_idempotent() -> None:
    timer = Path(
        "ops/systemd/pokebot-archaludon-ex-corpus-import.timer"
    ).read_text(encoding="utf-8")

    assert "OnUnitActiveSec=5min" in timer
    assert "Persistent=true" in timer
    assert "pokebot-archaludon-ex-corpus-import.service" in timer


def test_archaludon_final_validation_preserves_sparse_causal_masks() -> None:
    build = Path(
        "ops/elmo/"
        "build_archaludon_ex_full_public_schema7_guide_corpus.sh"
    ).read_text(encoding="utf-8")

    assert "ARCHALUDON_EX_GUIDE_CORPUS_READY.json" in build
    assert "positive_guide_days.issubset(nonzero_source_days)" in build
    assert "positive_guide_days != nonzero_source_days" not in build
    assert "zero_guide_nonzero_source_days" in build
    assert (
        "poke_bot.archaludon_ex_guide_corpus_validation/v2" in build
    )
    assert 'start_date="2026-06-16"' in build
    assert 'end_date="2026-07-29"' in build
    assert 'minimum_records="16639"' in build
    assert 'int(metadata.get("dataset_schema") or -1) != 7' in build
    assert "schema6_feature_reuse_allowed" in build
    assert "latest20 identity contract" not in build


def test_archaludon_identity_builder_requires_full_public_schema7() -> None:
    build = Path(
        "scripts/materialize_archaludon_ex_full_public_schema7_corpus.py"
    ).read_text(encoding="utf-8")

    assert "START = date(2026, 6, 16)" in build
    assert "END = date(2026, 7, 29)" in build
    assert "MINIMUM_MATCHING_GAMES = 16_639" in build
    assert "DATASET_SCHEMA = 7" in build
    assert "schema6_feature_reuse_allowed" in build
    assert '"reused_days": 0' in build
    assert '"newly_featurized_days": len(_days())' in build
    assert "select_context_preserved" in build
    assert "selected_is_stop_preserved" in build
    assert "--reuse-root" not in build
    assert "dataset_schema\", -1)) == 6" not in build
    assert "ARCHALUDON_EX_LATEST20_CORPUS_READY.json" not in build
    compatibility = Path(
        "scripts/materialize_archaludon_ex_latest20_corpus.py"
    ).read_text(encoding="utf-8")
    assert (
        "from materialize_archaludon_ex_full_public_schema7_corpus "
        "import main"
    ) in compatibility


def test_archaludon_source_lock_is_generated_fail_closed() -> None:
    source = Path(
        "scripts/create_archaludon_ex_schema7_source_lock.py"
    ).read_text(encoding="utf-8")

    assert "SOURCE_LOCK_SCHEMA" in source
    assert "_validate_audit(" in source
    assert "roster_sha256" in source
    assert "classifier_required_specialist_count" in source
    assert "selected source audit is not the locked audit file" in source
    assert '"authorizes_materialization": True' in source
    assert '"authorizes_import_or_training": False' in source
    assert "required_specialist_count" in source


def test_existing_archaludon_ready_must_match_current_source_lock(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    pointer = tmp_path / "PROTECTED_EXPERT_CORPUS.json"
    sources = tmp_path / "SOURCE_ARCHIVES.json"
    manifest.write_text("{}\n", encoding="utf-8")
    pointer.write_text("{}\n", encoding="utf-8")
    sources.write_text(
        json.dumps({"source_audit": {"sha256": "sha256:a"}}) + "\n",
        encoding="utf-8",
    )
    daily = []
    for day in corpus._days():
        receipt = tmp_path / f"{day}.json"
        receipt.write_text("{}\n", encoding="utf-8")
        daily.append(
            {
                "date": day,
                "source_kind": (
                    "original_public_archive_schema7_rematerialization"
                ),
                "receipt": receipt.name,
                "receipt_sha256": corpus._sha256(receipt),
            }
        )
    snapshot = {
        "classifier_sha256": "sha256:classifier",
        "lock_sha256": "sha256:lock",
        "lock": {
            "files": {
                "state/archaludon_public_full44_source_audit_v1.json": (
                    "sha256:a"
                )
            }
        },
    }
    ready = {
        "schema": corpus.READY_SCHEMA,
        "status": "ready_checksum_validated",
        "specialist_id": corpus.TARGET,
        "source_policy": {
            "date_start": corpus.START.isoformat(),
            "date_end": corpus.END.isoformat(),
        },
        "dataset_schema": corpus.DATASET_SCHEMA,
        "feature_schema": corpus.FEATURE_SCHEMA,
        "records": corpus.MINIMUM_MATCHING_GAMES,
        "unique_episodes": corpus.MINIMUM_MATCHING_GAMES,
        "minimum_matching_games_met": True,
        "minimum_unique_episode_games_met": True,
        "daily_receipts": daily,
        "source_archives": {
            "receipt": sources.name,
            "receipt_sha256": corpus._sha256(sources),
        },
        "manifest": manifest.name,
        "manifest_sha256": corpus._sha256(manifest),
        "protected_pointer": pointer.name,
        "protected_pointer_sha256": corpus._sha256(pointer),
        "classifier_sha256": snapshot["classifier_sha256"],
        "build_provenance": {
            "source_lock": {"sha256": snapshot["lock_sha256"]}
        },
    }
    (tmp_path / "ARCHALUDON_EX_FULL_PUBLIC_CORPUS_READY.json").write_text(
        json.dumps(ready) + "\n",
        encoding="utf-8",
    )

    assert corpus._validate_existing_ready(tmp_path, snapshot) == ready
    changed = {**snapshot, "lock_sha256": "sha256:changed"}
    with pytest.raises(RuntimeError, match="existing Archaludon"):
        corpus._validate_existing_ready(tmp_path, changed)


def test_archaludon_public_source_audit_is_exact_and_nonready() -> None:
    path = Path("state/archaludon_public_full44_source_audit_v1.json")
    audit = json.loads(path.read_text(encoding="utf-8"))
    rows = list(audit["daily_sources"])

    assert audit["status"] == (
        "source_audit_complete_schema7_rematerialization_required"
    )
    assert audit["date_start"] == "2026-06-16"
    assert audit["date_end"] == "2026-07-29"
    assert audit["days"] == 44
    assert len(rows) == 44
    assert audit["archive_inventory"][
        "daily_dataset_index_manifest_rows"
    ] == 44
    assert audit["archive_inventory"]["latest_completed_day"] == "2026-07-29"
    method = audit["audit_method"]
    assert method["classifier_required_specialist_count"] == 19
    assert method["classifier_roster_revision"] == 9
    assert method["classifier_roster_sha256"] == (
        "sha256:" + hashlib.sha256(
            Path("state/matchup_adapter_roster.json").read_bytes()
        ).hexdigest()
    )
    assert rows[0]["date"] == "2026-06-16"
    assert rows[-1]["date"] == "2026-07-29"
    assert sum(row["matching_acting_seats"] for row in rows) == 21_278
    assert audit["matching_acting_seats"] == 21_278
    assert audit["minimum_matching_games"] == 16_639
    assert audit["minimum_met_by_public_source_scan"] is True
    assert audit["minimum_first_met_on"] == "2026-07-02"
    cumulative = 0
    first_met = None
    for row in rows:
        cumulative += row["matching_acting_seats"]
        if cumulative >= audit["minimum_matching_games"] and first_met is None:
            first_met = row["date"]
    assert first_met == audit["minimum_first_met_on"]
    assert "ready" not in audit["status"]


def test_historical_day_downloader_uses_real_kaggle_cli() -> None:
    source = Path(
        "scripts/download_historical_episode_days.py"
    ).read_text(encoding="utf-8")

    assert 'str(kaggle),' in source
    assert '"-m",\n                "kaggle"' not in source
    assert 'parser.add_argument(\n        "--kaggle"' in source


def test_archaludon_elmo_build_units_are_inactive_fail_closed_contracts() -> None:
    identity = Path(
        "ops/elmo/"
        "pokebot-archaludon-ex-full-public-schema7-r56-v1.service"
    ).read_text(encoding="utf-8")
    guide = Path(
        "ops/elmo/"
        "pokebot-archaludon-ex-guide-full-public-schema7-r56-v1.service"
    ).read_text(encoding="utf-8")

    assert "ConditionPathExists=" in identity
    assert "archaludon_ex_schema7_source_lock_v1.json" in identity
    assert "--day-parallelism 4" in identity
    assert "--workers-per-day 3" in identity
    assert "CPUQuota=1200%" in identity
    assert "MemoryHigh=40G" in identity
    assert "MemoryMax=48G" in identity
    assert "OnSuccess=" not in identity
    assert "ARCHALUDON_EX_FULL_PUBLIC_CORPUS_READY.json" in guide
    assert "archaludon_ex_schema7_source_lock_v1.json" in guide
    assert "MemoryHigh=40G" in guide
    assert "MemoryMax=48G" in guide
    assert "OnSuccess=" not in guide
