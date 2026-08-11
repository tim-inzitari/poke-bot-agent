"""Offline contract coverage for r260 sidecar materialization and transport."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pytest

from poke_bot import own_deck_rollout_store as store
from poke_bot.r260_sidecar_materialization import (
    EVIDENCE_DIRECTORY_NAME,
    R260SidecarMaterializationError,
    attest_r260_causal_local_remote_parity,
    bind_r260_inzi_dataset,
    finalize_r260_inzi_sidecar,
    materialize_r260_sidecar,
    stage_r260_sidecar_prefix_to_inzi,
    transport_r260_sidecar_to_inzi,
)


@dataclass(frozen=True)
class _Contract:
    path: Path
    sha256: str
    source_manifest_sha256: str
    source_window_receipt_sha256: str
    side_store_root: str
    inzi_training_root: str
    inzi_prefix_staging_root: str


@pytest.fixture(autouse=True)
def _stub_board_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    """The producer consumes already sealed rows; fixtures need no CG runtime."""

    monkeypatch.setattr(
        store,
        "board_feature_fingerprint",
        lambda _observation, _deck: "sha256:" + "b" * 64,
    )


def _sha(value: bytes | str) -> str:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _observation() -> dict[str, object]:
    card = {"id": 1, "serial": 100, "hp": 100, "energyCards": []}
    opponent = {"id": 2, "serial": 200, "hp": 100, "energyCards": []}
    return {
        "current": {
            "yourIndex": 0,
            "turn": 1,
            "result": -1,
            "energyAttached": False,
            "retreated": False,
            "stadium": [],
            "looking": [],
            "players": [
                {
                    "hand": [card],
                    "handCount": 1,
                    "active": [],
                    "bench": [],
                    "discard": [],
                    "prize": [None] * 6,
                    "deckCount": 53,
                },
                {
                    "hand": None,
                    "handCount": 1,
                    "active": [opponent],
                    "bench": [],
                    "discard": [],
                    "prize": [None] * 6,
                    "deckCount": 53,
                },
            ],
        },
        "select": {"deck": [], "option": [{"type": "pass"}], "minCount": 0, "maxCount": 0},
    }


@dataclass
class _Fixture:
    root: Path
    source_manifest: Path
    source_window: Path
    contract: _Contract
    window: store.SourceWindow
    identity: object

    def add_days(self, count: int) -> None:
        daily_root = self.root / "daily"
        existing = (
            len([item for item in daily_root.iterdir() if not item.name.startswith(".")])
            if daily_root.is_dir()
            else 0
        )
        for index, archive in enumerate(self.window.archives[existing : existing + count], start=existing):
            record = {
                "episode_id": f"r260-sidecar-{index:02d}",
                "source": f"pokemon-tcg-ai-battle-episodes-{archive.day}",
                "seat": 0,
                "archetype": "alakazam",
                "deck": [1] * 60,
                "info_set_ok": True,
                "steps": [{"env_step": 0, "observation": _observation(), "action": []}],
            }
            store._build_one_day(  # noqa: SLF001 - exact immutable r259 fixture builder.
                output_root=self.root,
                window=self.window,
                archive=archive,
                identity=self.identity,
                records=[record],
                archive_native_accounting=store._ArchiveNativeAccounting(  # noqa: SLF001
                    episodes_seen=1,
                    verified_reward_episodes=1,
                    records_emitted=1,
                ),
            )


def _fixture(tmp_path: Path, *, days: int) -> _Fixture:
    root = tmp_path / "elmo-sidecar"
    root.mkdir()
    source_window = tmp_path / "versioned-window.json"
    source_window.write_text(json.dumps({"window": "fixture"}), encoding="utf-8")
    start = date(2026, 7, 22)
    counts = [4563] * 13 + [4562] * 7
    remaining_bytes = 14_842_033_482
    archive_rows: list[dict[str, object]] = []
    archives: list[store.SourceArchive] = []
    for index in range(20):
        day = (start + timedelta(days=index)).isoformat()
        bytes_for_day = 742_101_674 if index < 19 else remaining_bytes
        remaining_bytes -= bytes_for_day
        digest = "sha256:" + f"{index + 1:064x}"
        archive_rows.append(
            {
                "date": day,
                "dataset_slug": f"pokemon-tcg-ai-battle-episodes-{day}",
                "path": f"/immutable/{day}.zip",
                "bytes": bytes_for_day,
                "sha256": digest,
                "validated": True,
                "validated_episode_count": counts[index],
            }
        )
        archives.append(
            store.SourceArchive(
                day=day,
                path=tmp_path / f"unused-{day}.zip",
                sha256=digest,
                bytes=bytes_for_day,
                validated_episode_count=counts[index],
                source_slug=f"pokemon-tcg-ai-battle-episodes-{day}",
            )
        )
    source_manifest = tmp_path / "current.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema": store.ARCHIVE_RECEIPT_SCHEMA,
                "status": "ready",
                "window_policy": "exact_20_consecutive_calendar_days",
                "window_start": "2026-07-22",
                "window_end": "2026-08-10",
                "days": 20,
                "total_episodes": 91_253,
                "archives": archive_rows,
                "versioned_receipt": str(source_window),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha = _sha(source_manifest.read_bytes())
    window_sha = _sha(source_window.read_bytes())
    final_root = tmp_path / "inzi-final"
    staging_root = tmp_path / "inzi-staging"
    contract = _Contract(
        path=tmp_path / "owner.json",
        sha256=_sha("owner"),
        source_manifest_sha256=manifest_sha,
        source_window_receipt_sha256=window_sha,
        side_store_root=str(root),
        inzi_training_root=str(final_root),
        inzi_prefix_staging_root=str(staging_root),
    )
    window = store.SourceWindow(
        manifest_path=source_manifest,
        manifest_sha256=manifest_sha,
        original_manifest_path=str(source_manifest),
        original_versioned_receipt_path=str(source_window),
        versioned_receipt_path=source_window,
        versioned_receipt_sha256=window_sha,
        archives=tuple(archives),
    )
    identity = store._BuildIdentity(  # noqa: SLF001 - exact r259 metadata ABI fixture.
        mode="archive_native",
        source_snapshot_path="/fixture/snapshot",
        source_snapshot_tree_sha256=_sha("source-tree"),
        image_tag="fixture-image",
        image_id=_sha("image"),
        code_identities={"fixture.py": _sha("fixture-code")},
        classifier={"fixture": _sha("classifier")},
        protected_stream_sha256=None,
    )
    fixture = _Fixture(root, source_manifest, source_window, contract, window, identity)
    fixture.add_days(days)
    return fixture


def test_prefix_transfer_is_append_only_and_never_materializable(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, days=15)
    first = stage_r260_sidecar_prefix_to_inzi(
        source_sidecar_root=fixture.root,
        source_manifest=fixture.source_manifest,
        source_window_receipt=fixture.source_window,
        inzi_staging_root=fixture.contract.inzi_prefix_staging_root,
        expected_elmo_sidecar_root=fixture.root,
        owner_contract=fixture.contract,
    )
    assert len(first.committed_days) == 15
    assert len(first.copied_days) == 15
    payload = json.loads(first.prefix_receipt.path.read_text(encoding="utf-8"))
    assert payload["status"] == "partial_staging_not_training_eligible"
    assert payload["r260_training_eligible"] is False

    second = stage_r260_sidecar_prefix_to_inzi(
        source_sidecar_root=fixture.root,
        source_manifest=fixture.source_manifest,
        source_window_receipt=fixture.source_window,
        inzi_staging_root=fixture.contract.inzi_prefix_staging_root,
        expected_elmo_sidecar_root=fixture.root,
        owner_contract=fixture.contract,
    )
    assert second.copied_days == ()
    with pytest.raises(R260SidecarMaterializationError, match="complete twenty-day"):
        materialize_r260_sidecar(
            sidecar_root=fixture.root,
            source_manifest=fixture.source_manifest,
            source_window_receipt=fixture.source_window,
            evidence_root=fixture.root / EVIDENCE_DIRECTORY_NAME,
            expected_sidecar_root=fixture.root,
            owner_contract=fixture.contract,
        )


def test_full_materialization_transport_parity_and_final_bind(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, days=20)
    materialized = materialize_r260_sidecar(
        sidecar_root=fixture.root,
        source_manifest=fixture.source_manifest,
        source_window_receipt=fixture.source_window,
        evidence_root=fixture.root / EVIDENCE_DIRECTORY_NAME,
        expected_sidecar_root=fixture.root,
        owner_contract=fixture.contract,
    )
    assert materialized.row_count == 20
    assert materialized.joined_dataset.path.stat().st_mode & 0o777 == 0o444

    transported = transport_r260_sidecar_to_inzi(
        source_sidecar_root=fixture.root,
        source_evidence_root=materialized.evidence_root,
        inzi_sidecar_root=fixture.contract.inzi_training_root,
        expected_elmo_sidecar_root=fixture.root,
        owner_contract=fixture.contract,
    )
    parity = attest_r260_causal_local_remote_parity(
        elmo_sidecar_root=fixture.root,
        elmo_evidence_root=materialized.evidence_root,
        inzi_sidecar_root=transported.inzi_sidecar_root,
        inzi_evidence_root=transported.inzi_evidence_root,
        expected_elmo_sidecar_root=fixture.root,
        owner_contract=fixture.contract,
        sample_limit=8,
    )
    completion = finalize_r260_inzi_sidecar(
        inzi_sidecar_root=transported.inzi_sidecar_root,
        inzi_evidence_root=transported.inzi_evidence_root,
        expected_elmo_sidecar_root=fixture.root,
        owner_contract=fixture.contract,
        validate_with_launcher=False,
    )
    aggregate = json.loads(completion.aggregate_binding.path.read_text(encoding="utf-8"))
    assert aggregate["status"] == "complete_training_eligible"
    assert aggregate["day_count"] == 20
    binding_path = bind_r260_inzi_dataset(
        completion=completion,
        source_transport_receipt=transported.transport_receipt.path,
        owner_contract=fixture.contract,
    )
    binding = json.loads(binding_path.path.read_text(encoding="utf-8"))
    assert parity.path == completion.parity_receipt.path
    assert binding["inzi_sidecar_root"] == fixture.contract.inzi_training_root
    assert binding["sidecar_binding_sha256"] == aggregate["binding_sha256"]
    assert binding["sidecar_binding_file_sha256"] == _sha(
        completion.aggregate_binding.path.read_bytes()
    )


def test_inzi_destination_cannot_be_an_elmo_sidecar_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, days=1)
    with pytest.raises(R260SidecarMaterializationError, match="Elmo side-store"):
        stage_r260_sidecar_prefix_to_inzi(
            source_sidecar_root=fixture.root,
            source_manifest=fixture.source_manifest,
            source_window_receipt=fixture.source_window,
            inzi_staging_root=fixture.root / "not-inzi",
            expected_elmo_sidecar_root=fixture.root,
            owner_contract=fixture.contract,
        )
