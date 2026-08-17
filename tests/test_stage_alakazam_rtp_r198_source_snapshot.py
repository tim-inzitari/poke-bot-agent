"""Focused closure checks for the r198 immutable-source publisher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "scripts" / "stage_alakazam_rtp_r198_three_arm_eval_source_snapshot.py"


@pytest.fixture(scope="module")
def snapshot_module():
    spec = importlib.util.spec_from_file_location("r198_eval_source_snapshot_test", SNAPSHOT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_curated_poke_bot(root: Path, files: frozenset[str]) -> None:
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# sealed test source\n", encoding="utf-8")


def _write_candidate_completion_receipt(
    snapshot_module,
    monkeypatch: pytest.MonkeyPatch,
    path: Path,
    *,
    candidate_id: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "poke_bot.alakazam_rtp_r197_shadow_candidate/v1",
        "status": "completed_shadow_only",
        "candidate_id": candidate_id or snapshot_module.R198_CANDIDATE_ID,
        "candidate_contract_sha256": snapshot_module.R198_CANDIDATE_CONTRACT_SHA256,
    }
    data = snapshot_module._pretty_json(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    monkeypatch.setattr(
        snapshot_module,
        "CANDIDATE_COMPLETION_RECEIPT_SHA256",
        snapshot_module._sha256_bytes(data),
    )
    monkeypatch.setattr(
        snapshot_module,
        "CANDIDATE_COMPLETION_RECEIPT_BYTES",
        len(data),
    )
    return payload


def _write_curated_state_tree(snapshot_module, root: Path) -> Path:
    state_root = root / "state"
    state_root.mkdir(parents=True)
    (state_root / "alakazam-rtp-realignment-r197.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    roster = state_root / snapshot_module.MATCHUP_ADAPTER_ROSTER.name
    roster.write_bytes((ROOT / snapshot_module.MATCHUP_ADAPTER_ROSTER).read_bytes())
    return roster


def test_poke_bot_exact_runtime_closure_accepts_the_curated_tree(
    snapshot_module, tmp_path: Path
) -> None:
    source_root = tmp_path.resolve()
    _write_curated_poke_bot(source_root, snapshot_module.CURATED_POKE_BOT_FILES)

    snapshot_module._validate_poke_bot_tree(source_root)

    required_r198_modules = {
        "poke_bot/rtp_three_arm_evaluation.py",
        "poke_bot/rtp_three_arm_evaluation_runner.py",
        "poke_bot/rtp_r198_evaluation_input_materializer.py",
        "poke_bot/rtp_r198_production_factory.py",
        "poke_bot/engine_rebuild/rtp_pairing_snapshot.py",
        "poke_bot/recursive_turn_planner/r197_action_authority.py",
    }
    assert required_r198_modules <= snapshot_module.CURATED_POKE_BOT_FILES


def test_poke_bot_exact_runtime_closure_rejects_unlisted_nonhidden_payload(
    snapshot_module, tmp_path: Path
) -> None:
    source_root = tmp_path.resolve()
    _write_curated_poke_bot(source_root, snapshot_module.CURATED_POKE_BOT_FILES)
    payload = source_root / "poke_bot" / "private_payload.py"
    payload.write_text("secret = 'must not publish'\n", encoding="utf-8")

    with pytest.raises(
        snapshot_module.SnapshotError,
        match="unlisted regular file: poke_bot/private_payload.py",
    ):
        snapshot_module._validate_poke_bot_tree(source_root)


def test_state_tree_accepts_and_inventories_exact_matchup_adapter_roster(
    snapshot_module, tmp_path: Path
) -> None:
    source_root = tmp_path.resolve()
    roster = _write_curated_state_tree(snapshot_module, source_root)

    snapshot_module._validate_state_tree(source_root)
    entries, _ = snapshot_module._walk_physical_tree(source_root)
    roster_entry = snapshot_module._find_entry(
        entries,
        snapshot_module.MATCHUP_ADAPTER_ROSTER,
    )

    assert str(snapshot_module.MATCHUP_ADAPTER_ROSTER) in (
        snapshot_module._required_relative_files()
    )
    assert roster_entry["mode"] == 0o444
    assert roster_entry["sha256"] == snapshot_module.MATCHUP_ADAPTER_ROSTER_SHA256
    assert roster_entry["size"] == snapshot_module.MATCHUP_ADAPTER_ROSTER_BYTES
    assert roster.stat().st_size == snapshot_module.MATCHUP_ADAPTER_ROSTER_BYTES


def test_state_tree_rejects_tampered_matchup_adapter_roster(
    snapshot_module, tmp_path: Path
) -> None:
    source_root = tmp_path.resolve()
    roster = _write_curated_state_tree(snapshot_module, source_root)
    roster.write_bytes(roster.read_bytes() + b" ")

    with pytest.raises(
        snapshot_module.SnapshotError,
        match="matchup adapter roster identity changed",
    ):
        snapshot_module._validate_state_tree(source_root)


def test_state_tree_rejects_missing_matchup_adapter_roster(
    snapshot_module, tmp_path: Path
) -> None:
    source_root = tmp_path.resolve()
    state_root = source_root / "state"
    state_root.mkdir()
    (state_root / "alakazam-rtp-realignment-r197.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(snapshot_module.SnapshotError, match="missing physical component"):
        snapshot_module._validate_state_tree(source_root)


def test_state_tree_rejects_symlinked_matchup_adapter_roster(
    snapshot_module, tmp_path: Path
) -> None:
    source_root = tmp_path.resolve()
    state_root = source_root / "state"
    state_root.mkdir()
    (state_root / "alakazam-rtp-realignment-r197.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (state_root / snapshot_module.MATCHUP_ADAPTER_ROSTER.name).symlink_to(
        (ROOT / snapshot_module.MATCHUP_ADAPTER_ROSTER).resolve()
    )

    with pytest.raises(snapshot_module.SnapshotError, match="uncurated entries under state"):
        snapshot_module._validate_state_tree(source_root)


def test_state_tree_rejects_unlisted_files_and_directories(
    snapshot_module, tmp_path: Path
) -> None:
    file_root = tmp_path.resolve() / "file-case"
    _write_curated_state_tree(snapshot_module, file_root)
    (file_root / "state/private.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(snapshot_module.SnapshotError, match="private.json"):
        snapshot_module._validate_state_tree(file_root)

    directory_root = tmp_path.resolve() / "directory-case"
    _write_curated_state_tree(snapshot_module, directory_root)
    (directory_root / "state/private").mkdir()
    with pytest.raises(snapshot_module.SnapshotError, match="private"):
        snapshot_module._validate_state_tree(directory_root)


def test_candidate_completion_receipt_accepts_exact_completed_shadow_identity(
    snapshot_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = tmp_path.resolve() / "r197-completion-receipt.json"
    payload = _write_candidate_completion_receipt(
        snapshot_module,
        monkeypatch,
        receipt,
    )

    validated = snapshot_module._validate_candidate_completion_receipt(receipt)

    assert validated == payload
    assert (
        snapshot_module.CANDIDATE_ASSET_FILES["completion_receipt"]
        == "r197-completion-receipt.json"
    )
    digest_by_key = {
        "parent_checkpoint": snapshot_module.PARENT_SHA256,
        "sidecar": snapshot_module.SIDECAR_SHA256,
        "sidecar_receipt": snapshot_module.SIDECAR_RECEIPT_SHA256,
        "completion_receipt": snapshot_module.CANDIDATE_COMPLETION_RECEIPT_SHA256,
        "deck": snapshot_module.R195_DECK_CSV_SHA256,
        "matchup_tree": snapshot_module.R195_MATCHUP_TREE_SHA256,
    }
    entries = [
        {
            "path": str(snapshot_module.CANDIDATE_ASSET_ROOT / filename),
            "type": "file",
            "mode": 0o444,
            "size": (
                snapshot_module.CANDIDATE_COMPLETION_RECEIPT_BYTES
                if key == "completion_receipt"
                else 1
            ),
            "sha256": digest_by_key[key],
        }
        for key, filename in snapshot_module.CANDIDATE_ASSET_FILES.items()
    ]
    candidate_snapshot = snapshot_module._candidate_snapshot_payload(
        entries,
        published_root=tmp_path / "published",
    )
    assert set(candidate_snapshot["artifacts"]) == set(
        snapshot_module.CANDIDATE_ASSET_FILES
    )
    assert candidate_snapshot["artifacts"]["completion_receipt"] == {
        "path": str(
            tmp_path
            / "published"
            / snapshot_module.CANDIDATE_ASSET_ROOT
            / "r197-completion-receipt.json"
        ),
        "sha256": snapshot_module.CANDIDATE_COMPLETION_RECEIPT_SHA256,
        "bytes": snapshot_module.CANDIDATE_COMPLETION_RECEIPT_BYTES,
    }


def test_candidate_completion_receipt_rejects_tampered_bytes_and_candidate(
    snapshot_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = tmp_path.resolve() / "r197-completion-receipt.json"
    _write_candidate_completion_receipt(snapshot_module, monkeypatch, receipt)
    receipt.write_bytes(receipt.read_bytes() + b" ")

    with pytest.raises(
        snapshot_module.SnapshotError,
        match="completion receipt identity changed",
    ):
        snapshot_module._validate_candidate_completion_receipt(receipt)

    _write_candidate_completion_receipt(
        snapshot_module,
        monkeypatch,
        receipt,
        candidate_id="r197-wrong-candidate",
    )
    with pytest.raises(
        snapshot_module.SnapshotError,
        match="does not bind the completed r198 candidate",
    ):
        snapshot_module._validate_candidate_completion_receipt(receipt)


def test_candidate_completion_receipt_rejects_missing_physical_copy(
    snapshot_module, tmp_path: Path
) -> None:
    with pytest.raises(snapshot_module.SnapshotError, match="missing physical component"):
        snapshot_module._validate_candidate_completion_receipt(
            tmp_path.resolve() / "r197-completion-receipt.json"
        )


def test_publish_renames_a_complete_readonly_partial_tree_atomically(
    snapshot_module, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The final content-addressed name is absent until one root rename."""

    staging = (tmp_path / "staging").resolve()
    deployments = (tmp_path / "deployments").resolve()
    staging.mkdir()
    deployments.mkdir()
    payload = staging / "payload.py"
    payload.write_text("sealed payload\n", encoding="utf-8")

    monkeypatch.setattr(snapshot_module, "_required_relative_files", lambda: ["payload.py"])
    monkeypatch.setattr(snapshot_module, "OFFICIAL_PANEL_IDS", ())
    monkeypatch.setattr(
        snapshot_module,
        "_generated_snapshot_artifacts",
        lambda entries, *, published_root: ({}, {}),
    )
    monkeypatch.setattr(
        snapshot_module,
        "_eval_cg_closure_binding",
        lambda entries, *, published_root: {},
    )

    entries = [
        {
            "path": "payload.py",
            "type": "file",
            "mode": 0o444,
            "size": payload.stat().st_size,
            "sha256": snapshot_module._sha256_file(payload),
        }
    ]
    directories = [{"path": ".", "mode": 0o555}]
    tree = snapshot_module._tree_sha256(entries=entries, directories=directories)
    target_name = snapshot_module.DEPLOYMENT_PREFIX + tree.removeprefix("sha256:")[:12]
    published = deployments / target_name
    rendered = (
        "[Service]\n"
        f"WorkingDirectory={published}\n"
        f"Environment=PYTHONPATH={published}\n"
        f"Environment=CG_LIB_PATH={published}/{snapshot_module.EVAL_CG_ROOT}\n"
        f"ConditionPathExists={published}/{snapshot_module.MANIFEST_NAME}\n"
        f"ConditionPathExists={published}/{snapshot_module.MATCHUP_ADAPTER_ROSTER}\n"
        f"Environment=POKEBOT_R198_EVAL_SOURCE_SNAPSHOT_ROOT={published}\n"
        f"Environment=POKEBOT_R198_EVAL_SOURCE_TREE_SHA256={tree}\n"
        f"ExecStartPre={snapshot_module.PYTHON} -u "
        f"{snapshot_module.SNAPSHOT_SCRIPT} verify --published-root {published}\n"
    ).encode("utf-8")
    manifest = {
        "schema": snapshot_module.SCHEMA,
        "source_tree_sha256": tree,
        "target_name": target_name,
        "source_directories": directories,
        "source_entries": entries,
        "required_relative_files": ["payload.py"],
        "physical_no_symlinks": True,
        "published_file_mode": 0o444,
        "published_directory_mode": 0o555,
        "official_control_panel": {
            "schema": "poke_bot.rtp_three_arm_official_control_panel/v1",
            "controls": [],
        },
        "generated_artifacts": {},
        "eval_cg_closure": {},
        "rendered_unit": {
            "path": str(snapshot_module.RENDERED_UNIT_RELATIVE),
            "sha256": snapshot_module._sha256_bytes(rendered),
            "size": len(rendered),
            "mode": 0o444,
            "template_path": str(snapshot_module.UNIT_TEMPLATE_RELATIVE),
            "template_sha256": "sha256:" + "0" * 64,
        },
    }

    monkeypatch.setattr(
        snapshot_module,
        "build_plan",
        lambda source, destination: (published, manifest, rendered),
    )
    rename_calls: list[tuple[Path, Path]] = []
    real_rename = snapshot_module._rename_no_replace

    def observed_rename(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        rename_calls.append((source_path, destination_path))
        assert source_path.parent == deployments
        assert source_path.name.startswith(f".{target_name}.")
        assert source_path.name.endswith(".partial")
        assert source_path.stat().st_mode & 0o777 == 0o555
        assert destination_path == published
        assert not destination_path.exists()
        real_rename(source_path, destination_path)

    monkeypatch.setattr(snapshot_module, "_rename_no_replace", observed_rename)

    if snapshot_module.sys.platform == "darwin":
        # Darwin rejects moving a directory after its own write bit is removed
        # (even plain rename(2) returns EACCES).  The publisher must preserve
        # the sealed partial rather than re-open it at the final public name.
        with pytest.raises(
            snapshot_module.SnapshotError,
            match="macOS refuses atomic no-clobber rename of a sealed snapshot root",
        ):
            snapshot_module.publish(staging, deployments)
        partials = list(deployments.glob(f".{target_name}.*.partial"))
        assert len(partials) == 1
        assert partials[0].stat().st_mode & 0o777 == 0o555
        assert rename_calls and len(rename_calls) == 1
        assert not published.exists()
        return

    result = snapshot_module.publish(staging, deployments)

    assert result["status"] == "published"
    assert result["published_root"] == str(published)
    assert rename_calls and len(rename_calls) == 1
    assert snapshot_module.validate_published_root(published)["status"] == "valid"
    assert (published / "payload.py").stat().st_mode & 0o777 == 0o444
    assert (published / snapshot_module.MANIFEST_NAME).stat().st_mode & 0o777 == 0o444
    assert (published / snapshot_module.RENDERED_UNIT_RELATIVE).stat().st_mode & 0o777 == 0o444
    assert published.stat().st_mode & 0o777 == 0o555
    assert (published / snapshot_module.RENDERED_UNIT_RELATIVE.parent).stat().st_mode & 0o777 == 0o555
    assert not list(deployments.glob(f".{target_name}.*.partial"))


def test_atomic_publish_rename_rejects_an_empty_existing_target(
    snapshot_module, tmp_path: Path
) -> None:
    deployments = tmp_path.resolve()
    partial = deployments / ".partial"
    destination = deployments / "content-addressed-target"
    partial.mkdir(mode=0o700)
    destination.mkdir(mode=0o700)
    partial.chmod(0o555)

    with pytest.raises(snapshot_module.SnapshotError, match="refusing to overwrite existing"):
        snapshot_module._rename_no_replace(partial, destination)

    assert partial.is_dir()
    assert partial.stat().st_mode & 0o777 == 0o555
    assert destination.is_dir()
