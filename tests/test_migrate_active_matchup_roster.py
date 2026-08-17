from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.migrate_active_matchup_roster import preserve_bootstrap_authorization


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_roster_migration_preserves_bootstrap_authorization_identity(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "bootstrap.pt"
    parent.write_bytes(b"immutable-bootstrap")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model_path": str(parent.resolve()),
                "checkpoint_digest": _digest(parent),
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "authorization.json"
    source.write_text(
        json.dumps(
            {
                "schema": (
                    "poke_bot.matchup_adapter_specialist_bootstrap_"
                    "authorization/v1"
                ),
                "parent_checkpoint": str(parent.resolve()),
                "parent_checkpoint_digest": _digest(parent),
                "protected_manifest": str(manifest.resolve()),
                "protected_manifest_digest": _digest(manifest),
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "migrated-authorization.json"

    result = preserve_bootstrap_authorization(source, output)

    assert json.loads(output.read_text(encoding="utf-8")) == json.loads(
        source.read_text(encoding="utf-8")
    )
    assert result["parent_checkpoint"] == str(parent.resolve())
    assert result["preserved_immutable_bootstrap_identity"] is True
