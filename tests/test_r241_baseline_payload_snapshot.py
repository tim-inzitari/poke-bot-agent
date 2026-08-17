"""Small containment regressions for the separately mounted r241 baseline tree."""

from __future__ import annotations

from pathlib import Path

import pytest

from poke_bot import r241_baseline_payload_snapshot as payload


def test_exact_roster_rejects_a_symlinked_group_ancestor(tmp_path: Path) -> None:
    """Do not inspect/copy a package that resolves outside its declared root."""

    source = tmp_path / "generic-baselines"
    source.mkdir()
    external_package = tmp_path / "outside" / "community" / "opponent"
    external_package.mkdir(parents=True)
    (external_package / "model.pt").write_bytes(b"outside package")
    (source / "community").symlink_to(external_package.parent, target_is_directory=True)
    roster = [
        {
            "id": "outside-opponent",
            "group": "community",
            "dir": "opponent",
            "content_digest": payload.content_digest(external_package),
        }
    ]

    with pytest.raises(payload.R241BaselinePayloadError, match="escapes source root"):
        payload.inventory_exact_roster(
            source,
            roster,
            generated_manifest=payload.minimal_manifest_bytes(roster),
        )
