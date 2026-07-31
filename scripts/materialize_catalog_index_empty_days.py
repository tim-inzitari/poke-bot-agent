#!/usr/bin/env python3
"""Create audited empty guide shards for catalog-proved zero-match days.

The exact public catalog already scanned every archive and checksum-binds both
the archive and its per-day match index. Reopening every replay on a day whose
indexed acting-seat count is zero adds no training evidence, so this command
materializes the established empty shard payload with an explicit
``checksum_bound_catalog_zero_match`` receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def materialize(
    *,
    catalog_path: Path,
    output_root: Path,
    template_day: str,
) -> list[str]:
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    specialist = str(catalog.get("specialist_id") or "")
    observed = {
        str(day): int(count)
        for day, count in dict(catalog.get("observed_by_day") or {}).items()
    }
    archives = {
        str(row.get("date") or ""): dict(row)
        for row in (catalog.get("source_archives") or ())
        if isinstance(row, dict)
    }
    facts = list(catalog.get("source_match_facts") or ())
    fact_days = {str(row[0]) for row in facts if isinstance(row, list) and row}
    if (
        catalog.get("schema") != "poke_bot.public_deck_archetype_catalog/v1"
        or not specialist
        or set(observed) != set(archives)
        or any(day in fact_days for day, count in observed.items() if count == 0)
    ):
        raise RuntimeError("public catalog zero-day evidence is invalid")

    template = output_root / f"{specialist}-{template_day}.features"
    template_sidecar = template.with_name(template.name + ".json")
    template_receipt = template.with_name(template.name + ".receipt.json")
    sidecar_base = json.loads(template_sidecar.read_text(encoding="utf-8"))
    receipt_base = json.loads(template_receipt.read_text(encoding="utf-8"))
    if (
        observed.get(template_day) != 0
        or int((sidecar_base.get("stats") or {}).get("records_kept", -1)) != 0
        or _sha256(template) != (sidecar_base.get("sha256") or "")
    ):
        raise RuntimeError("empty template shard identity changed")

    created: list[str] = []
    for day, count in sorted(observed.items()):
        if count != 0 or day == template_day:
            continue
        archive_row = archives[day]
        archive = catalog_path.parent.parent / "episode-days" / str(
            archive_row.get("archive") or ""
        )
        # The caller may keep the catalog outside the archive tree. In that
        # case only the catalog checksum evidence is required here; the daily
        # materializer will still require real archives for nonempty days.
        archive_bytes = archive.stat().st_size if archive.is_file() else None
        output = output_root / f"{specialist}-{day}.features"
        sidecar_path = output.with_name(output.name + ".json")
        receipt_path = output.with_name(output.name + ".receipt.json")
        existing = (output.exists(), sidecar_path.exists(), receipt_path.exists())
        if all(existing):
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if (
                sidecar.get("source_dates") != [day]
                or sidecar.get("sha256") != _sha256(output)
                or receipt.get("source_date") != day
                or (receipt.get("zero_match_materialization") or {}).get("mode")
                != "checksum_bound_catalog_zero_match"
            ):
                raise RuntimeError(f"existing zero-day artifact changed: {day}")
            continue
        if any(existing):
            raise FileExistsError(f"partial zero-day artifact exists: {day}")
        shutil.copyfile(template, output)
        os.chmod(output, 0o444)
        sidecar = dict(sidecar_base)
        sidecar.update(
            {
                "path": output.name,
                "source_archive": archive_row["archive"],
                "source_archive_sha256": archive_row["archive_sha256"],
                "source_dates": [day],
                "elapsed_seconds": 0.0,
                "zero_match_materialization": (
                    "checksum_bound_catalog_zero_match"
                ),
            }
        )
        sidecar["stats"] = {
            "records_total": 0,
            "records_kept": 0,
            "decisions_kept": 0,
            "target_coverage": {},
            "expanded_strategic_targets": (
                (sidecar_base.get("stats") or {}).get(
                    "expanded_strategic_targets"
                )
            ),
        }
        _atomic_json(sidecar_path, sidecar)
        receipt = dict(receipt_base)
        receipt.update(
            {
                "source_date": day,
                "elapsed_seconds": 0.0,
                "source_archive": {
                    "path": str(archive),
                    "bytes": archive_bytes,
                    "episode_members": int(
                        archive_row.get("json_replays") or 0
                    ),
                    "sha256": archive_row["archive_sha256"],
                    "validation_source": (
                        "checksum_bound_public_catalog_match_index"
                    ),
                },
                "stats": sidecar["stats"],
                "output": {
                    "path": str(output),
                    "bytes": output.stat().st_size,
                    "sha256": _sha256(output),
                    "metadata_path": str(sidecar_path),
                    "metadata_sha256": _sha256(sidecar_path),
                },
                "zero_match_materialization": {
                    "mode": "checksum_bound_catalog_zero_match",
                    "catalog": str(catalog_path),
                    "catalog_sha256": _sha256(catalog_path),
                    "source_match_rows_for_day": 0,
                },
            }
        )
        _atomic_json(receipt_path, receipt)
        created.append(day)
    return created


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--template-day", required=True)
    args = parser.parse_args()
    created = materialize(
        catalog_path=args.catalog.resolve(),
        output_root=args.output_root.resolve(),
        template_day=args.template_day,
    )
    print(json.dumps({"created_zero_days": created}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
