#!/usr/bin/env python3
"""Build bounded temporal-expert shards for rare-route evidence days.

This runs in a separate capped container on Elmo.  It never changes the live
worker image or active expert corpus.  Each completed day is checksum-backed
and reusable by the later additive specialist-corpus assembly.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from time import time
from typing import Any


IMAGE = "poke-bot-truenas-worker:matchup-v33-runtime"
ROOT = Path("/mnt/Main/main/poke-adapter-oracle-v29")
ARCHIVE_ROOT = Path("/srv/poke-bot-agent/archive/episode-days")
OUTPUT_ROOT = Path(
    "/srv/poke-bot-agent/archive/"
    "rare-route-expert-history-20260626-20260701"
)
ARCHETYPES = ROOT / "src/poke_bot/archetypes_v33.py"
COLLECTOR = ROOT / "src/scripts/collect_top_ladder_replays.py"
FEATURIZER = ROOT / "src/scripts/featurize_bootstrap_shard.py"
FEATURE_SHARDS = ROOT / "src/poke_bot/feature_shards.py"
MATCHUP_ADAPTERS = ROOT / "src/poke_bot/matchup_adapters.py"
ASSEMBLER = ROOT / "src/scripts/assemble_feature_manifest.py"
SPLITTER = ROOT / "src/scripts/split_expert_manifest_by_archetype.py"
FILTER = ROOT / "src/scripts/filter_feature_manifest.py"
TRAINING_MIX_ROOT = Path(
    "/srv/poke-bot-agent/privileged-collector-v1/"
    "data/training_mixes"
)
EPISODES_INDEX_ROOT = Path(
    "/srv/poke-bot-agent/privileged-collector-v1/"
    "kaggle/input/pokemon-tcg-ai-battle-episodes-index"
)
DAYS = (
    "2026-06-26",
    "2026-06-27",
    "2026-06-28",
    "2026-06-29",
    "2026-06-30",
    "2026-07-01",
    # The refreshed index and already-downloaded archives add the newest
    # evidence without replacing the protected 20-day corpus.
    "2026-07-22",
    "2026-07-23",
)
RARE_ARCHETYPES = (
    "dragapult-blaziken",
    "dragapult-dusknoir",
    "dudunsparce",
    "gardevoir",
    "ns-zoroark",
    "walrein",
)
EXPECTED_ADDITIVE_BY_DAY = {
    "2026-06-26": ("dragapult-dusknoir", "dudunsparce", "walrein"),
    "2026-06-27": ("dragapult-dusknoir", "dudunsparce", "walrein"),
    "2026-06-28": ("dragapult-dusknoir", "dudunsparce", "walrein"),
    "2026-06-29": ("dragapult-dusknoir", "walrein"),
    "2026-06-30": ("dragapult-dusknoir", "walrein"),
    "2026-07-01": (),
    "2026-07-22": (),
    "2026-07-23": ("dragapult-blaziken", "ns-zoroark"),
}
MIN_RECOGNIZED_RECORDS_PER_DAY = 5000
COLLECTOR_CONTRACT = "additive_allowed_archetypes_v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validated_day(day: str) -> dict[str, Any] | None:
    shard = OUTPUT_ROOT / f"all-recognized-{day}.features"
    sidecar = Path(str(shard) + ".json")
    identity_path = OUTPUT_ROOT / f"all-recognized-{day}.identity.json"
    if (
        not shard.is_file()
        or not sidecar.is_file()
        or not identity_path.is_file()
    ):
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    if (
        payload.get("format") != "pokebot-bootstrap-feature-shard"
        or int(payload.get("format_version") or 0) != 1
        or payload.get("compact_mode") != "temporal-expert-v1"
        or int(payload.get("max_context") or 0) != 320
        or list(payload.get("source_dates") or ()) != [day]
        or str(payload.get("sha256") or "") != _sha256(shard)
        or int((payload.get("stats") or {}).get("decisions_kept") or 0) <= 0
        or identity.get("schema")
        != "poke_bot.rare_route_expert_shard_identity/v1"
        or tuple(identity.get("additive_archetypes") or ())
        != RARE_ARCHETYPES
        or identity.get("collector_contract") != COLLECTOR_CONTRACT
        or not set(EXPECTED_ADDITIVE_BY_DAY[day]).issubset(
            set(identity.get("observed_additive_archetypes") or ())
        )
        or identity.get("shard_sha256") != payload.get("sha256")
    ):
        # A prior collector contract may have produced an otherwise valid
        # feature shard without the additive specialist seats.  Treat that as
        # a cache miss so the caller can discard and rebuild it under the
        # current contract; it is not a terminal service failure.
        return None
    return {
        "day": day,
        "shard": str(shard),
        "sidecar": str(sidecar),
        "sha256": payload["sha256"],
        "records": int((payload.get("stats") or {}).get("records_kept") or 0),
        "decisions": int((payload.get("stats") or {}).get("decisions_kept") or 0),
    }


def _validated_jsonl(day: str) -> Path | None:
    jsonl = OUTPUT_ROOT / f"all-recognized-{day}.jsonl"
    meta_path = jsonl.with_suffix(".meta.json")
    if not jsonl.is_file() or not meta_path.is_file():
        return None
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    stats = dict(payload.get("stats") or {})
    observed_archetypes = set(
        str(value) for value in (stats.get("record_archetypes") or {})
    )
    quality = dict(payload.get("quality_gates") or {})
    if (
        payload.get("schema") != "poke_bot.top_ladder_bootstrap/v1"
        or payload.get("policy_scope") != "recognized_families_only"
        or tuple(
            (payload.get("classifier") or {}).get(
                "additive_registered_ids"
            )
            or ()
        )
        != RARE_ARCHETYPES
        or quality.get("passed") is not True
        or int(stats.get("records_written") or 0)
        < MIN_RECOGNIZED_RECORDS_PER_DAY
        or payload.get("output_sha256") != _sha256(jsonl)
        or not set(EXPECTED_ADDITIVE_BY_DAY[day]).issubset(
            observed_archetypes
        )
    ):
        return None
    return jsonl


def _discard_invalid_day_outputs(day: str) -> None:
    shard = OUTPUT_ROOT / f"all-recognized-{day}.features"
    for path in (
        shard,
        Path(str(shard) + ".json"),
        OUTPUT_ROOT / f"all-recognized-{day}.identity.json",
    ):
        path.unlink(missing_ok=True)


def main() -> int:
    for required in (
        ARCHETYPES,
        COLLECTOR,
        FEATURIZER,
        FEATURE_SHARDS,
        MATCHUP_ADAPTERS,
        ASSEMBLER,
        SPLITTER,
        FILTER,
        TRAINING_MIX_ROOT / "top_ladder.v1.json",
        TRAINING_MIX_ROOT / "top_ladder_representatives.v1.json",
        EPISODES_INDEX_ROOT / "manifest.csv",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    status = OUTPUT_ROOT / "status.json"
    completed: list[dict[str, Any]] = []
    started = time()
    for index, day in enumerate(DAYS, start=1):
        existing = _validated_day(day)
        if existing is not None:
            completed.append(existing)
            continue
        _discard_invalid_day_outputs(day)
        archive = ARCHIVE_ROOT / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        _atomic(
            status,
            {
                "schema": "poke_bot.rare_route_expert_shards/v1",
                "phase": "featurizing",
                "day": day,
                "current": index,
                "total": len(DAYS),
                "completed": completed,
                "updated_at": time(),
            },
        )
        stem = f"all-recognized-{day}"
        collect = ""
        if _validated_jsonl(day) is None:
            additive_flags = " ".join(
                f"--additive-archetype {archetype}"
                for archetype in RARE_ARCHETYPES
            )
            collect = (
                f"python -u scripts/collect_top_ladder_replays.py "
                f"--start-date {day} --end-date {day} --workers 4 "
                f"--min-sequences {MIN_RECOGNIZED_RECORDS_PER_DAY} "
                "--min-recognized-seat-frac 0.0 --recognized-only "
                f"{additive_flags} "
                f"--out /output/{stem}.jsonl --skip-download --replace; "
            )
        shell = (
            "set -euo pipefail; "
            f"{collect}"
            f"python -u scripts/featurize_bootstrap_shard.py "
            f"--jsonl /output/{stem}.jsonl "
            f"--out /output/{stem}.features "
            f"--source-date {day} --workers 4 --max-in-flight 8 "
            "--max-context 320 --compact-mode temporal-expert-v1; "
            f"rm -f /output/{stem}.jsonl"
        )
        subprocess.run(
            [
                "sudo",
                "docker",
                "run",
                "--rm",
                "--label",
                "pokebot.managed-unit=rare-route-expert-shards-v1",
                "--name",
                f"pokebot-rare-expert-{day.replace('-', '')}",
                "--cpus",
                "4",
                "--memory",
                "24g",
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                "MKL_NUM_THREADS=1",
                "-e",
                "OPENBLAS_NUM_THREADS=1",
                "-v",
                f"{ARCHIVE_ROOT}:/workspace/data/episodes/raw:ro",
                "-v",
                f"{TRAINING_MIX_ROOT}:/workspace/data/training_mixes:ro",
                "-v",
                (
                    f"{EPISODES_INDEX_ROOT}:"
                    "/workspace/kaggle/input/"
                    "pokemon-tcg-ai-battle-episodes-index:ro"
                ),
                "-v",
                f"{OUTPUT_ROOT}:/output",
                "-v",
                f"{ARCHETYPES}:/workspace/poke_bot/archetypes.py:ro",
                "-v",
                f"{COLLECTOR}:/workspace/scripts/collect_top_ladder_replays.py:ro",
                "-v",
                f"{FEATURIZER}:/workspace/scripts/featurize_bootstrap_shard.py:ro",
                "-v",
                f"{FEATURE_SHARDS}:/workspace/poke_bot/feature_shards.py:ro",
                "-w",
                "/workspace",
                "--entrypoint",
                "/bin/bash",
                IMAGE,
                "-lc",
                shell,
            ],
            check=True,
        )
        shard = OUTPUT_ROOT / f"{stem}.features"
        sidecar = json.loads(
            Path(str(shard) + ".json").read_text(encoding="utf-8")
        )
        collector_meta = json.loads(
            (OUTPUT_ROOT / f"{stem}.meta.json").read_text(encoding="utf-8")
        )
        observed_additive = sorted(
            set((collector_meta.get("stats") or {}).get("record_archetypes") or ())
            & set(RARE_ARCHETYPES)
        )
        _atomic(
            OUTPUT_ROOT / f"{stem}.identity.json",
            {
                "schema": "poke_bot.rare_route_expert_shard_identity/v1",
                "day": day,
                "additive_archetypes": list(RARE_ARCHETYPES),
                "collector_contract": COLLECTOR_CONTRACT,
                "observed_additive_archetypes": observed_additive,
                "shard_sha256": sidecar.get("sha256"),
            },
        )
        completed.append(_validated_day(day) or {})

    # Split all additive evidence days once, on Elmo, before anything crosses the
    # LAN.  The resulting per-archetype shards are much smaller than the mixed
    # feature set and can be merged with the protected 20-day corpora on Inzi.
    mixed_manifest = OUTPUT_ROOT / "manifest.json"
    specialist_root = OUTPUT_ROOT / "specialists-v1"
    if not specialist_root.joinpath("SPECIALIST_CORPORA_READY.json").is_file():
        expected_dates = " ".join(
            f"--expected-date {day}" for day in DAYS
        )
        archetypes = " ".join(
            f"--archetype {archetype}" for archetype in RARE_ARCHETYPES
        )
        shell = (
            "set -euo pipefail; "
            "python -u scripts/assemble_feature_manifest.py "
            f"--staging-dir /history --out /history/{mixed_manifest.name} "
            f"{expected_dates} "
            "--compact-mode temporal-expert-v1 "
            "--expected-max-context 320 "
            # Historical public episodes always preserve demonstrated
            # temporal actions, but some older exports omit private auxiliary
            # labels.  Retain those rows as valid policy supervision; the
            # later merge audits auxiliary coverage separately.
            "--require-target-coverage temporal_action_rows; "
            "python -u scripts/split_expert_manifest_by_archetype.py "
            f"--source-manifest /history/{mixed_manifest.name} "
            "--output-root /history/specialists-v1 "
            f"{archetypes} --minimum-decisions 1 --progress-every 1000"
        )
        subprocess.run(
            [
                "sudo",
                "docker",
                "run",
                "--rm",
                "--label",
                "pokebot.managed-unit=rare-route-expert-shards-v1",
                "--name",
                "pokebot-rare-expert-split-v1",
                "--cpus",
                "4",
                "--memory",
                "24g",
                "-e",
                "OMP_NUM_THREADS=1",
                "-e",
                "MKL_NUM_THREADS=1",
                "-e",
                "OPENBLAS_NUM_THREADS=1",
                "-v",
                f"{OUTPUT_ROOT}:/history",
                "-v",
                f"{ARCHETYPES}:/workspace/poke_bot/archetypes.py:ro",
                "-v",
                f"{FEATURE_SHARDS}:/workspace/poke_bot/feature_shards.py:ro",
                "-v",
                (
                    f"{MATCHUP_ADAPTERS}:"
                    "/workspace/poke_bot/matchup_adapters.py:ro"
                ),
                "-v",
                (
                    f"{ASSEMBLER}:"
                    "/workspace/scripts/assemble_feature_manifest.py:ro"
                ),
                "-v",
                (
                    f"{SPLITTER}:"
                    "/workspace/scripts/split_expert_manifest_by_archetype.py:ro"
                ),
                "-v",
                (
                    f"{FILTER}:"
                    "/workspace/scripts/filter_feature_manifest.py:ro"
                ),
                "-w",
                "/workspace",
                "--entrypoint",
                "/bin/bash",
                IMAGE,
                "-lc",
                shell,
            ],
            check=True,
        )
    if not specialist_root.joinpath("SPECIALIST_CORPORA_READY.json").is_file():
        raise RuntimeError("rare-route specialist split did not seal")
    _atomic(
        status,
        {
            "schema": "poke_bot.rare_route_expert_shards/v1",
            "phase": "ready",
            "current": len(completed),
            "total": len(DAYS),
            "completed": completed,
            "elapsed_seconds": time() - started,
            "updated_at": time(),
            "live_corpus_modified": False,
            "specialist_split": str(specialist_root),
            "specialist_split_ready": str(
                specialist_root / "SPECIALIST_CORPORA_READY.json"
            ),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
