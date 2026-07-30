#!/usr/bin/env bash
# Build the inactive Archaludon guide corpus over the full schema-7 identity
# corpus. No latest-20 or schema-6 artifact is accepted by this contract.
set -euo pipefail

mode="${1:-run}"
if [[ "$mode" != "run" && "$mode" != "--check" ]]; then
  echo "usage: $0 [--check]" >&2
  exit 2
fi

source_root="${POKEBOT_GUIDE_SOURCE:-/home/admin/pokebot-archaludon-r56-src-v1}"
archive_root="${POKEBOT_EPISODE_ARCHIVE:-/mnt/Main/main/poke-bot-agent/archive/episode-days}"
identity_root="${POKEBOT_ARCHALUDON_IDENTITY_ROOT:-/mnt/Main/main/poke-bot-agent/archive/archaludon-ex-full-public-2026-06-16_2026-07-29-schema7-r56-v1}"
output_root="${POKEBOT_ARCHALUDON_GUIDE_OUTPUT:-/mnt/Main/main/poke-bot-agent/archive/archaludon-ex-guide-full-public-schema7-r56-v1}"
cg_runtime="${POKEBOT_CG_RUNTIME:-/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1}"
source_lock="${POKEBOT_ARCHALUDON_SOURCE_LOCK:-$source_root/state/archaludon_ex_schema7_source_lock_v1.json}"
identity_ready="$identity_root/ARCHALUDON_EX_FULL_PUBLIC_CORPUS_READY.json"
identity_sources="$identity_root/SOURCE_ARCHIVES.json"
status="$output_root/status/window.json"
start_date="2026-06-16"
end_date="2026-07-29"
minimum_records="16639"

test -f "$cg_runtime/cg/__init__.py"
test -s "$source_root/scripts/materialize_authoritative_guide_window_parallel.py"
test -s "$source_root/scripts/materialize_authoritative_alakazam_day.py"
test -s "$source_root/scripts/finalize_current_deck_guide_window.py"
test -s "$source_root/poke_bot/archaludon_ex_heuristics.py"
test -s "$source_root/data/training_mixes/specialist_representatives.v1.json"
test -s "$source_lock"
test -s "$identity_ready"
test -s "$identity_sources"

python3 - \
  "$identity_ready" \
  "$identity_sources" \
  "$source_lock" \
  "$source_root" \
  "$cg_runtime" \
  "$start_date" \
  "$end_date" \
  "$minimum_records" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

identity_path = Path(sys.argv[1])
sources_path = Path(sys.argv[2])
lock_path = Path(sys.argv[3])
source_root = Path(sys.argv[4])
cg_runtime = Path(sys.argv[5])
start, end, minimum = sys.argv[6], sys.argv[7], int(sys.argv[8])
ready = json.loads(identity_path.read_text(encoding="utf-8"))
sources = json.loads(sources_path.read_text(encoding="utf-8"))
lock = json.loads(lock_path.read_text(encoding="utf-8"))
daily = list(ready.get("daily_receipts") or [])
source_rows = list(sources.get("archives") or [])
source_policy = dict(ready.get("source_policy") or {})
lock_digest = "sha256:" + hashlib.sha256(lock_path.read_bytes()).hexdigest()
required_locked_files = {
    "config/deck_guides/archaludon-ex.yaml",
    "data/training_mixes/specialist_representatives.v1.json",
    "poke_bot/archaludon_ex_heuristics.py",
    "poke_bot/authoritative_visual_trace.py",
    "poke_bot/dataset.py",
    "poke_bot/feature_shards.py",
    "scripts/materialize_authoritative_guide_window_parallel.py",
    "scripts/materialize_authoritative_alakazam_day.py",
    "scripts/finalize_current_deck_guide_window.py",
    "scripts/assemble_feature_manifest.py",
    "ops/elmo/build_archaludon_ex_full_public_schema7_guide_corpus.sh",
}
locked_files = dict(lock.get("files") or {})
if not required_locked_files.issubset(locked_files):
    raise SystemExit("source lock omits an Archaludon guide dependency")
for relative, expected_digest in locked_files.items():
    path = source_root / relative
    if not path.is_file() or (
        "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        != expected_digest
    ):
        raise SystemExit(f"source-lock mismatch: {relative}")
cg_library = cg_runtime / "cg/libcg.so"
if (
    not cg_library.is_file()
    or "sha256:" + hashlib.sha256(cg_library.read_bytes()).hexdigest()
    != lock.get("cg_library_sha256")
):
    raise SystemExit("source-lock mismatch: cg/libcg.so")
if (
    ready.get("schema") != "poke_bot.archaludon_ex_full_public_corpus/v2"
    or ready.get("specialist_id") != "archaludon-ex"
    or ready.get("status") != "ready_checksum_validated"
    or source_policy.get("date_start") != start
    or source_policy.get("date_end") != end
    or int(source_policy.get("days") or 0) != 44
    or source_policy.get("schema6_feature_reuse_allowed") is not False
    or int(ready.get("records") or 0) < minimum
    or int(ready.get("unique_episodes") or 0) < minimum
    or int(ready.get("minimum_matching_games") or 0) != minimum
    or ready.get("minimum_matching_games_met") is not True
    or ready.get("minimum_unique_episode_games_met") is not True
    or int(ready.get("dataset_schema") or -1) != 7
    or int(ready.get("feature_schema") or -1) != 5
    or int(ready.get("guide_rows") or -1) != 0
    or ready.get("non_guide_corpus") is not True
    or ready.get("immutable_staging_artifact") is not True
    or len(daily) != 44
    or any(
        row.get("source_kind")
        != "original_public_archive_schema7_rematerialization"
        for row in daily
    )
    or sources.get("schema")
    != "poke_bot.archaludon_ex_full_public_sources/v2"
    or sources.get("all_archive_checksums_validated") is not True
    or sources.get("all_archive_manifest_memberships_validated") is not True
    or sources.get("schema6_feature_reuse_allowed") is not False
    or list(sources.get("reused_feature_dates") or []) != []
    or len(source_rows) != 44
    or lock.get("schema") != "poke_bot.archaludon_ex_schema7_source_lock/v1"
    or lock.get("status") != "locked_checksum_validated"
    or int(lock.get("goal_revision") or -1) < 56
    or int(lock.get("dataset_schema") or -1) != 7
    or int(lock.get("feature_schema") or -1) != 5
    or (
        (ready.get("build_provenance") or {}).get("source_lock") or {}
    ).get("sha256") != lock_digest
):
    raise SystemExit("full-public schema-7 Archaludon identity is not ready")
PY

if [[ "$mode" == "--check" ]]; then
  echo "archaludon-ex full-public schema-7 guide build preflight passed"
  exit 0
fi

mkdir -p "$output_root/status"
export CG_LIB_PATH="$cg_runtime"
cd "$source_root"
python3 scripts/materialize_authoritative_guide_window_parallel.py \
  --start "$start_date" \
  --end "$end_date" \
  --archive-dir "$archive_root" \
  --out-dir "$output_root" \
  --status "$status" \
  --required-archetype archaludon-ex \
  --current-deck-guide archaludon-ex \
  --day-parallelism 4 \
  --workers-per-day 3 \
  --max-in-flight-per-day 6 \
  --max-context 320 \
  --memory-floor-gib 16 \
  --min-records 0

python3 scripts/finalize_current_deck_guide_window.py \
  --status "$status" \
  --out-dir "$output_root" \
  --start "$start_date" \
  --end "$end_date" \
  --specialist-id archaludon-ex \
  --guide-version archaludon-ex-north-star-v1 \
  --minimum-records "$minimum_records"

python3 - \
  "$identity_ready" \
  "$source_lock" \
  "$output_root/CURRENT_DECK_GUIDE_CORPUS_READY.json" \
  "$output_root/ARCHALUDON_EX_GUIDE_CORPUS_READY.json" <<'PY'
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

identity_path = Path(sys.argv[1])
source_lock_path = Path(sys.argv[2])
guide_path = Path(sys.argv[3])
validation_path = Path(sys.argv[4])
identity = json.loads(identity_path.read_text(encoding="utf-8"))
guide = json.loads(guide_path.read_text(encoding="utf-8"))
expected = {
    str(row["date"]): {
        "records": int(row["records"]),
        "decisions": int(row["decisions"]),
    }
    for row in identity.get("daily_receipts") or []
}
actual = {
    str(row["date"]): {
        "records": int(row["records"]),
        "decisions": int(row["decisions"]),
        "guide_rows": int(row["guide_rows"]),
    }
    for row in guide.get("daily_shards") or []
}
actual_identity = {
    day: {"records": row["records"], "decisions": row["decisions"]}
    for day, row in actual.items()
}
nonzero_source_days = {
    day for day, row in expected.items() if row["records"] > 0
}
positive_guide_days = {
    day for day, row in actual.items() if row["guide_rows"] > 0
}
zero_guide_nonzero_source_days = sorted(
    nonzero_source_days - positive_guide_days
)
if (
    guide.get("status") != "ready"
    or guide.get("specialist_id") != "archaludon-ex"
    or guide.get("guide_version") != "archaludon-ex-north-star-v1"
    or guide.get("date_start") != "2026-06-16"
    or guide.get("date_end") != "2026-07-29"
    or int(guide.get("days") or 0) != 44
    or int(guide.get("records") or 0) != int(identity["records"])
    or int(guide.get("records") or 0) < 16639
    or int(guide.get("decisions") or 0) != int(identity["decisions"])
    or int(guide.get("guide_rows") or 0) <= 0
    or actual_identity != expected
    or not positive_guide_days.issubset(nonzero_source_days)
    or sum(row["guide_rows"] for row in actual.values())
    != int(guide.get("guide_rows") or 0)
    or guide.get("source_policy") is not None
    or guide.get("active_training_modified") is not False
):
    raise SystemExit("sealed Archaludon guide corpus failed its final contract")

sidecars = sorted(guide_path.parent.glob("*.features.json"))
if len(sidecars) != 44:
    raise SystemExit("guide corpus does not contain 44 daily schema-7 sidecars")
for sidecar in sidecars:
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    shard = sidecar.with_suffix("")
    with shard.open("rb") as stream:
        header = pickle.load(stream)
    if (
        int(metadata.get("dataset_schema") or -1) != 7
        or int(metadata.get("feature_schema") or -1) != 5
        or int(header.get("dataset_schema") or -1) != 7
        or int(header.get("feature_schema") or -1) != 5
    ):
        raise SystemExit(f"non-schema-7 guide shard: {shard}")

def sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

validation = {
    "schema": "poke_bot.archaludon_ex_guide_corpus_validation/v2",
    "status": "ready_checksum_validated",
    "specialist_id": "archaludon-ex",
    "guide_version": "archaludon-ex-north-star-v1",
    "source_window": {
        "start": "2026-06-16",
        "end": "2026-07-29",
        "days": 44,
    },
    "records": int(guide["records"]),
    "unique_episodes": int(identity["unique_episodes"]),
    "mirror_episodes": int(identity["mirror_episodes"]),
    "single_seat_episodes": int(identity["single_seat_episodes"]),
    "decisions": int(guide["decisions"]),
    "guide_rows": int(guide["guide_rows"]),
    "dataset_schema": 7,
    "feature_schema": 5,
    "minimum_matching_games": 16639,
    "minimum_unique_episode_games": 16639,
    "minimum_unique_episode_games_met": True,
    "schema6_feature_reuse_allowed": False,
    "daily_shards": len(actual),
    "nonzero_source_days": sorted(nonzero_source_days),
    "positive_guide_days": sorted(positive_guide_days),
    "zero_guide_nonzero_source_days": zero_guide_nonzero_source_days,
    "zero_guide_day_policy": (
        "exact_causal_mask_not_zero_for_unsupported_single_choice_"
        "ambiguous_or_low_margin_stages"
    ),
    "identity_ready_receipt": str(identity_path),
    "identity_ready_receipt_sha256": sha256(identity_path),
    "source_lock": str(source_lock_path),
    "source_lock_sha256": sha256(source_lock_path),
    "guide_ready_receipt": str(guide_path),
    "guide_ready_receipt_sha256": sha256(guide_path),
    "active_training_modified": False,
}
partial = validation_path.with_name(
    f".{validation_path.name}.partial.{os.getpid()}"
)
with partial.open("x", encoding="utf-8") as handle:
    json.dump(validation, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
os.replace(partial, validation_path)
print(json.dumps(validation, sort_keys=True))
PY
