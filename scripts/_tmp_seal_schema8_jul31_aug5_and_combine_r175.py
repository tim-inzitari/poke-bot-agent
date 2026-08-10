#!/usr/bin/env python3
"""Watch Jul31-Aug5 schema8+combo remat; seal; sync; combine Jul24-Aug5 on train."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

OUT = Path(
    "/mnt/Main/main/poke-bot-agent/archive/"
    "teal-mask-ogerpon-ex-guide-corpus-full-v5-r170-schema8-jul31-aug5"
)
PRIOR = Path(
    "/mnt/Main/main/poke-bot-agent/archive/"
    "teal-mask-ogerpon-ex-guide-corpus-full-v5-r170-schema8-jul24-30"
)
COMBINED = Path(
    "/mnt/Main/main/poke-bot-agent/archive/"
    "teal-mask-ogerpon-ex-guide-corpus-full-v5-r170-schema8-jul24-aug5"
)
STATUS = OUT / "status/window_schema8_jul31_aug5.json"
DAYS_NEW = [
    "2026-07-31",
    "2026-08-01",
    "2026-08-02",
    "2026-08-03",
    "2026-08-04",
    "2026-08-05",
]
DAYS_PRIOR = [f"2026-07-{d:02d}" for d in range(24, 31)]
DAYS_ALL = DAYS_PRIOR + DAYS_NEW
SPECIALIST = "teal-mask-ogerpon-ex"
SSH_KEY = "/home/admin/.ssh/id_ed25519_poke_lan"
TRAIN_EXPERT = (
    "/home/inzi/poke-bot-agent/data/bootstrap/"
    "expert-slop-box-schema8-jul24-aug5-r175/teal-mask-ogerpon-ex"
)
EXPECTED = {
    "2026-07-31": 135,
    "2026-08-01": 173,
    "2026-08-02": 88,
    "2026-08-03": 106,
    "2026-08-04": 146,
    "2026-08-05": 54,
}


def _read(path: Path):
    return json.loads(path.read_text())


def _sha_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _day_ok(root: Path, day: str) -> tuple[bool, dict]:
    feat = root / f"{SPECIALIST}-{day}.features"
    receipt = root / f"{SPECIALIST}-{day}.features.receipt.json"
    if not feat.is_file() or not receipt.is_file():
        return False, {"day": day, "present": False}
    rec = _read(receipt)
    stats = rec.get("stats") or {}
    records = int(stats.get("records_kept") or 0)
    decisions = int(stats.get("decisions_kept") or 0)
    schema = int((rec.get("schemas") or {}).get("dataset") or 0)
    combo_rows = int(((stats.get("target_coverage") or {}).get("combo_state_rows")) or 0)
    ok = records > 0 and decisions > 0 and schema >= 8 and combo_rows > 0
    return ok, {
        "day": day,
        "present": True,
        "records": records,
        "decisions": decisions,
        "dataset_schema": schema,
        "combo_state_rows": combo_rows,
        "sha256": _sha_file(feat),
        "bytes": feat.stat().st_size,
        "ok": ok,
    }


def _ssh(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            "ssh",
            "-o",
            "IdentitiesOnly=yes",
            "-i",
            SSH_KEY,
            "train",
            cmd,
        ],
        check=check,
        capture_output=True,
        text=True,
    )


def _scp(files: list[str], dest: str) -> None:
    subprocess.run(
        [
            "scp",
            "-o",
            "IdentitiesOnly=yes",
            "-i",
            SSH_KEY,
            *files,
            dest,
        ],
        check=True,
    )


def main() -> int:
    print("[seal-watch-jul31-aug5] start", flush=True)
    while True:
        if not STATUS.is_file():
            print("[seal-watch] waiting status", flush=True)
            time.sleep(30)
            continue
        st = _read(STATUS)
        state = st.get("state")
        completed = list(st.get("completed") or [])
        print(
            f"[seal-watch] state={state} completed={len(completed)} "
            f"current={st.get('current_dates')}",
            flush=True,
        )
        for row in completed:
            if int(row.get("records") or 0) <= 0 or row.get("zero_guide_rows"):
                day = row["date"]
                q = OUT / (
                    "quarantine_empty_post_cg_"
                    + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                )
                q.mkdir(parents=True, exist_ok=True)
                for ext in ("", ".json", ".receipt.json"):
                    src = OUT / f"{SPECIALIST}-{day}.features{ext}"
                    if src.exists():
                        src.rename(q / src.name)
                raise SystemExit(
                    f"EMPTY_SEAL_AFTER_CG day={day} records={row.get('records')}"
                )
        if state != "complete":
            time.sleep(45)
            continue
        days_new = []
        for day in DAYS_NEW:
            ok, info = _day_ok(OUT, day)
            days_new.append(info)
            if not ok:
                raise SystemExit(f"DAY_NOT_OK {info}")
            exp = EXPECTED.get(day)
            if exp and int(info["records"]) < max(1, int(exp * 0.5)):
                print(
                    f"[seal-watch] WARN {day} records={info['records']} "
                    f"expected~{exp}",
                    flush=True,
                )
        proof_day = DAYS_NEW[0]
        env = os.environ.copy()
        env["PYTHONPATH"] = "/home/admin/pokebot-expert-guide-src-schema8-full-r170"
        env["CG_LIB_PATH"] = "/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1"
        code = f"""
import json
from pathlib import Path
from poke_bot.feature_shards import iter_feature_shard
path = Path({str(OUT / f'{SPECIALIST}-{proof_day}.features')!r})
contexts=set(); stops=0; rows=0; games=0
for seq in iter_feature_shard(path):
    games += 1
    for d in seq.decisions:
        for st in getattr(d, "policy_stages", []) or []:
            rows += 1
            contexts.add(int(getattr(st, "select_context", -1)))
            stops += int(bool(getattr(st, "selected_is_stop", False)))
print(json.dumps({{"games":games,"stage_rows":rows,"select_contexts":sorted(contexts),"any_setup": any(c in (0,1) for c in contexts),"selected_is_stop_true":stops}}, sort_keys=True))
"""
        proc = subprocess.run(
            ["python3", "-c", code], env=env, capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise SystemExit(f"SELECT_CONTEXT_PROOF_FAILED {proc.stderr[-500:]}")
        proof = json.loads(proc.stdout.strip().splitlines()[-1])
        proof["ok"] = bool(proof.get("any_setup")) and int(proof.get("stage_rows") or 0) > 0
        if not proof["ok"]:
            raise SystemExit(f"SELECT_CONTEXT_PROOF_FAILED {proof}")

        totals_new = {
            "records": sum(int(d["records"]) for d in days_new),
            "decisions": sum(int(d["decisions"]) for d in days_new),
            "combo_state_rows": sum(int(d["combo_state_rows"]) for d in days_new),
            "days": len(days_new),
        }
        seal_new = {
            "schema": "poke_bot.slop_box_schema8_jul31_aug5_sealed_r175/v1",
            "status": "sealed_ready",
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "window": {"start": DAYS_NEW[0], "end": DAYS_NEW[-1]},
            "out_dir": str(OUT),
            "catalog_sha256": "sha256:4d6ae1713cdd09b3fbb76602638ba5c58be2c8fe0a157966d5c022ef8bfcf1c1",
            "cg_lib_path": "/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1",
            "days": days_new,
            "totals": totals_new,
            "setup_board_labels_present": True,
            "any_setup_labels": True,
            "combo_state_labels_present": True,
            "select_context_proof": proof,
            "packing_schema": 8,
            "records": totals_new["records"],
            "records_kept": totals_new["records"],
            "bootstrap_held": True,
        }
        seal_new_path = OUT / "status/schema8_jul31_aug5_sealed_r175.json"
        seal_new_path.write_text(json.dumps(seal_new, indent=2, sort_keys=True) + "\n")
        print("SEALED_NEW", seal_new_path, totals_new, flush=True)

        COMBINED.mkdir(parents=True, exist_ok=True)
        (COMBINED / "status").mkdir(exist_ok=True)
        catalog_src = OUT / "PUBLIC_DECK_ARCHETYPE_CATALOG.json"
        (COMBINED / "PUBLIC_DECK_ARCHETYPE_CATALOG.json").write_bytes(
            catalog_src.read_bytes()
        )
        days_all = []
        for day in DAYS_PRIOR:
            ok, info = _day_ok(PRIOR, day)
            if not ok:
                raise SystemExit(f"PRIOR_DAY_NOT_OK {info}")
            days_all.append(info)
            for ext in ("", ".json", ".receipt.json"):
                src = PRIOR / f"{SPECIALIST}-{day}.features{ext}"
                dst = COMBINED / src.name
                if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                    dst.write_bytes(src.read_bytes())
        for day in DAYS_NEW:
            ok, info = _day_ok(OUT, day)
            days_all.append(info)
            for ext in ("", ".json", ".receipt.json"):
                src = OUT / f"{SPECIALIST}-{day}.features{ext}"
                dst = COMBINED / src.name
                dst.write_bytes(src.read_bytes())
        totals_all = {
            "records": sum(int(d["records"]) for d in days_all),
            "decisions": sum(int(d["decisions"]) for d in days_all),
            "combo_state_rows": sum(int(d["combo_state_rows"]) for d in days_all),
            "days": len(days_all),
        }
        seal_all = {
            "schema": "poke_bot.slop_box_schema8_jul24_aug5_sealed_r175/v1",
            "status": "sealed_ready",
            "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
            "window": {"start": DAYS_ALL[0], "end": DAYS_ALL[-1]},
            "out_dir": str(COMBINED),
            "sources": {"jul24_30": str(PRIOR), "jul31_aug5": str(OUT)},
            "catalog_sha256": "sha256:4d6ae1713cdd09b3fbb76602638ba5c58be2c8fe0a157966d5c022ef8bfcf1c1",
            "cg_lib_path": "/mnt/Main/main/poke-bot-agent/engine-runtimes/znver3-v1",
            "days": days_all,
            "totals": totals_all,
            "setup_board_labels_present": True,
            "any_setup_labels": True,
            "combo_state_labels_present": True,
            "packing_schema": 8,
            "records": totals_all["records"],
            "records_kept": totals_all["records"],
            "bootstrap_held": True,
            "do_not_start_bootstrap": True,
        }
        seal_all_path = COMBINED / "status/schema8_jul24_aug5_sealed_r175.json"
        seal_all_path.write_text(json.dumps(seal_all, indent=2, sort_keys=True) + "\n")
        print("SEALED_COMBINED", seal_all_path, totals_all, flush=True)

        _ssh(f"mkdir -p {TRAIN_EXPERT}")
        files = [
            str(COMBINED / "PUBLIC_DECK_ARCHETYPE_CATALOG.json"),
            str(seal_all_path),
            str(seal_new_path),
        ]
        for day in DAYS_ALL:
            for ext in ("", ".json", ".receipt.json"):
                files.append(str(COMBINED / f"{SPECIALIST}-{day}.features{ext}"))
        _scp(files, f"train:{TRAIN_EXPERT}/")
        _scp(
            [str(seal_all_path)],
            "train:/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-schema8-jul24-aug5-sealed-r175.json",
        )
        _scp(
            [str(seal_new_path)],
            "train:/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-schema8-jul31-aug5-sealed-r175.json",
        )
        remat = {
            "schema": "poke_bot.slop_box_full_archive_rematerialize_r170/v1",
            "status": "jul31_aug5_schema8_combo_sealed_jul24_aug5_combined",
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "fixed_catalog": {
                "sha256": "sha256:4d6ae1713cdd09b3fbb76602638ba5c58be2c8fe0a157966d5c022ef8bfcf1c1",
                "elmo": str(OUT / "PUBLIC_DECK_ARCHETYPE_CATALOG.fixed-4d6ae171.json"),
            },
            "jul31_aug5": seal_new,
            "jul24_aug5": {
                "records": totals_all["records"],
                "decisions": totals_all["decisions"],
                "combo_state_rows": totals_all["combo_state_rows"],
                "out_dir": str(COMBINED),
                "train_expert": TRAIN_EXPERT,
            },
            "no_ready": True,
            "no_rl": True,
            "bootstrap_held": True,
            "defect_days": [],
        }
        remat_path = Path("/tmp/slop-box-full-archive-rematerialize-r170.json")
        remat_path.write_text(json.dumps(remat, indent=2, sort_keys=True) + "\n")
        _scp(
            [str(remat_path)],
            "train:/home/inzi/poke-bot-agent/outputs/state/"
            "slop-box-full-archive-rematerialize-r170.json",
        )
        print("SYNCED_TO_TRAIN", TRAIN_EXPERT, flush=True)
        print("READY_FOR_PACK_AND_MARKUP", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
