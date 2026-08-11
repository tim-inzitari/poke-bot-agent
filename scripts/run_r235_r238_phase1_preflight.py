#!/usr/bin/env python3
"""Run the R235/R240/R242/R246 two-lane package preflight.

This command only validates an explicit staged directory, archive, and member
manifest.  It never contacts Kaggle, starts a managed service, or changes a
selector.  Actual probe commands are JSON argv arrays and run only through a
fresh noninteractive exact-child session/process group.  ``--dry-run`` accepts
offline fixture JSON instead and produces a receipt that cannot satisfy an
owner execution gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.r235_kaggle_phase1_preflight import (  # noqa: E402
    ImmutableReceiptError,
    R235PreflightFailure,
    R235PreflightInputs,
    R235PreflightLimits,
    run_r235_phase1_preflight,
)


def _read_json(path: Path, *, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(f"{label} is not readable JSON: {path}") from exc


def _argv_from_json(path: Path) -> list[str]:
    payload = _read_json(path, label="probe command")
    if not isinstance(payload, list) or not payload or not all(
        isinstance(part, str) and part for part in payload
    ):
        raise argparse.ArgumentTypeError("probe command JSON must be a non-empty string argv array")
    return list(payload)


def _exports_from_json(path: Path) -> list[str]:
    payload = _read_json(path, label="offline exports")
    exports = payload.get("exports") if isinstance(payload, dict) else payload
    if not isinstance(exports, list) or not all(isinstance(item, str) for item in exports):
        raise argparse.ArgumentTypeError("offline exports JSON must be an exports string array")
    return list(exports)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-archive-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--actual-gate-receipts-dir",
        type=Path,
        help=(
            "existing physical directory for actual-only derived binder receipts; "
            "required outside --dry-run"
        ),
    )
    parser.add_argument("--entrypoint-relative-path", default="main.py")
    parser.add_argument(
        "--r225-contract",
        type=Path,
        default=ROOT / "state/alakazam-r222-shared-tree-eight-lane-kaggle-diagnostic-r225.json",
    )
    parser.add_argument(
        "--r236-contract", type=Path, default=ROOT / "state/canonical-libcg-r236.json"
    )
    parser.add_argument("--probe-timeout-seconds", type=float, required=True)
    parser.add_argument("--term-grace-seconds", type=float, required=True)
    parser.add_argument("--kill-grace-seconds", type=float, required=True)
    parser.add_argument("--max-startup-seconds", type=float, required=True)
    parser.add_argument("--max-decision-latency-seconds", type=float, required=True)
    parser.add_argument("--max-full-game-cumulative-seconds", type=float, required=True)
    parser.add_argument("--min-throughput-decisions-per-second", type=float, required=True)
    parser.add_argument("--min-throughput-decision-count", type=int, required=True)
    parser.add_argument("--probe-command-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--offline-probe-json", type=Path)
    parser.add_argument("--offline-exports-json", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dry_run:
        if args.probe_command_json is not None:
            raise SystemExit("--dry-run cannot accept --probe-command-json")
        if args.offline_probe_json is None or args.offline_exports_json is None:
            raise SystemExit(
                "--dry-run requires --offline-probe-json and --offline-exports-json"
            )
        probe_command = None
        offline_probe = _read_json(args.offline_probe_json, label="offline probe")
        if not isinstance(offline_probe, dict):
            raise SystemExit("offline probe JSON must be an object")
        offline_exports = _exports_from_json(args.offline_exports_json)
        if args.actual_gate_receipts_dir is not None:
            raise SystemExit("--dry-run cannot emit --actual-gate-receipts-dir")
    else:
        if args.probe_command_json is None:
            raise SystemExit("actual preflight requires --probe-command-json")
        if args.offline_probe_json is not None or args.offline_exports_json is not None:
            raise SystemExit("actual preflight cannot consume offline fixture files")
        if args.actual_gate_receipts_dir is None:
            raise SystemExit("actual preflight requires --actual-gate-receipts-dir")
        probe_command = _argv_from_json(args.probe_command_json)
        offline_probe = None
        offline_exports = None
    inputs = R235PreflightInputs(
        stage_dir=args.stage_dir,
        archive_path=args.archive,
        manifest_path=args.manifest,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        receipt_path=args.receipt,
        r225_contract_path=args.r225_contract,
        r236_contract_path=args.r236_contract,
        entrypoint_relative_path=args.entrypoint_relative_path,
        actual_gate_receipts_dir=args.actual_gate_receipts_dir,
    )
    limits = R235PreflightLimits(
        probe_timeout_seconds=args.probe_timeout_seconds,
        term_grace_seconds=args.term_grace_seconds,
        kill_grace_seconds=args.kill_grace_seconds,
        max_startup_seconds=args.max_startup_seconds,
        max_decision_latency_seconds=args.max_decision_latency_seconds,
        max_full_game_cumulative_seconds=args.max_full_game_cumulative_seconds,
        min_throughput_decisions_per_second=args.min_throughput_decisions_per_second,
        min_throughput_decision_count=args.min_throughput_decision_count,
    )
    try:
        receipt = run_r235_phase1_preflight(
            inputs=inputs,
            limits=limits,
            probe_command=probe_command,
            dry_run=args.dry_run,
            offline_probe_payload=offline_probe,
            offline_exports=offline_exports,
        )
    except R235PreflightFailure as exc:
        print(
            json.dumps(
                {"status": "failed", "receipt": str(exc.path), "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except ImmutableReceiptError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
