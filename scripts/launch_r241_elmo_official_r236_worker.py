#!/usr/bin/env python3
"""Preflight or serve the isolated r241 Elmo official-r236 collection worker.

This command deliberately has no endpoint fallback.  It accepts only the new
``192.168.1.143:8767`` worker identity, validates all host-path-bound r241
receipts, and then (only with the explicit ``serve`` subcommand) delegates to
the generic remote worker with a sealed direct-policy environment.

``preflight`` performs local file checks and writes immutable receipts.  It
does not connect to Elmo, bind a port, create a worker pool, call native libcg
functions, or start a game.  This source file is a launcher template input;
running it is still an operator action after the r241 activation gates pass.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.r241_elmo_official_r236_remote_worker import (
    DEFAULT_ELMO_RECEIPT_DIR,
    ELMO_R241_ENDPOINT,
    ELMO_R241_ENDPOINT_PORT,
    R241ElmoPreflight,
    R241ElmoRemoteWorkerError,
    build_r241_elmo_preflight_receipts,
    preflight_r241_elmo_remote_collection,
    validate_r241_elmo_activation_overlay,
    validate_r241_elmo_preflight_manifest,
    write_r241_elmo_preflight_receipts,
)


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--endpoint",
        required=True,
        help=f"Must be the explicit eligible endpoint {ELMO_R241_ENDPOINT}",
    )
    parser.add_argument(
        "--source-snapshot-root",
        type=Path,
        required=True,
        help="Authenticated immutable r241 source execution root on Elmo",
    )
    parser.add_argument(
        "--source-snapshot-manifest",
        type=Path,
        required=True,
        help="The snapshot-local r241-source-snapshot-manifest.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cg-lib-path", type=Path, required=True)
    parser.add_argument("--adapter-receipt", type=Path, required=True)
    parser.add_argument("--learner-matchup-tree", type=Path, required=True)
    parser.add_argument(
        "--baselines-root",
        type=Path,
        required=True,
        help="Receipt-bound external baseline payload root; must equal POKEBOT_BASELINES_DIR",
    )
    parser.add_argument(
        "--checkpoint-transport-host-root",
        type=Path,
        required=True,
        help=(
            "Elmo host-side content-addressed checkpoint staging root; it is "
            "the sole /workspace/checkpoint container mount"
        ),
    )
    parser.add_argument(
        "--checkpoint-transport-staging-receipt",
        type=Path,
        required=True,
        help="Create-only Elmo checkpoint-transport staging receipt",
    )
    parser.add_argument(
        "--checkpoint-transport-staging-receipt-sha256",
        required=True,
        help="Exact SHA-256 of --checkpoint-transport-staging-receipt",
    )
    parser.add_argument(
        "--source-staging-receipt",
        type=Path,
        required=True,
        help="Create-only Elmo source-snapshot staging receipt",
    )
    parser.add_argument(
        "--source-staging-receipt-sha256",
        required=True,
        help="Exact SHA-256 of --source-staging-receipt",
    )
    parser.add_argument(
        "--baseline-staging-receipt",
        type=Path,
        required=True,
        help="Create-only Elmo external-baseline staging receipt",
    )
    parser.add_argument(
        "--baseline-staging-receipt-sha256",
        required=True,
        help="Exact SHA-256 of --baseline-staging-receipt",
    )
    parser.add_argument(
        "--canonical-roster-receipt",
        type=Path,
        required=True,
        help="Externally derived full public-baseline roster receipt",
    )
    parser.add_argument(
        "--canonical-roster-receipt-sha256",
        required=True,
        help="Exact SHA-256 of --canonical-roster-receipt",
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        required=True,
        help=(
            "Dedicated canonical :8767 receipt directory (must be "
            f"{DEFAULT_ELMO_RECEIPT_DIR})"
        ),
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the deterministic manifest after local preflight",
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser(
        "preflight", help="validate local r241 inputs and write receipts only"
    )
    _common_arguments(preflight)
    serve = subparsers.add_parser(
        "serve", help="preflight then start only the isolated :8767 worker"
    )
    _common_arguments(serve)
    serve.add_argument(
        "--activation-overlay",
        type=Path,
        required=True,
        help=(
            "Read-only Elmo mirror of the one byte-identical r241 activation "
            "overlay; unavailable to offline preflight"
        ),
    )
    serve.add_argument(
        "--activation-overlay-sha256",
        required=True,
        help="Shared SHA-256 of the byte-identical Inzi/Elmo activation overlay",
    )
    serve.add_argument(
        "worker_args",
        nargs=argparse.REMAINDER,
        help="Optional generic-worker capacity flags after --; endpoint/model flags are forbidden",
    )
    return parser.parse_args(argv)


@contextmanager
def _sealed_process_environment(environment: dict[str, str]) -> Iterator[None]:
    """Run the generic worker without a parent shell leaking controls into it."""

    prior = dict(os.environ)
    os.environ.clear()
    os.environ.update(environment)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(prior)


def _run_preflight(args: argparse.Namespace) -> tuple[R241ElmoPreflight, dict]:
    receipt_dir = args.receipt_dir.expanduser().resolve()
    if receipt_dir != DEFAULT_ELMO_RECEIPT_DIR:
        raise R241ElmoRemoteWorkerError(
            "the :8767 worker must write its receipt set only to the canonical "
            f"Elmo path {DEFAULT_ELMO_RECEIPT_DIR}, got {receipt_dir}"
        )
    preflight = preflight_r241_elmo_remote_collection(
        endpoint=args.endpoint,
        repo_root=args.source_snapshot_root,
        source_snapshot_manifest=args.source_snapshot_manifest,
        checkpoint=args.checkpoint,
        cg_lib_path=args.cg_lib_path,
        adapter_receipt=args.adapter_receipt,
        learner_matchup_tree=args.learner_matchup_tree,
        baselines_root=args.baselines_root,
        checkpoint_transport_host_root=args.checkpoint_transport_host_root,
        checkpoint_transport_staging_receipt=args.checkpoint_transport_staging_receipt,
        checkpoint_transport_staging_receipt_sha256=args.checkpoint_transport_staging_receipt_sha256,
        source_staging_receipt=args.source_staging_receipt,
        source_staging_receipt_sha256=args.source_staging_receipt_sha256,
        baseline_staging_receipt=args.baseline_staging_receipt,
        baseline_staging_receipt_sha256=args.baseline_staging_receipt_sha256,
        canonical_roster_receipt=args.canonical_roster_receipt,
        canonical_roster_receipt_sha256=args.canonical_roster_receipt_sha256,
        environment=os.environ,
    )
    if ROOT != preflight.repo_root:
        raise R241ElmoRemoteWorkerError(
            "the :8767 launcher must itself execute from the verified source snapshot; "
            f"launcher_root={ROOT} snapshot_root={preflight.repo_root}"
        )
    receipts = build_r241_elmo_preflight_receipts(
        preflight, receipt_dir=receipt_dir
    )
    paths = write_r241_elmo_preflight_receipts(
        receipts, receipt_dir=receipt_dir
    )
    manifest = validate_r241_elmo_preflight_manifest(paths["manifest"])
    return preflight, manifest


def _validated_worker_args(values: Sequence[str]) -> list[str]:
    """Keep the generic worker from changing the pinned endpoint or policy."""

    forwarded = list(values)
    if forwarded[:1] == ["--"]:
        forwarded = forwarded[1:]
    forbidden_exact = {
        "--host",
        "--port",
        "--checkpoint",
        "--cg-lib-path",
        "--smoke",
    }
    forbidden_prefixes = (
        "--host=",
        "--port=",
        "--checkpoint=",
        "--cg-lib-path=",
        "--mcts",
        "--rtp",
        "--search",
        "--recursive-turn-planner",
    )
    invalid = [
        token
        for token in forwarded
        if token in forbidden_exact or token.startswith(forbidden_prefixes)
    ]
    if invalid:
        raise R241ElmoRemoteWorkerError(
            "r241 :8767 serve refuses endpoint/checkpoint/MCTS/RTP/search overrides: "
            + ", ".join(invalid)
        )
    return forwarded


def _serve(preflight: R241ElmoPreflight, worker_args: Sequence[str]) -> int:
    """Delegate only after preflight; never invoke this from an import/test."""

    forwarded = _validated_worker_args(worker_args)
    # This import is intentionally delayed until all local receipt validation
    # succeeds.  The generic runner supplies TCP/worker-pool mechanics; its
    # optional restricted-kind/capability mode is set in the sealed mapping.
    from scripts import run_remote_worker

    command = [
        "--host",
        "0.0.0.0",
        "--port",
        str(ELMO_R241_ENDPOINT_PORT),
        "--checkpoint",
        str(preflight.checkpoint),
        "--cg-lib-path",
        str(preflight.cg_lib_path),
        *forwarded,
    ]
    with _sealed_process_environment(preflight.environment):
        return int(run_remote_worker.main(command))


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        preflight, manifest = _run_preflight(args)
        if args.print_json:
            print(
                json.dumps(
                    {
                        "path": manifest["path"],
                        "sha256": manifest["sha256"],
                        "manifest": manifest["payload"],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
        if args.command == "serve":
            validate_r241_elmo_activation_overlay(
                overlay_path=args.activation_overlay,
                overlay_sha256=args.activation_overlay_sha256,
                preflight=preflight,
                manifest=manifest,
            )
            return _serve(preflight, args.worker_args)
        return 0
    except R241ElmoRemoteWorkerError as exc:
        print(f"[r241-elmo-r236-worker] ERROR: {exc}", file=sys.stderr)
        return 78


if __name__ == "__main__":
    raise SystemExit(main())
