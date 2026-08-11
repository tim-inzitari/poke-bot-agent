#!/usr/bin/env python3
"""Create one data-only r241 H10 Marnie direct-policy receipt.

The historical H10 package is a data source only.  This command validates its
model, deck, matchup tree, disabled search configuration, and historical
embedded libcg without importing its ``main.py`` or loading model weights.  It
also repeats the sealed official-r236 symbol-only preflight before publishing
one create-only receipt.  It never starts a worker, listener, service, battle,
search, trainer, or submission client.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot import baselines_runtime  # noqa: E402
from poke_bot import r241_direct_policy_runtime as runtime  # noqa: E402
from poke_bot import r241_marnie_direct_policy_adapter as adapter  # noqa: E402


class R241MarnieAdapterReceiptError(RuntimeError):
    """The create-only H10 data-only receipt cannot be safely published."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _regular_directory(path: Path, *, label: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise R241MarnieAdapterReceiptError(f"{label} must be a real directory: {raw}")
    return raw.resolve()


def _regular_file(path: Path, *, label: str) -> Path:
    raw = path.expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241MarnieAdapterReceiptError(
            f"{label} must be a regular non-symlink file: {raw}"
        )
    return raw.resolve()


def _load_stager(source_root: Path) -> Any:
    path = _regular_file(
        source_root / "scripts/stage_r241_official_libcg_direct_policy.py",
        label="sealed r241 official-libcg stager",
    )
    spec = importlib.util.spec_from_file_location("r241_official_libcg_stager", path)
    if spec is None or spec.loader is None:
        raise R241MarnieAdapterReceiptError("cannot load sealed r241 official-libcg stager")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sealed_environment(*, cg_root: Path, receipt: Path) -> dict[str, str]:
    """Return exactly the no-search environment consumed by the adapter."""

    return {
        runtime.R241_DIRECT_POLICY_ONLY_ENV: "1",
        runtime.R241_DIRECT_POLICY_RECEIPT_ENV: str(receipt),
        "CG_LIB_PATH": str(cg_root),
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
        "POKEBOT_SEARCH_MODE": "policy",
        "POKEBOT_SUBMISSION_SEARCH_DISABLE": "1",
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
    }


def _payload(
    *,
    source_root: Path,
    source_manifest: Path,
    stager_path: Path,
    export_preflight: Mapping[str, Any],
    cg_root: Path,
    package_root: Path,
) -> dict[str, Any]:
    deck_path = package_root / "deck.csv"
    deck = baselines_runtime.deck_pool.read_deck(deck_path)
    content_digest = baselines_runtime.baseline_content_digest(package_root)
    return {
        "schema": adapter.R241_MARNIE_ADAPTER_RECEIPT_SCHEMA,
        "revision": runtime.R241_REVISION,
        "status": "passed",
        "passed": True,
        "immutable": True,
        "write_once": True,
        "offline_only": True,
        "direct_policy_only": True,
        "action_selector": "direct_policy_only",
        "runtime": {
            "package_main_imported": False,
            "package_search_invoked": False,
            "embedded_cg_loaded": False,
            "matchup_adapter_runtime": True,
            "matchup_adapter_tree_loaded": True,
            "mcts_calls": 0,
            "rtp_calls": 0,
            "search_calls": 0,
        },
        "package": {
            "opponent_id": adapter.R241_H10_OPPONENT_ID,
            "root_path": str(package_root),
            "content_sha256": content_digest,
            "model": {
                "relative_path": "model.pt",
                "sha256": _sha256(package_root / "model.pt"),
                "size_bytes": (package_root / "model.pt").stat().st_size,
            },
            "deck": {
                "relative_path": "deck.csv",
                "sha256": _sha256(deck_path),
                "card_count": len(deck),
            },
            "matchup_tree": {
                "relative_path": "matchup_tree.json",
                "sha256": _sha256(package_root / "matchup_tree.json"),
                "size_bytes": (package_root / "matchup_tree.json").stat().st_size,
            },
        },
        "sealed_runtime": {
            "cg_lib_path": str(cg_root),
            "linux_x86_64_sha256": runtime.R241_OFFICIAL_LINUX_LIBCG_SHA256,
            "official_preflight_receipt": str(
                cg_root / runtime.R241_OFFICIAL_LIBCG_RECEIPT_FILENAME
            ),
            "official_preflight_receipt_sha256": _sha256(
                cg_root / runtime.R241_OFFICIAL_LIBCG_RECEIPT_FILENAME
            ),
        },
        "offline_preflight": {
            "source_snapshot_root": str(source_root),
            "source_snapshot_manifest": str(source_manifest),
            "source_snapshot_manifest_sha256": _sha256(source_manifest),
            "stager_path": str(stager_path),
            "stager_sha256": _sha256(stager_path),
            "official_exports_attested": list(
                export_preflight["native_export_attestation"]["attested_exports"]
            ),
            "native_function_calls": int(
                export_preflight["native_export_attestation"]["native_function_calls"]
            ),
            "search_calls_made": int(export_preflight["search_calls_made"]),
            "simulator_battles_started": int(
                export_preflight["simulator_battles_started"]
            ),
            "model_weights_loaded": False,
            "baseline_package_main_imported": False,
        },
    }


def _create_only_validate_and_publish(
    *, payload: Mapping[str, Any], output: Path, package_root: Path, cg_root: Path
) -> dict[str, Any]:
    raw_output = output.expanduser()
    if raw_output.exists() or raw_output.is_symlink():
        raise R241MarnieAdapterReceiptError(
            f"create-only H10 adapter receipt already exists: {raw_output}"
        )
    parent = _regular_directory(raw_output.parent, label="H10 adapter receipt parent")
    output = parent / raw_output.name
    if output.exists() or output.is_symlink():
        raise R241MarnieAdapterReceiptError(
            f"create-only H10 adapter receipt already exists: {output}"
        )
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=".r241-h10-adapter-", dir=parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        environment = _sealed_environment(cg_root=cg_root, receipt=temporary)
        spec = SimpleNamespace(
            id=adapter.R241_H10_OPPONENT_ID,
            dir_name=adapter.R241_H10_DIR_NAME,
            path=package_root,
        )
        model, deck, tree, validated_receipt = (
            adapter.validate_r241_marnie_direct_policy_adapter(
                spec, environment=environment
            )
        )
        if (
            model != (package_root / "model.pt").resolve()
            or len(deck) != 60
            or tree != (package_root / "matchup_tree.json").resolve()
            or validated_receipt != temporary.resolve()
        ):
            raise R241MarnieAdapterReceiptError(
                "H10 data-only adapter validator returned an unexpected binding"
            )
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, output)
        except FileExistsError as exc:
            raise R241MarnieAdapterReceiptError(
                f"create-only H10 adapter receipt already exists: {output}"
            ) from exc
        final_environment = _sealed_environment(cg_root=cg_root, receipt=output)
        adapter.validate_r241_marnie_direct_policy_adapter(spec, environment=final_environment)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(output),
        "sha256": _sha256(output),
        "mode": oct(output.stat().st_mode & 0o777),
    }


def generate_receipt(
    *,
    source_snapshot_root: Path,
    source_snapshot_manifest: Path,
    cg_lib_path: Path,
    package_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Run export-only and data-only validation, then create exactly one receipt."""

    if output.expanduser().name != runtime.R241_H10_ADAPTER_RECEIPT_BASENAME:
        raise R241MarnieAdapterReceiptError(
            "H10 adapter receipt output must use the predeclared successor path "
            f"{runtime.R241_H10_ADAPTER_RECEIPT_BASENAME}"
        )
    source_root = _regular_directory(source_snapshot_root, label="source snapshot root")
    if source_root != ROOT.resolve():
        raise R241MarnieAdapterReceiptError(
            "receipt producer must execute from the supplied immutable source snapshot"
        )
    manifest = _regular_file(source_snapshot_manifest, label="source snapshot manifest")
    cg_root = _regular_directory(cg_lib_path, label="sealed CG_LIB_PATH")
    package = _regular_directory(package_root, label="H10 baseline package root")
    stager = _load_stager(source_root)
    stager_path = source_root / "scripts/stage_r241_official_libcg_direct_policy.py"
    try:
        export_preflight = stager.preflight_staged_runtime(
            runtime_root=cg_root, environment={}
        )
    except Exception as exc:  # the source stager owns its detailed error
        raise R241MarnieAdapterReceiptError(
            f"official r236 export-only preflight failed: {type(exc).__name__}: {exc}"
        ) from exc
    attestation = dict(export_preflight.get("native_export_attestation") or {})
    if (
        export_preflight.get("status") != "passed"
        or export_preflight.get("passed") is not True
        or export_preflight.get("search_calls_made") != 0
        or export_preflight.get("simulator_battles_started") != 0
        or attestation.get("native_function_calls") != 0
        or tuple(attestation.get("attested_exports") or ())
        != runtime.R241_REQUIRED_NATIVE_EXPORTS
    ):
        raise R241MarnieAdapterReceiptError(
            "official r236 export-only preflight is not a passed no-search attestation"
        )
    payload = _payload(
        source_root=source_root,
        source_manifest=manifest,
        stager_path=stager_path,
        export_preflight=export_preflight,
        cg_root=cg_root,
        package_root=package,
    )
    published = _create_only_validate_and_publish(
        payload=payload, output=output, package_root=package, cg_root=cg_root
    )
    return {
        "status": "passed",
        "receipt": published,
        "official_r236": {
            "cg_lib_path": str(cg_root),
            "linux_x86_64_sha256": runtime.R241_OFFICIAL_LINUX_LIBCG_SHA256,
            "native_function_calls": 0,
            "search_calls_made": 0,
            "simulator_battles_started": 0,
        },
        "h10": {
            "package_root": str(package),
            "content_sha256": adapter.R241_H10_CONTENT_SHA256,
            "model_sha256": adapter.R241_H10_MODEL_SHA256,
            "matchup_tree_sha256": adapter.R241_H10_MATCHUP_TREE_SHA256,
            "package_main_imported": False,
            "model_weights_loaded": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot-root", type=Path, default=ROOT)
    parser.add_argument("--source-snapshot-manifest", type=Path)
    parser.add_argument("--cg-lib-path", type=Path, required=True)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    source_root = args.source_snapshot_root.expanduser().resolve()
    manifest = (
        args.source_snapshot_manifest
        if args.source_snapshot_manifest is not None
        else source_root / "r241-source-snapshot-manifest.json"
    )
    try:
        result = generate_receipt(
            source_snapshot_root=source_root,
            source_snapshot_manifest=manifest,
            cg_lib_path=args.cg_lib_path,
            package_root=args.package_root,
            output=args.output,
        )
    except R241MarnieAdapterReceiptError as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
