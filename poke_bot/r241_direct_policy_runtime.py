"""Fail-closed identity checks for the isolated r241 direct-policy runtime.

The final-format H10 Marnie package is a useful immutable model artifact, but
it also contains the historical Python entry point and an old embedded
``libcg``.  r241 is deliberately not permitted to execute either.  This
module contains the small, dependency-light checks shared by the r241 adapter
and its launch wrapper before they load any policy code.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


R241_REVISION = 241
R241_H10_OPPONENT_ID = "specialist-marnie-final-format-h10-f20efb20f5c3"
R241_H10_DIR_NAME = "marnie-final-format-h10-f20efb20f5c3"
R241_H10_MODEL_SHA256 = (
    "sha256:f20efb20f5c30820c7e23004e529d326ec87f91b026c1fe3bbb431f9c8b44381"
)
R241_H10_MODEL_SIZE_BYTES = 132_425_453
R241_H10_CONTENT_SHA256 = (
    "sha256:f7c25cfd0bba674ceb4c2156a6e2fef87a3ff9effc74ed41b33fbb17fd627787"
)
# The r251 through r251-v7 H10 receipts and unsuffixed/v1/v2/v3/v4/v5 peak paths
# are immutable evidence for inactive source snapshots.  Every active
# r241 consumer must therefore name these predeclared successors explicitly
# rather than accepting a generic or prior receipt path.
R241_H10_ADAPTER_RECEIPT_BASENAME = (
    "marnie-h10-direct-policy-adapter-r251-v8.json"
)
R241_PEAK_R195_PRESERVATION_RECEIPT_BASENAME = (
    "peak-r195-preservation-v6.json"
)
R241_OLD_EMBEDDED_LIBCG_SHA256 = (
    "sha256:ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
)

R241_OFFICIAL_LIBCG_RECEIPT_SCHEMA = (
    "poke_bot.r241_official_libcg_direct_policy_preflight/v1"
)
R241_OFFICIAL_LIBCG_RECEIPT_FILENAME = (
    "r241_official_libcg_direct_policy_preflight.json"
)
R241_OFFICIAL_LINUX_LIBCG_SHA256 = (
    "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7"
)
R241_OFFICIAL_LINUX_LIBCG_SIZE_BYTES = 1_342_400
R241_REQUIRED_NATIVE_EXPORTS = (
    "AgentStart",
    "BattleStart",
    "SearchBegin",
    "SearchStep",
    "SearchRelease",
    "SearchEnd",
)
R241_CANONICAL_NATIVE_MEMBERS: dict[str, dict[str, Any]] = {
    "linux_x86_64": {
        "package_relative_path": "cg/libcg.so",
        "sha256": R241_OFFICIAL_LINUX_LIBCG_SHA256,
        "size_bytes": R241_OFFICIAL_LINUX_LIBCG_SIZE_BYTES,
        "format": "ELF 64-bit LSB shared object x86-64",
    },
    "linux_aarch64": {
        "package_relative_path": "cg/libcg-arm64.so",
        "sha256": (
            "sha256:1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2"
        ),
        "size_bytes": 1_296_464,
        "format": "ELF 64-bit LSB shared object ARM aarch64",
    },
    "macos_arm64": {
        "package_relative_path": "cg/libcg.dylib",
        "sha256": (
            "sha256:7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30"
        ),
        "size_bytes": 1_245_544,
        "format": "Mach-O 64-bit dynamically linked shared library arm64",
    },
    "windows_x86_64": {
        "package_relative_path": "cg/cg.dll",
        "sha256": (
            "sha256:eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771"
        ),
        "size_bytes": 1_525_248,
        "format": "PE32+ x86-64 DLL",
    },
}

FORBIDDEN_LIBCG_OVERRIDE_KEYS = (
    "POKEBOT_LIBCG_PATH",
    "POKEBOT_BATCH_LIBCG",
)
R241_DIRECT_POLICY_RECEIPT_ENV = "POKEBOT_R241_DIRECT_POLICY_ADAPTER_RECEIPT"
R241_DIRECT_POLICY_ONLY_ENV = "POKEBOT_R241_DIRECT_POLICY_ONLY"


class R241DirectPolicyRuntimeError(RuntimeError):
    """An r241 direct-policy identity or safety invariant was not satisfied."""


def _exact_int(value: object, *, label: str) -> int:
    """Return a JSON integer without treating a legitimate zero as absent."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise R241DirectPolicyRuntimeError(f"{label} must be an exact integer")
    return value


def sha256_file(path: Path | str) -> str:
    """Hash a regular file without ever importing/executing its contents."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _resolved_directory(path: Path | str, *, label: str) -> Path:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise R241DirectPolicyRuntimeError(
            f"{label} must be a real directory: {raw}"
        )
    return raw.resolve()


def regular_child(root: Path | str, relative: str, *, label: str) -> Path:
    """Resolve one literal child and reject symlink/path-escape substitutions."""

    directory = _resolved_directory(root, label=f"{label} root")
    member = Path(relative)
    if (
        not relative
        or member.is_absolute()
        or ".." in member.parts
        or len(member.parts) == 0
    ):
        raise R241DirectPolicyRuntimeError(f"unsafe {label} member: {relative!r}")
    raw = directory.joinpath(*member.parts)
    if raw.is_symlink() or not raw.is_file():
        raise R241DirectPolicyRuntimeError(
            f"{label} must be a regular non-symlink file: {raw}"
        )
    resolved = raw.resolve()
    if directory not in resolved.parents:
        raise R241DirectPolicyRuntimeError(f"{label} escapes its root: {raw}")
    return resolved


def read_json_object(path: Path | str, *, label: str) -> tuple[Path, dict[str, Any]]:
    raw = Path(path).expanduser()
    if raw.is_symlink() or not raw.is_file():
        raise R241DirectPolicyRuntimeError(
            f"{label} must be a regular non-symlink file: {raw}"
        )
    resolved = raw.resolve()
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise R241DirectPolicyRuntimeError(f"{label} is unreadable JSON: {resolved}") from exc
    if not isinstance(payload, dict):
        raise R241DirectPolicyRuntimeError(f"{label} must contain a JSON object")
    return resolved, payload


def normalized_path(value: Path | str, *, label: str) -> Path:
    raw = Path(value).expanduser()
    if not str(value).strip():
        raise R241DirectPolicyRuntimeError(f"{label} is missing")
    return raw.resolve()


def assert_direct_policy_environment(
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Reject every search/planning selector before loading an r241 policy."""

    env = dict(os.environ if environment is None else environment)
    present_overrides = [key for key in FORBIDDEN_LIBCG_OVERRIDE_KEYS if key in env]
    if present_overrides:
        raise R241DirectPolicyRuntimeError(
            "r241 forbids private/batch libcg overrides: "
            + ", ".join(present_overrides)
        )
    if env.get(R241_DIRECT_POLICY_ONLY_ENV) != "1":
        raise R241DirectPolicyRuntimeError(
            f"{R241_DIRECT_POLICY_ONLY_ENV}=1 is required"
        )
    required = {
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
        "POKEBOT_SEARCH_MODE": "policy",
        "POKEBOT_SUBMISSION_SEARCH_DISABLE": "1",
    }
    mismatched = [
        f"{key}={env.get(key)!r}" for key, expected in required.items() if env.get(key) != expected
    ]
    if mismatched:
        raise R241DirectPolicyRuntimeError(
            "r241 direct-policy environment is incomplete: " + ", ".join(mismatched)
        )
    forbidden_prefixes = (
        "POKEBOT_MCTS_",
        "POKEBOT_RTP_",
        "POKEBOT_BELIEF_",
        "POKEBOT_POKE_RLM_",
        "POKEBOT_SLOWKING_DISTILL_",
        "POKEBOT_GUIDE2VEC_",
    )
    # Matchup Adapters are not a search planner.  r241 preserves the trained
    # r195 learner bank and H10's own checksum-bound tree; the adapter passes
    # the latter explicitly as a constructor argument rather than inheriting a
    # package-owned startup environment.  Oracle deck access remains barred.
    forbidden_exact = {"POKEBOT_ALLOW_ORACLE_DECK"}
    forbidden_search = [
        key
        for key in env
        if key in forbidden_exact
        or key.startswith(forbidden_prefixes)
        or (
            key.startswith("POKEBOT_SEARCH_")
            and key not in {"POKEBOT_SEARCH_MODE"}
        )
    ]
    if forbidden_search:
        raise R241DirectPolicyRuntimeError(
            "r241 direct policy rejects search/planning environment: "
            + ", ".join(sorted(forbidden_search))
        )
    raw_cg = str(env.get("CG_LIB_PATH") or "").strip()
    if not raw_cg:
        raise R241DirectPolicyRuntimeError("CG_LIB_PATH is required for r241")
    return _resolved_directory(raw_cg, label="CG_LIB_PATH")


def validate_sealed_official_libcg(
    cg_lib_path: Path | str | None = None,
    *,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Verify the staged r241 official ``cg`` root and its no-call receipt.

    The receipt is not merely provenance: it binds this exact loader root,
    all platform sibling identities, and a symbol-resolution-only attestation.
    The physical Linux file is hashed again here, so a copied receipt cannot
    authorize a later mixed or old library.
    """

    env = dict(os.environ if environment is None else environment)
    env_root = assert_direct_policy_environment(env)
    root = env_root if cg_lib_path is None else _resolved_directory(
        cg_lib_path, label="sealed CG_LIB_PATH"
    )
    if root != env_root:
        raise R241DirectPolicyRuntimeError(
            "sealed CG_LIB_PATH disagrees with the process CG_LIB_PATH"
        )
    receipt_path, receipt = read_json_object(
        root / R241_OFFICIAL_LIBCG_RECEIPT_FILENAME,
        label="r241 official libcg preflight receipt",
    )
    if (
        receipt.get("schema") != R241_OFFICIAL_LIBCG_RECEIPT_SCHEMA
        or int(receipt.get("revision") or 0) != R241_REVISION
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("direct_policy_only") is not True
        or _exact_int(
            receipt.get("search_calls_made"), label="official libcg search_calls_made"
        )
        != 0
        or _exact_int(
            receipt.get("simulator_battles_started"),
            label="official libcg simulator_battles_started",
        )
        != 0
    ):
        raise R241DirectPolicyRuntimeError(
            f"official libcg preflight is not an r241 passed no-search receipt: {receipt_path}"
        )
    if normalized_path(str(receipt.get("cg_lib_path") or ""), label="receipt cg_lib_path") != root:
        raise R241DirectPolicyRuntimeError("official libcg receipt binds a different CG_LIB_PATH")
    receipt_environment = dict(receipt.get("environment") or {})
    if (
        receipt_environment.get("forbidden_override_keys_absent") is not True
        or set(receipt_environment.get("forbidden_override_keys") or ())
        != set(FORBIDDEN_LIBCG_OVERRIDE_KEYS)
        or normalized_path(
            str(receipt_environment.get("CG_LIB_PATH") or ""),
            label="receipt environment CG_LIB_PATH",
        )
        != root
    ):
        raise R241DirectPolicyRuntimeError("official libcg receipt environment binding is invalid")

    native_members = dict(receipt.get("canonical_native_members") or {})
    if set(native_members) != set(R241_CANONICAL_NATIVE_MEMBERS):
        raise R241DirectPolicyRuntimeError("official libcg receipt has an incomplete native-member set")
    for platform_name, expected in R241_CANONICAL_NATIVE_MEMBERS.items():
        actual = dict(native_members.get(platform_name) or {})
        # The staging receipt's public schema deliberately calls this member
        # key ``path``.  ``package_relative_path`` is the internal typed
        # contract name used to locate the same member before staging.
        if (
            actual.get("path") != expected["package_relative_path"]
            or actual.get("sha256") != expected["sha256"]
            or _exact_int(
                actual.get("size_bytes"),
                label=f"official libcg {platform_name} member size",
            )
            != expected["size_bytes"]
            or actual.get("format") != expected["format"]
        ):
            raise R241DirectPolicyRuntimeError(
                f"official libcg receipt native identity drifted for {platform_name}"
            )

    library = regular_child(root, "cg/libcg.so", label="official Linux libcg")
    loaded = dict(receipt.get("loaded_library") or {})
    if (
        loaded.get("target_platform") != "linux_x86_64"
        or normalized_path(str(loaded.get("path") or ""), label="loaded library path")
        != library
        or loaded.get("sha256") != R241_OFFICIAL_LINUX_LIBCG_SHA256
        or int(loaded.get("size_bytes") or -1) != R241_OFFICIAL_LINUX_LIBCG_SIZE_BYTES
        or library.stat().st_size != R241_OFFICIAL_LINUX_LIBCG_SIZE_BYTES
        or sha256_file(library) != R241_OFFICIAL_LINUX_LIBCG_SHA256
    ):
        raise R241DirectPolicyRuntimeError("official Linux libcg identity is not r236 D162")
    # The Python wrapper is needed by CABT, but the adapter never imports a
    # package-owned wrapper.  Confirm the staged official root has one.
    regular_child(root, "cg/api.py", label="official cg wrapper")

    attestation = dict(receipt.get("native_export_attestation") or {})
    if (
        attestation.get("method") != "ctypes_symbol_resolution_only"
        or _exact_int(
            attestation.get("native_function_calls"),
            label="official libcg native_function_calls",
        )
        != 0
        or tuple(attestation.get("required_exports") or ())
        != R241_REQUIRED_NATIVE_EXPORTS
        or tuple(attestation.get("attested_exports") or ())
        != R241_REQUIRED_NATIVE_EXPORTS
    ):
        raise R241DirectPolicyRuntimeError("official libcg native export attestation is invalid")
    return root
