"""Checksum-bound, direct-policy-only loader for r241's H10 Marnie model.

This intentionally treats the frozen specialist package as a *data package*.
It reads its deck and model bytes after verifying their immutable identity; it
never imports the package's ``main.py``, ``cg`` package, ``poke_bot`` tree, or
search configuration.  The policy is reconstructed through this repository's
r241 process under the sealed official ``CG_LIB_PATH`` instead.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, TYPE_CHECKING

from . import deck_pool
from .r241_direct_policy_runtime import (
    R241_DIRECT_POLICY_ONLY_ENV,
    R241_DIRECT_POLICY_RECEIPT_ENV,
    R241_H10_CONTENT_SHA256,
    R241_H10_DIR_NAME,
    R241_H10_MODEL_SHA256,
    R241_H10_MODEL_SIZE_BYTES,
    R241_H10_OPPONENT_ID,
    R241_OFFICIAL_LINUX_LIBCG_SHA256,
    R241_OLD_EMBEDDED_LIBCG_SHA256,
    R241_REVISION,
    R241DirectPolicyRuntimeError,
    assert_direct_policy_environment,
    normalized_path,
    read_json_object,
    regular_child,
    sha256_file,
    validate_sealed_official_libcg,
)

if TYPE_CHECKING:
    from .baselines_runtime import AgentFn, BaselineSpec


R241_MARNIE_ADAPTER_RECEIPT_SCHEMA = (
    "poke_bot.alakazam_new_list_direct_policy_r241_marnie_adapter/v1"
)
R241_H10_MATCHUP_TREE_SHA256 = (
    "sha256:da223c4903dd37511e5cb7656fe405bc0baac085be4f131faef136b7056c4588"
)
R241_H10_MATCHUP_TREE_SIZE_BYTES = 2_509_756


class R241MarnieAdapterError(R241DirectPolicyRuntimeError):
    """The H10 model cannot be safely admitted to the r241 process."""


def _as_exact_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise R241MarnieAdapterError(f"{label} must be an exact integer")
    return value


def _package_root(spec: Any) -> Path:
    raw = Path(spec.path).expanduser()
    if raw.is_symlink() or not raw.is_dir():
        raise R241MarnieAdapterError(
            f"r241 H10 package must be a real directory: {raw}"
        )
    root = raw.resolve()
    if (
        str(getattr(spec, "id", "")) != R241_H10_OPPONENT_ID
        or str(getattr(spec, "dir_name", "")) != R241_H10_DIR_NAME
        or root.name != R241_H10_DIR_NAME
    ):
        raise R241MarnieAdapterError("r241 adapter received a non-canonical H10 spec")
    return root


def _validate_direct_receipt(
    *,
    receipt_path: Path | str,
    package_root: Path,
    environment: Mapping[str, str],
) -> tuple[Path, list[int], Path, Path]:
    """Validate receipt, immutable model/deck data, and sealed native root."""

    receipt_file, receipt = read_json_object(
        receipt_path, label="r241 Marnie direct-policy adapter receipt"
    )
    runtime = dict(receipt.get("runtime") or {})
    if (
        receipt.get("schema") != R241_MARNIE_ADAPTER_RECEIPT_SCHEMA
        or _as_exact_int(receipt.get("revision"), label="adapter receipt revision")
        != R241_REVISION
        or receipt.get("status") != "passed"
        or receipt.get("passed") is not True
        or receipt.get("direct_policy_only") is not True
        or receipt.get("action_selector") != "direct_policy_only"
        or runtime.get("package_main_imported") is not False
        or runtime.get("package_search_invoked") is not False
        or runtime.get("embedded_cg_loaded") is not False
        or runtime.get("matchup_adapter_runtime") is not True
        or runtime.get("matchup_adapter_tree_loaded") is not True
        or _as_exact_int(runtime.get("mcts_calls"), label="adapter mcts_calls") != 0
        or _as_exact_int(runtime.get("rtp_calls"), label="adapter rtp_calls") != 0
        or _as_exact_int(runtime.get("search_calls"), label="adapter search_calls") != 0
    ):
        raise R241MarnieAdapterError(
            f"adapter receipt is not an r241 direct-policy no-search receipt: {receipt_file}"
        )
    package = dict(receipt.get("package") or {})
    if (
        package.get("opponent_id") != R241_H10_OPPONENT_ID
        or normalized_path(str(package.get("root_path") or ""), label="receipt package root")
        != package_root
        or package.get("content_sha256") != R241_H10_CONTENT_SHA256
    ):
        raise R241MarnieAdapterError("adapter receipt package identity is invalid")

    model = dict(package.get("model") or {})
    deck = dict(package.get("deck") or {})
    matchup_tree = dict(package.get("matchup_tree") or {})
    if (
        model.get("relative_path") != "model.pt"
        or model.get("sha256") != R241_H10_MODEL_SHA256
        or _as_exact_int(model.get("size_bytes"), label="H10 model size")
        != R241_H10_MODEL_SIZE_BYTES
        or deck.get("relative_path") != "deck.csv"
        or not str(deck.get("sha256") or "").startswith("sha256:")
        or len(str(deck.get("sha256") or "")) != 71
        or matchup_tree.get("relative_path") != "matchup_tree.json"
        or matchup_tree.get("sha256") != R241_H10_MATCHUP_TREE_SHA256
        or _as_exact_int(
            matchup_tree.get("size_bytes"), label="H10 matchup tree size"
        )
        != R241_H10_MATCHUP_TREE_SIZE_BYTES
    ):
        raise R241MarnieAdapterError("adapter receipt model/deck member declaration is invalid")
    model_path = regular_child(package_root, "model.pt", label="H10 model")
    deck_path = regular_child(package_root, "deck.csv", label="H10 deck")
    matchup_tree_path = regular_child(
        package_root, "matchup_tree.json", label="H10 matchup tree"
    )
    search_config_path = regular_child(
        package_root, "search_config.json", label="H10 search configuration"
    )
    if (
        model_path.stat().st_size != R241_H10_MODEL_SIZE_BYTES
        or sha256_file(model_path) != R241_H10_MODEL_SHA256
        or sha256_file(deck_path) != deck.get("sha256")
        or matchup_tree_path.stat().st_size != R241_H10_MATCHUP_TREE_SIZE_BYTES
        or sha256_file(matchup_tree_path) != R241_H10_MATCHUP_TREE_SHA256
    ):
        raise R241MarnieAdapterError("H10 model/deck file identity mismatches its receipt")
    declared_cards = _as_exact_int(deck.get("card_count"), label="H10 deck card_count")
    loaded_deck = deck_pool.read_deck(deck_path)
    if declared_cards != 60 or len(loaded_deck) != 60:
        raise R241MarnieAdapterError("H10 adapter deck must contain exactly 60 cards")
    # This package data file is not executed.  Its disabled state is asserted
    # anyway so a receipt cannot treat an altered search-enabled package as a
    # harmless model/deck payload.
    _, search_config = read_json_object(
        search_config_path, label="H10 search configuration"
    )
    if search_config.get("enabled") is not False:
        raise R241MarnieAdapterError("H10 package search configuration is not disabled")

    # Hashing the complete package binds the legacy package contents without
    # executing them.  This is intentionally after literal member checks so a
    # receipt cannot point at an arbitrary same-sized model file.
    from .baselines_runtime import baseline_content_digest

    if baseline_content_digest(package_root) != R241_H10_CONTENT_SHA256:
        raise R241MarnieAdapterError("H10 package content digest is not f7-pinned")
    embedded = regular_child(package_root, "cg/libcg.so", label="H10 embedded libcg")
    if sha256_file(embedded) != R241_OLD_EMBEDDED_LIBCG_SHA256:
        raise R241MarnieAdapterError("H10 package does not contain its known old embedded libcg")

    sealed = dict(receipt.get("sealed_runtime") or {})
    cg_root = assert_direct_policy_environment(environment)
    if (
        normalized_path(str(sealed.get("cg_lib_path") or ""), label="adapter sealed CG_LIB_PATH")
        != cg_root
        or sealed.get("linux_x86_64_sha256") != R241_OFFICIAL_LINUX_LIBCG_SHA256
    ):
        raise R241MarnieAdapterError("adapter receipt does not bind the sealed official libcg")
    # The old package root must never be the runtime root or an ancestor of it.
    if cg_root == package_root or package_root in cg_root.parents or cg_root in package_root.parents:
        raise R241MarnieAdapterError("r241 refuses a CG_LIB_PATH mixed with the H10 package")
    validate_sealed_official_libcg(cg_root, environment=environment)
    return model_path, loaded_deck, matchup_tree_path, receipt_file


def _build_direct_policy(
    model_path: Path, deck: list[int], matchup_tree_path: Path
) -> Any:
    """Reconstruct direct H10 policy through r241 code, never package code."""

    import torch

    from .agent import PolicyAgent
    from .train import load_model_from_checkpoint

    device = torch.device("cpu")
    model = load_model_from_checkpoint(model_path, device=device)
    return PolicyAgent(
        model=model,
        deck=list(deck),
        use_mcts=False,
        use_recursive_turn_planner=False,
        oracle_mode=False,
        belief_mcts=False,
        max_sims=0,
        move_time_s=0.0,
        collect_targets=False,
        sample_actions=False,
        leaf_backend=None,
        strict_runtime=True,
        matchup_adapter_shadow=False,
        # The H10 package's direct path has a trained Matchup Adapter bank.
        # Bind its own immutable data-only tree explicitly; never import the
        # package main.py merely to let it set this process-global state.
        matchup_adapter_runtime=True,
        matchup_adapter_tree_path=str(matchup_tree_path),
        device=device,
    )


def _direct_callable(policy: Any) -> Callable[[dict], list[int]]:
    """Reset the rebuilt local policy at deck selection, like a baseline agent."""

    def agent(observation: dict) -> list[int]:
        if observation is None or observation.get("select") is None:
            reset = getattr(policy, "reset_game", None)
            if callable(reset):
                reset()
        return [int(value) for value in policy(observation)]

    return agent


def validate_r241_marnie_direct_policy_adapter(
    spec: Any,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[Path, list[int], Path, Path]:
    """Validate the adapter binding without loading model weights or package code."""

    env = dict(os.environ if environment is None else environment)
    package_root = _package_root(spec)
    receipt_raw = str(env.get(R241_DIRECT_POLICY_RECEIPT_ENV) or "").strip()
    if not receipt_raw:
        raise R241MarnieAdapterError(
            f"{R241_DIRECT_POLICY_RECEIPT_ENV} is required for the H10 adapter"
        )
    return _validate_direct_receipt(
        receipt_path=receipt_raw,
        package_root=package_root,
        environment=env,
    )


def maybe_load_r241_direct_policy_agent(
    spec: Any,
    *,
    environment: Mapping[str, str] | None = None,
) -> tuple[Callable[[dict], list[int]], list[int]] | None:
    """Load only the explicitly receipted H10 direct policy.

    Returning ``None`` leaves all ordinary baseline behavior unchanged.  The
    direct-only environment binds the *H10 action selector*, not the entire
    established public roster: r241 deliberately preserves ordinary diverse
    public and research-control packages.
    """

    env = dict(os.environ if environment is None else environment)
    receipt_raw = str(env.get(R241_DIRECT_POLICY_RECEIPT_ENV) or "").strip()
    strict = env.get(R241_DIRECT_POLICY_ONLY_ENV) == "1"
    if not receipt_raw:
        if strict and str(getattr(spec, "id", "")) == R241_H10_OPPONENT_ID:
            raise R241MarnieAdapterError(
                f"{R241_DIRECT_POLICY_RECEIPT_ENV} is required for direct H10 mode"
            )
        return None
    if str(getattr(spec, "id", "")) != R241_H10_OPPONENT_ID:
        return None
    model_path, deck, matchup_tree_path, _receipt = validate_r241_marnie_direct_policy_adapter(
        spec, environment=env
    )
    return _direct_callable(
        _build_direct_policy(model_path, deck, matchup_tree_path)
    ), deck
