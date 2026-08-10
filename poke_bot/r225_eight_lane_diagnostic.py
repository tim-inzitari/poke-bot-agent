"""Persistent r225 shared-tree gameplay viability wrapper.

At every genuinely branching real-game decision this module opens eight
isolated stock-search arenas, keeps their SearchIds alive across depth waves,
and returns the backed action from one master-owned shared tree.  The frozen
r195 direct policy is only the contractually narrow clean-deadline fallback
when all eight sessions opened and cleaned without producing a backup.

The eight internal arenas are simulator contexts, not competition agents and
not an r224 root-parallel forest.  They all use the same loaded package-local
stock ``libcg`` DSO, frozen r195 model, and Matchup Adapter path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import threading
import time
from typing import Any, Mapping, Sequence


# This is the exact schema of the frozen owner contract.  Keep the runtime
# receipt under the same identity so a staged package cannot accidentally bind
# to an earlier r224/root-parallel diagnostic contract.
R225_SCHEMA = "poke_bot.alakazam_r222_shared_tree_eight_lane_kaggle_diagnostic_r225/v1"
R225_EXPECTED_CONFIG = "r225_eight_lane_diagnostic_config.json"
R225_EXPECTED_LABEL = "DONT USE FOR REVIEW — 8-LANE SHARED-TREE VIABILITY"
R225_CONTRACT_SHA256 = "sha256:7db1b6770bf71623cf0ed48ddeeaa503f5921cbfd7f39c6b8a9322b138a803f4"
R222_CONTRACT_SHA256 = "sha256:8b5a19e8746b8e5f667683ad6437a2f3506aa0fbdcca8495ec2b8bbd1eebeb7e"
R222_PERSISTENT_CORE_SHA256 = "sha256:7526aadc10aa41789cf1956f98eca80895b7bb8451d357c4c32b0dfb373ee148"
R195_MODEL_SHA256 = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
R195_MATCHUP_TREE_SHA256 = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
STOCK_LIBCG_SHA256 = "sha256:ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c"
STOCK_LIBCG_BYTES = 1_342_400
LANE_COUNT = 8
R225_DECISION_PREFIX = "R225_PERSISTENT_EVERY_BRANCH_DECISION"
R225_SUCCESS_PREFIX = "R225_PERSISTENT_EVERY_BRANCH_FULL_GAME_VALIDATION"


class R225DiagnosticError(RuntimeError):
    """A branching r225 decision lacks a structurally valid result."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _json_line(prefix: str, payload: Mapping[str, Any]) -> None:
    print(prefix + " " + json.dumps(payload, sort_keys=True, default=str), flush=True)


def _raw_observation(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        raw = asdict(value)
        if isinstance(raw, dict):
            return raw
    raise R225DiagnosticError(
        f"stock search returned an unsupported observation {type(value).__name__}"
    )


def _rss_bytes() -> int:
    # Linux reports KiB; macOS reports bytes.  Kaggle is Linux, but keep the
    # local receipt honest when package smoke runs elsewhere.
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value * 1024 if os.uname().sysname != "Darwin" else value


def _resource_probe() -> dict[str, Any]:
    try:
        physical_memory = int(os.sysconf("SC_PAGE_SIZE")) * int(
            os.sysconf("SC_PHYS_PAGES")
        )
    except (AttributeError, OSError, ValueError):
        physical_memory = None
    result: dict[str, Any] = {
        "cpu_count": os.cpu_count(),
        "physical_memory_bytes": physical_memory,
        "peak_rss_bytes": _rss_bytes(),
        "cuda_available": False,
        "gpu_name": None,
        "gpu_total_memory_bytes": None,
        "gpu_peak_allocated_bytes": None,
        "gpu_peak_reserved_bytes": None,
        "expected_h100_80gb_16vcpu_256gib_match": False,
    }
    try:
        import torch

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            name = str(properties.name)
            total = int(properties.total_memory)
            result.update(
                {
                    "cuda_available": True,
                    "cuda_device": int(device),
                    "gpu_name": name,
                    "gpu_total_memory_bytes": total,
                    "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
            )
            # 80 GB H100 allocations are normally roughly 80 GiB.  Do not
            # pretend a smaller/larger device is the requested environment.
            result["expected_h100_80gb_16vcpu_256gib_match"] = (
                "h100" in name.lower()
                and total >= 75 * 1024**3
                and int(os.cpu_count() or 0) >= 16
                and isinstance(physical_memory, int)
                and physical_memory >= 256 * 1024**3
            )
    except Exception as exc:  # resource telemetry itself cannot change action
        result["resource_probe_error"] = f"{type(exc).__name__}: {exc}"
    return result


def _is_explicit_chance(raw: Mapping[str, Any]) -> bool:
    """Recognize only the stock transport's explicit manual-coin context."""

    selection = raw.get("select")
    if not isinstance(selection, Mapping):
        return False
    # r222's stock ABI seam has a narrow attested manual coin prompt: 46.
    # Treating other public-looking contexts as random would be guessing game
    # rules, which this diagnostic is not allowed to do.
    return selection.get("context") == 46


def _get_policy_route(policy: Any) -> int:
    getter = getattr(policy, "_matchup_model_route", None)
    if not callable(getter):
        raise R225DiagnosticError("frozen r195 policy lacks its Matchup Adapter route getter")
    route = getter()
    if type(route) is not int:
        raise R225DiagnosticError("frozen r195 Matchup Adapter route is not an integer")
    return int(route)


def _check_config(stage: Path) -> dict[str, Any]:
    config_path = stage / R225_EXPECTED_CONFIG
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise R225DiagnosticError("r225 packaged diagnostic config is unreadable") from exc
    if not isinstance(payload, dict) or payload.get("schema") != R225_SCHEMA:
        raise R225DiagnosticError("r225 packaged diagnostic config schema changed")
    if (
        payload.get("owner_contract_sha256") != R225_CONTRACT_SHA256
        or payload.get("r222_contract_sha256") != R222_CONTRACT_SHA256
        or payload.get("persistent_core_sha256") != R222_PERSISTENT_CORE_SHA256
        or payload.get("submission_label_required") != R225_EXPECTED_LABEL
    ):
        raise R225DiagnosticError("r225 package contract/label binding changed")
    direct = payload.get("direct_policy")
    if not isinstance(direct, dict) or (
        direct.get("r195_model_sha256") != R195_MODEL_SHA256
        or direct.get("r195_matchup_tree_sha256") != R195_MATCHUP_TREE_SHA256
        or direct.get("rtp") != "disabled"
        or direct.get("turn_order_preference") != "first_if_allowed"
    ):
        raise R225DiagnosticError("r225 packaged direct-policy binding changed")
    shared = payload.get("shared_tree")
    if not isinstance(shared, dict) or (
        shared.get("lane_count") != LANE_COUNT
        or shared.get("one_shared_tree") is not True
        or shared.get("manual_coin_required") is not True
        or shared.get("unsafe_public_lookalike_merge_allowed") is not False
        or int(shared.get("max_depth_waves", 0)) < 2
        or shared.get("search_begin_once_per_lane_per_branching_decision") is not True
        or shared.get("retain_search_ids_across_depth_waves") is not True
        or shared.get("require_full_native_step_overlap") is not False
    ):
        raise R225DiagnosticError("r225 shared-tree runtime config changed")
    if not (stage / "contracts/r225-typed-contract.json").is_file() or not (
        stage / "contracts/r222-typed-contract.json"
    ).is_file():
        raise R225DiagnosticError("r225 package lacks embedded typed contracts")
    if _sha256_file(stage / "contracts/r225-typed-contract.json") != R225_CONTRACT_SHA256:
        raise R225DiagnosticError("r225 embedded typed contract digest changed")
    if _sha256_file(stage / "contracts/r222-typed-contract.json") != R222_CONTRACT_SHA256:
        raise R225DiagnosticError("r222 embedded typed contract digest changed")
    return payload


def _leaf_packet(
    *,
    raw: dict[str, Any],
    root_seat: int,
    own_deck: Sequence[int],
    opponent_guess: Sequence[int],
    route: int,
    combos: Sequence[Sequence[int]] | None = None,
) -> Any:
    from poke_bot.batched_infer import LeafPacket
    from poke_bot import features

    current = raw.get("current")
    if not isinstance(current, dict):
        raise R225DiagnosticError("stock leaf has no current state")
    actor = int(current.get("yourIndex", -1))
    if actor not in (0, 1):
        raise R225DiagnosticError("stock leaf has invalid acting seat")
    acting_deck = list(own_deck if actor == root_seat else opponent_guess)
    if len(acting_deck) != 60:
        raise R225DiagnosticError("stock leaf acting deck hypothesis is not 60 cards")
    if combos is None:
        combos = features.enumerate_action_combos(raw)
    materialized = [list(map(int, action)) for action in combos]
    if not materialized:
        raise R225DiagnosticError("stock leaf has no legal action combinations")
    return LeafPacket(
        obs=raw,
        your_deck=acting_deck,
        root_seat=int(root_seat),
        action_combos_override=materialized,
        matchup_route=int(route),
    )


def _lane_overlap_receipt(executions: Sequence[Any]) -> dict[str, Any]:
    """Require explicit core timestamps rather than infer concurrency from IDs."""

    # r222's physical preflight exposes these exact monotonic timestamps on
    # each result.  Treat their absence as a failed capability, not a made-up
    # zero-latency overlap measurement.
    required = (
        "search_begin_started_monotonic",
        "search_begin_completed_monotonic",
        "first_search_step_started_monotonic",
        "cleanup_completed_monotonic",
    )
    rows: list[dict[str, Any]] = []
    for execution in executions:
        values: dict[str, float] = {}
        for field in required:
            value = getattr(execution, field, None)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise R225DiagnosticError(
                    "r222 core did not attest required lane overlap timestamp: " + field
                )
            values[field] = float(value)
        if values["search_begin_completed_monotonic"] < values["search_begin_started_monotonic"]:
            raise R225DiagnosticError("lane SearchBegin timestamps are reversed")
        if values["first_search_step_started_monotonic"] < values["search_begin_completed_monotonic"]:
            raise R225DiagnosticError("lane SearchStep began before its SearchBegin completed")
        rows.append({"lane_id": int(execution.lane_id), **values})
    latest_begin = max(row["search_begin_completed_monotonic"] for row in rows)
    earliest_step = min(row["first_search_step_started_monotonic"] for row in rows)
    earliest_cleanup = min(row["cleanup_completed_monotonic"] for row in rows)
    all_started = latest_begin <= earliest_step
    if not all_started:
        raise R225DiagnosticError("not all eight lanes began before the first SearchStep")
    if latest_begin > earliest_cleanup:
        raise R225DiagnosticError("a stock lane completed before all eight lanes began")
    return {
        "all_eight_started_before_any_search_step": True,
        "all_eight_started_before_any_complete": True,
        "latest_search_begin_completed_monotonic": latest_begin,
        "earliest_search_step_started_monotonic": earliest_step,
        "earliest_cleanup_completed_monotonic": earliest_cleanup,
        "overlap_window_seconds": max(0.0, earliest_step - min(row["search_begin_started_monotonic"] for row in rows)),
        "lanes": rows,
    }


def _sealed_world_key(payload: Mapping[str, Any], *, prefix: str) -> str:
    """Hash a complete simulator descriptor, never a public-lookalike key."""

    try:
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise R225DiagnosticError("stock world descriptor is not canonical JSON") from exc
    return prefix + hashlib.sha256(encoded).hexdigest()


@dataclass
class R225DiagnosticEntrypoint:
    """Package-local once-per-process trigger used by the diagnostic wrapper."""

    stage: Path
    config: dict[str, Any]
    attempted: bool = False

    @classmethod
    def from_packaged_files(cls, stage: Path) -> "R225DiagnosticEntrypoint":
        return cls(stage=stage.resolve(), config=_check_config(stage.resolve()))

    def maybe_run(
        self,
        obs_dict: Mapping[str, Any],
        *,
        direct_action: Sequence[int],
        model: Any,
        policy: Any,
        deck: Sequence[int] | None,
    ) -> None:
        if self.attempted:
            return
        # A forced one-action prompt proves thread creation but not eight
        # workers exploring one tree.  Wait for the first genuinely branching
        # decision so all arenas start from the same state while virtual loss
        # can reserve distinct legal root paths.
        from poke_bot import features

        if len(features.enumerate_action_combos(dict(obs_dict))) < 2:
            return
        self.attempted = True
        started = time.monotonic()
        before_resources = _resource_probe()
        try:
            payload = self._run(
                dict(obs_dict),
                direct_action=tuple(int(item) for item in direct_action),
                model=model,
                policy=policy,
                deck=deck,
            )
            after_resources = _resource_probe()
            payload["resource_probe_before"] = before_resources
            payload["resource_probe_after"] = after_resources
            payload["elapsed_seconds"] = max(0.0, time.monotonic() - started)
            payload["status"] = (
                "viable"
                if after_resources.get("expected_h100_80gb_16vcpu_256gib_match") is True
                else "structural_pass_resource_unverified"
            )
            payload["gameplay_action_authority"] = "exact_archived_r195_direct_policy_only"
            _json_line("R225_EIGHT_LANE_DIAGNOSTIC", payload)
        except Exception as exc:
            after_resources = _resource_probe()
            _json_line(
                "R225_EIGHT_LANE_DIAGNOSTIC",
                {
                    "schema": R225_SCHEMA,
                    "status": "not_viable",
                    "failure_reason": f"{type(exc).__name__}: {exc}",
                    "elapsed_seconds": max(0.0, time.monotonic() - started),
                    "resource_probe_before": before_resources,
                    "resource_probe_after": after_resources,
                    "gameplay_action_authority": "exact_archived_r195_direct_policy_only",
                    "partial_lane_statistics_used": False,
                },
            )

    def _run(
        self,
        obs: dict[str, Any],
        *,
        direct_action: tuple[int, ...],
        model: Any,
        policy: Any,
        deck: Sequence[int] | None,
    ) -> dict[str, Any]:
        if model is None or policy is None or deck is None:
            raise R225DiagnosticError("archived r195 direct runtime was not initialized")
        own_deck = tuple(int(card) for card in deck)
        if len(own_deck) != 60:
            raise R225DiagnosticError("archived r195 deck is not exactly 60 cards")
        model_path = self.stage / "model.pt"
        tree_path = self.stage / "matchup_tree.json"
        library_path = self.stage / "cg/libcg.so"
        if _sha256_file(model_path) != R195_MODEL_SHA256:
            raise R225DiagnosticError("packaged r195 model digest changed")
        if _sha256_file(tree_path) != R195_MATCHUP_TREE_SHA256:
            raise R225DiagnosticError("packaged r195 Matchup Adapter tree digest changed")
        if (
            _sha256_file(library_path) != STOCK_LIBCG_SHA256
            or library_path.stat().st_size != STOCK_LIBCG_BYTES
        ):
            raise R225DiagnosticError("packaged stock r195 libcg identity changed")

        from poke_bot import cg_env, features
        from poke_bot.batched_infer import forward_leaf_batch
        from poke_bot.r222_stock_shared_tree_batch import (
            R222FrozenLeafMicrobatchBroker,
            R222SharedLogicalMCTSTree,
            R222SharedTreeLaneSeed,
            R222SharedTreeLeaf,
            R222SharedTreeLeafWork,
            R222StockSearchLanePool,
            R222StockSharedTreeMCTS,
            attest_loaded_stock_r195_library,
            attest_stock_r195_library,
            canonical_observation_fingerprint,
        )
        from poke_bot.r225_stock_native_lane import (
            R225StockNativeSearchLane,
            prewarm_stock_cg,
        )

        stock = attest_stock_r195_library(library_path)
        root_seat = int(dict(obs.get("current") or {}).get("yourIndex", -1))
        if root_seat not in (0, 1):
            raise R225DiagnosticError("real diagnostic observation has no acting seat")
        canonical = tuple(tuple(int(item) for item in action) for action in features.enumerate_action_combos(obs))
        if not canonical or direct_action not in canonical:
            raise R225DiagnosticError("archived r195 direct action is not in complete legal root order")
        route = _get_policy_route(policy)
        search_inputs = cg_env.build_search_inputs(
            obs, list(own_deck), opponent_deck_guess=list(own_deck)
        )
        opponent_guess = tuple(int(card) for card in search_inputs["opponent_deck"][:60])
        if len(opponent_guess) != 60:
            # The helper sizes by public remaining deck count.  A model leaf
            # needs a full hypothesis, so repeat a visible-r195 deck only as
            # a deterministic non-oracle simulation hypothesis.
            opponent_guess = tuple(own_deck)

        root_packet = _leaf_packet(
            raw=obs,
            root_seat=root_seat,
            own_deck=own_deck,
            opponent_guess=opponent_guess,
            route=route,
            combos=canonical,
        )
        root_evaluation = forward_leaf_batch(model, [root_packet])[0]
        priors = tuple(float(value) for value in root_evaluation.priors)
        if len(priors) != len(canonical) or any(
            not math.isfinite(value) or value < 0.0 for value in priors
        ):
            raise R225DiagnosticError("frozen r195 root prior vector is malformed")
        if not math.isfinite(sum(priors)) or sum(priors) <= 0.0:
            raise R225DiagnosticError("frozen r195 root prior vector has no mass")
        decision_fingerprint = canonical_observation_fingerprint(obs)
        shared_config = dict(self.config["shared_tree"])
        total_budget = float(shared_config["decision_budget_seconds"])
        deadline = time.monotonic() + total_budget
        root_world_key = _sealed_world_key(
            {
                "root_observation": obs,
                "search_inputs": {
                    str(key): [int(item) for item in value]
                    for key, value in sorted(search_inputs.items())
                },
            },
            prefix="r225-complete-root-world:",
        )
        tree = R222SharedLogicalMCTSTree(
            decision_fingerprint=decision_fingerprint,
            root_actions=canonical,
            root_priors=priors,
            root_actor=root_seat,
        )

        api, sim = prewarm_stock_cg()
        # The lane factory below calls ``sim.lib`` directly.  Attesting a
        # second ctypes handle proves only that a good archive member exists;
        # bind the *already loaded* native object to the exact package-local
        # stock r195 path before a single AgentStart call is allowed.
        loaded_stock = attest_loaded_stock_r195_library(
            sim.lib, expected_path=library_path
        )

        def factory(lane_id: int) -> R225StockNativeSearchLane:
            return R225StockNativeSearchLane(lane_id, lib=sim.lib, api_module=api)

        broker = R222FrozenLeafMicrobatchBroker(
            lambda packets: forward_leaf_batch(model, packets),
            checkpoint_digest=R195_MODEL_SHA256,
            max_batch_rows=LANE_COUNT,
            coalesce_ms=0.0,
        )
        lane_pool: Any | None = None
        batches: list[dict[str, Any]] = []
        try:
            lane_pool = R222StockSearchLanePool(factory, lane_count=LANE_COUNT)
            mcts = R222StockSharedTreeMCTS(
                tree=tree,
                lane_pool=lane_pool,
                leaf_broker=broker,
                root_observation=obs,
                root_actor=root_seat,
                direct_policy_action=direct_action,
            )
            for batch_index in range(int(shared_config["bounded_batches"])):
                if time.monotonic() >= deadline:
                    raise R225DiagnosticError("shared-tree diagnostic reached its bounded deadline")
                executions: dict[int, Any] = {}

                def leaf_key(reservation: Any, raw: Mapping[str, Any]) -> str:
                    return _sealed_world_key(
                        {
                            "root_world_key": reservation.root_world_key,
                            "selected_path": [list(action) for action in reservation.action_path],
                            "full_leaf_observation": raw,
                        },
                        prefix="r225-complete-leaf-world:",
                    )

                def build_leaf_work(reservation: Any, execution: Any) -> Any:
                    raw = _raw_observation(execution.final_observation)
                    executions[int(execution.lane_id)] = execution
                    # A chance prompt is evaluated but never selected or
                    # advanced: manual coin is enforced by the stock transport.
                    packet = _leaf_packet(
                        raw=raw,
                        root_seat=root_seat,
                        own_deck=own_deck,
                        opponent_guess=opponent_guess,
                        route=route,
                    )
                    return R222SharedTreeLeafWork(
                        model_packet=packet,
                        safe_model_input_key=leaf_key(reservation, raw),
                    )

                def decode_leaf(reservation: Any, execution: Any, leaf: Any) -> Any:
                    raw = _raw_observation(execution.final_observation)
                    current = raw.get("current")
                    if not isinstance(current, Mapping):
                        raise R225DiagnosticError("stock leaf has no acting-seat state")
                    actor = int(current.get("yourIndex", -1))
                    if actor not in (0, 1):
                        raise R225DiagnosticError("stock leaf has invalid acting seat")
                    if _is_explicit_chance(raw):
                        return R222SharedTreeLeaf(
                            value=float(leaf.value),
                            semantic_state_key=leaf_key(reservation, raw),
                            actor=actor,
                            expandable=False,
                            boundary_kind="pre_random_frozen_model_leaf",
                        )
                    combos = tuple(tuple(int(item) for item in action) for action in leaf.combos)
                    priors = tuple(float(value) for value in leaf.priors)
                    return R222SharedTreeLeaf(
                        value=float(leaf.value),
                        semantic_state_key=leaf_key(reservation, raw),
                        actor=actor,
                        legal_actions=combos,
                        priors=priors,
                        expandable=True,
                    )

                receipt = mcts.run_eight(
                    tuple(
                        R222SharedTreeLaneSeed(
                            lane_id=lane_id,
                            search_inputs=search_inputs,
                            root_world_key=_sealed_world_key(
                                {
                                    "complete_root_world": root_world_key,
                                    "internal_simulator_arena_lane": lane_id,
                                },
                                prefix="r225-arena-world:",
                            ),
                        )
                        for lane_id in range(LANE_COUNT)
                    ),
                    deadline_monotonic=deadline,
                    make_leaf_work=build_leaf_work,
                    decode_model_leaf=decode_leaf,
                )
                if sorted(executions) != list(range(LANE_COUNT)):
                    raise R225DiagnosticError("shared-tree batch omitted lane execution telemetry")
                overlap = _lane_overlap_receipt([executions[index] for index in range(LANE_COUNT)])
                if (
                    receipt.requested_lane_count != LANE_COUNT
                    or receipt.unique_raw_handle_count != LANE_COUNT
                    or receipt.active_lane_count != LANE_COUNT
                    or receipt.max_concurrent_active_lanes != LANE_COUNT
                    or not receipt.all_eight_began_before_first_step
                    or receipt.partial_lane_statistics_used
                    or not receipt.all_lane_work_finished_before_return
                    or receipt.forest_merge_used
                    or receipt.root_visit_delta != LANE_COUNT
                    or receipt.completed_backed_simulations != LANE_COUNT
                    or receipt.outstanding_reservations != 0
                    or receipt.outstanding_virtual_loss != 0
                    or receipt.native_search_id_cross_lane_reuse != 0
                ):
                    raise R225DiagnosticError("shared-tree core receipt is incomplete")
                batches.append(
                    {
                        "batch_index": batch_index,
                        "receipt": receipt.as_dict(),
                        "lane_overlap": overlap,
                    }
                )
        finally:
            # The shared-tree core rolls back its virtual loss on any failed
            # transaction.  Join both persistent internal-arena workers and
            # the frozen-model broker before returning the direct action.
            try:
                if lane_pool is not None:
                    lane_pool.close()
            finally:
                broker.close()

        cleanup_completed = time.monotonic()
        cleanup_slack = float(shared_config.get("deadline_cleanup_slack_seconds", 0.0))
        if cleanup_completed > deadline + cleanup_slack:
            raise R225DiagnosticError("shared-tree cleanup exceeded its bounded deadline slack")
        if tree.outstanding_reservations or tree.outstanding_virtual_loss:
            raise R225DiagnosticError("shared-tree returned with outstanding reservations")
        eight_elapsed = sum(float(row["receipt"]["elapsed_seconds"]) for row in batches)
        eight_backups = sum(int(row["receipt"]["completed_backed_simulations"]) for row in batches)
        eight_rate = eight_backups / eight_elapsed if eight_elapsed > 0.0 else 0.0
        if eight_backups < LANE_COUNT or eight_rate <= 0.0:
            raise R225DiagnosticError("eight-lane probe produced no backed simulations")
        per_lane_totals = []
        for lane_id in range(LANE_COUNT):
            rows = [
                next(
                    row
                    for row in batch["receipt"]["per_lane"]
                    if int(row["lane_id"]) == lane_id
                )
                for batch in batches
            ]
            per_lane_totals.append(
                {
                    "lane_id": lane_id,
                    "transactions": len(rows),
                    "search_begin_calls": sum(int(row["search_begin_calls"]) for row in rows),
                    "search_step_calls": sum(int(row["search_step_calls"]) for row in rows),
                    "search_release_calls": sum(int(row["search_release_calls"]) for row in rows),
                    "search_end_calls": sum(int(row["search_end_calls"]) for row in rows),
                    "backups": sum(int(row["backups"]) for row in rows),
                    "max_trajectory_depth": max(int(row["trajectory_depth"]) for row in rows),
                }
            )
        tree_receipt = {
            "one_shared_logical_tree": True,
            "shared_logical_tree_id": tree.tree_id,
            "root_visits": tree.root_visits,
            "path_reservations": sum(int(row["receipt"]["virtual_loss_reserved"]) for row in batches),
            "completed_backups": eight_backups,
            "virtual_loss_after": tree.outstanding_virtual_loss,
            "outstanding_reservations": tree.outstanding_reservations,
            "zero_outstanding_reservations": True,
            "inflight_leaf_eval_coalescing": sum(int(row["receipt"]["inflight_eval_coalesced"]) for row in batches),
            "cache_hits": sum(int(row["receipt"]["eval_cache_hits"]) for row in batches),
            "duplicate_paths_avoided": sum(int(row["receipt"]["duplicate_path_avoided"]) for row in batches),
            "unavoidable_distinct_hidden_random_world_repeats": sum(
                int(row["receipt"]["unavoidable_distinct_world_repeats"])
                for row in batches
            ),
            "unsafe_public_lookalike_merge_prohibited": True,
            "unsafe_public_lookalike_merges": 0,
        }
        return {
            "schema": R225_SCHEMA,
            "stock_libcg": loaded_stock.as_dict(),
            "stock_libcg_archive_member": stock.as_dict(),
            "frozen_model_sha256": R195_MODEL_SHA256,
            "matchup_tree_sha256": R195_MATCHUP_TREE_SHA256,
            "matchup_adapter_route": route,
            "direct_policy_action": list(direct_action),
            "simulator_topology": {
                "kaggle_processes": 1,
                "loaded_stock_libcg_dsos": 1,
                "internal_agent_start_arenas": LANE_COUNT,
                "competition_agents": 1,
                "unique_raw_handle_count": LANE_COUNT,
            },
            "eight_lane": {
                "requested_lane_count": LANE_COUNT,
                "active_lane_count": LANE_COUNT,
                "batches": batches,
                "completed_backed_simulations": eight_backups,
                "elapsed_seconds": eight_elapsed,
                "backed_simulations_per_second": eight_rate,
                "per_lane": per_lane_totals,
            },
            "shared_tree": tree_receipt,
            "private_random_samples": 0,
            "guessed_random_rules": 0,
            "unobserved_random_advances": 0,
            "partial_lane_statistics_used": False,
            "deadline_cleanup": {
                "deadline_monotonic": deadline,
                "completed_before_deadline": time.monotonic() <= deadline,
                "cleanup_slack_seconds": float(
                    shared_config.get("deadline_cleanup_slack_seconds", 0.0)
                ),
                "no_background_native_work_after_return": True,
            },
        }

__all__ = ["R225DiagnosticEntrypoint", "R225DiagnosticError", "R225_SCHEMA"]
