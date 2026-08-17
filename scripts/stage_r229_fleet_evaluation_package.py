#!/usr/bin/env python3
"""Seal the r253 BO evaluator with restarting serial root rollouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Sequence

R228_ARCHIVE = "sha256:59531249f106d55d6606b186aee3d3a3e5ec8a3f0e9760c963e08cfd8b9d67d4"
MODEL = "sha256:261d367e131eeaacc62f86f8f0443250d187daf82bcbcaa88fafad7c9199cc3a"
TREE = "sha256:e60efb2f31225c89dbd78169d26f54bc2014cb4ab0bb1587ac2a9fe0194c9049"
R233_RUNTIME_COMPONENTS = {
    "main.py": "sha256:ab517561c58ee32f0d15fdcfb4fccd1edc51ec606ae3161aa501ac13979b3f5b",
    "poke_bot/r228_async_shared_tree_queue.py": "sha256:3729da928a7d9754fa0d45597f0a06abffac178a9c7f9b6f01ca0a98395aa4d8",
    "poke_bot/r228_kaggle_async_runtime.py": "sha256:d1dd78189df57253d0354aaf57a66fc99493b1e8ac3c4c2771e003b8c6e576a9",
}
WHEEL_FILENAME = "kaggle_environments-1.32.6-py3-none-any.whl"
WHEEL_SHA256 = "sha256:e70a7d7765b16deb1fcfa00532eb5197f28bc9fbfa07a0eee150a17d67bd77ab"
WHEEL_SIZE_BYTES = 60_677_343
NATIVE_LIBRARY_UPDATE_COMMIT = "03ab2cc235b719e5a3bd0d19e2d2c62c65a4c303"
CANONICAL_LIBRARIES = {
    "linux_x86_64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.so",
        "package_relative_path": "cg/libcg.so",
        "sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "size_bytes": 1_342_400,
    },
    "linux_aarch64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg-arm64.so",
        "package_relative_path": "cg/libcg-arm64.so",
        "sha256": "sha256:1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
        "size_bytes": 1_296_464,
    },
    "macos_arm64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/libcg.dylib",
        "package_relative_path": "cg/libcg.dylib",
        "sha256": "sha256:7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
        "size_bytes": 1_245_544,
    },
    "windows_x86_64": {
        "wheel_member": "kaggle_environments/envs/cabt/cg/cg.dll",
        "package_relative_path": "cg/cg.dll",
        "sha256": "sha256:eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
        "size_bytes": 1_525_248,
    },
}
OVERLAYS = {
    "run_r229_process_watchdog.py": "scripts/run_r229_process_watchdog.py",
    "run_r229_mirror_game.py": "scripts/run_r229_mirror_game.py",
    "poke_bot/r249_process_search_lane.py": "poke_bot/r249_process_search_lane.py",
    "poke_bot/r250_recovering_serial_tree.py": "poke_bot/r250_recovering_serial_tree.py",
    "poke_bot/r252_search_leaf_boundary.py": "poke_bot/r252_search_leaf_boundary.py",
    "poke_bot/r253_restarting_serial_mcts.py": "poke_bot/r253_restarting_serial_mcts.py",
}


class StageError(RuntimeError):
    pass


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def tree_sha(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(row for row in root.rglob("*") if row.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return "sha256:" + digest.hexdigest()


def raise_packaged_action_cap(path: Path) -> None:
    old = b"MAX_ACTION_COMBOS: int = 4096"
    new = b"MAX_ACTION_COMBOS: int = 65536"
    payload = path.read_bytes()
    if payload.count(old) != 1 or new in payload:
        raise StageError("base r228 feature cap is not the exact 4,096 contract")
    path.write_bytes(payload.replace(old, new, 1))


def _replace_once(payload: bytes, old: bytes, new: bytes, *, label: str) -> bytes:
    if payload.count(old) != 1 or new in payload:
        raise StageError(f"r233 runtime is not the exact pre-transform source: {label}")
    return payload.replace(old, new, 1)


def repair_r233_runtime_for_r229(path: Path) -> None:
    """Apply BO identity, serial topology, and bounded process recovery."""

    payload = path.read_bytes()
    payload = _replace_once(
        payload,
        b'''"""Stock-libcg eight-worker shared-tree action authority for the r228 smoke.

This is intentionally a small viability runtime.  One competition process owns
one frozen model and one shared search tree per decision.  Eight persistent
thread-affine ``AgentStart`` arenas advance independent simulator states and
feed ready leaves to the same model-backed coordinator queue.
"""''',
        b'''"""Serial process-owned stock-libcg MCTS for the r253 BO evaluation.

One authoritative game process owns the frozen model and one logical tree. One
thread-affine proxy talks to one child-owned ``AgentStart`` handle; every MCTS
rollout independently reopens the exact physical root, expands and backs one
leaf, then boundedly releases before the next rollout.
"""''',
        label="serial runtime description",
    )
    old_hashes = b'''STOCK_LIBRARY_SHA256 = {
    "libcg.so": "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c",
    "libcg.dylib": "77bb978a8129b094452679e0daf0da69593afda7331685f4642c0d4a94d39d82",
    "libcg-arm64.so": "030b4728ce9fb9e90b75830b7cf7236f71859732a05ec4a377078eee0421bbe5",
    "cg.dll": "9ea2b0a751029689bff3ddccb5f29a98edd46961dad264490ed121ef704fb500",
}'''
    new_hashes = b'''STOCK_LIBRARY_SHA256 = {
    "libcg.so": "d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
    "libcg.dylib": "7a157f045d333f99d1996d49c12bdbdd148072a619af246385c7295518776e30",
    "libcg-arm64.so": "1670740b73fab46586fd25c0a1f96608ea75b1f39381d66a0b8d9486bea6d4a2",
    "cg.dll": "eae88634e26dc31d94150a4d8202fc9d32596b8c688ef67e14cb4088cd4d5771",
}'''
    payload = _replace_once(payload, old_hashes, new_hashes, label="r236 hashes")
    payload = _replace_once(
        payload,
        b'SCHEMA = "poke_bot.r228_async_eight_worker_kaggle_viability/v1"',
        b'SCHEMA = "poke_bot.r253_restarting_serial_mcts_fleet_mirror/v1"',
        label="r253 runtime schema",
    )
    payload = _replace_once(
        payload,
        b'DECISION_PREFIX = "R228_ASYNC_EIGHT_WORKER_DECISION"',
        b'DECISION_PREFIX = "R253_RESTARTING_SERIAL_MCTS_DECISION"',
        label="r253 decision marker",
    )
    payload = _replace_once(
        payload,
        b"search_inputs=tuple(dict(search_inputs) for _ in range(8))",
        b"search_inputs=tuple(dict(search_inputs) for _ in range(1))",
        label="one serial search input",
    )
    old_lane_receipt = b'''                    "per_lane_depth": list(receipt.per_lane_depth),
                    "search_release_calls": receipt.search_release_calls,'''
    new_lane_receipt = b'''                    "per_lane_depth": list(receipt.per_lane_depth),
                    "per_lane_search_id_chains": [
                        list(chain) for chain in receipt.per_lane_search_id_chains
                    ],
                    "handle_scoped_first_search_id_composite_states": [
                        {
                            "lane_id": lane_id,
                            "handle_identity": receipt.per_lane_handle_identities[lane_id],
                            "first_search_id": receipt.per_lane_search_id_chains[lane_id][0],
                        }
                        for lane_id in range(1)
                    ],
                    "rollout_count": receipt.rollout_count,
                    "rollout_search_id_chains": [
                        list(chain) for chain in receipt.rollout_search_id_chains
                    ],
                    "rollout_search_begin_states": [
                        {
                            "rollout_index": rollout_index,
                            "handle_identity": receipt.per_lane_handle_identities[0],
                            "first_search_id": chain[0],
                        }
                        for rollout_index, chain in enumerate(
                            receipt.rollout_search_id_chains
                        )
                    ],
                    "rollout_root_actions": [
                        list(action) for action in receipt.rollout_root_actions
                    ],
                    "root_action_visit_counts": list(
                        receipt.root_action_visit_counts
                    ),
                    "distinct_root_actions_visited": (
                        receipt.distinct_root_actions_visited
                    ),
                    "max_rollout_depth": receipt.max_rollout_depth,
                    "rollout_stop_reason": receipt.stop_reason,
                    "rollout_ceiling": receipt.rollout_ceiling,
                    "search_release_calls": receipt.search_release_calls,'''
    payload = _replace_once(
        payload, old_lane_receipt, new_lane_receipt, label="serial search id"
    )
    old_arena_receipt = b'''                    "arena_count": receipt.arena_count,
                    "unique_handle_count": receipt.unique_handle_count,'''
    new_arena_receipt = b'''                    "requested_simulator_lane_count": 1,
                    "active_simulator_lane_count": receipt.arena_count,
                    "arena_count": receipt.arena_count,
                    "unique_handle_count": receipt.unique_handle_count,
                    "per_lane_handle_identities": list(
                        receipt.per_lane_handle_identities
                    ),'''
    payload = _replace_once(
        payload, old_arena_receipt, new_arena_receipt, label="requested active lanes"
    )
    old_actor = b'''            decoded[index] = DecodedLeaf(
                state_key=_state_key(lane_id=frontier.lane_id, raw=frontier.raw),
                value=float(leaf.value),'''
    new_actor = b'''            current = frontier.raw.get("current")
            if not isinstance(current, Mapping):
                raise R228GameplayError("simulator leaf has no current state")
            actor = int(current.get("yourIndex", -1))
            if actor not in (0, 1):
                raise R228GameplayError("simulator leaf has invalid acting seat")
            decoded[index] = DecodedLeaf(
                state_key=_state_key(lane_id=frontier.lane_id, raw=frontier.raw),
                value=float(leaf.value),'''
    payload = _replace_once(payload, old_actor, new_actor, label="leaf actor seat")
    payload = _replace_once(
        payload,
        b''')

SCHEMA = "poke_bot.r253_restarting_serial_mcts_fleet_mirror/v1"''',
        b''')
from .r250_recovering_serial_tree import R250SerialRecoveryExhausted
from .r252_search_leaf_boundary import classify_search_leaf

SCHEMA = "poke_bot.r253_restarting_serial_mcts_fleet_mirror/v1"''',
        label="module-global recovery and leaf-boundary imports",
    )
    payload = _replace_once(
        payload,
        b'''            combos = tuple(
                tuple(int(item) for item in action)
                for action in features.enumerate_action_combos(raw)
            )
            if not combos:
                raise R228GameplayError("nonterminal simulator leaf has no legal actions")
            boundary = _chance_boundary(raw)
            packets.append(self._leaf_packet(raw, combos=combos))
            pending.append((index, frontier, combos, boundary))''',
        b'''            ordered_count = int(features.ordered_action_count(raw))
            boundary_row = classify_search_leaf(
                raw, ordered_action_count=ordered_count
            )
            boundary = bool(boundary_row.is_boundary)
            if boundary:
                representative = boundary_row.representative_action
                if representative is None:
                    raise R228GameplayError(
                        "value-only simulator boundary has no representative"
                    )
                combos = (tuple(int(item) for item in representative),)
                reason = str(boundary_row.reason)
                context["internal_value_boundary_count"] += 1
                reasons = context["internal_value_boundary_reasons"]
                reasons[reason] = int(reasons.get(reason, 0)) + 1
            else:
                combos = tuple(
                    tuple(int(item) for item in action)
                    for action in features.enumerate_action_combos(raw)
                )
            context["max_internal_ordered_action_count"] = max(
                int(context["max_internal_ordered_action_count"]), ordered_count
            )
            if not combos:
                raise R228GameplayError("nonterminal simulator leaf has no legal actions")
            packets.append(self._leaf_packet(raw, combos=combos))
            pending.append((index, frontier, combos, boundary))''',
        label="value-only chance and oversized internal leaf boundary",
    )
    payload = _replace_once(
        payload,
        b'''        root_leaf = forward_leaf_batch(self.model, [root_packet])[0]
        root_priors = tuple(float(value) for value in root_leaf.priors)''',
        b'''        print(
            f"R252_SERIAL_ROOT_MODEL_BEGIN legal_action_count={len(legal)}",
            flush=True,
        )
        root_leaf = forward_leaf_batch(self.model, [root_packet])[0]
        root_priors = tuple(float(value) for value in root_leaf.priors)
        print(
            f"R252_SERIAL_ROOT_MODEL_READY legal_action_count={len(root_priors)}",
            flush=True,
        )''',
        label="bounded-search root model phase markers",
    )
    old_lane_import = b'''        from .r225_stock_native_lane import (
            R225StockNativeSearchLane,
            prewarm_stock_cg,
        )'''
    new_lane_import = b'''        from .r225_stock_native_lane import prewarm_stock_cg
        from .r249_process_search_lane import R249ProcessSearchLane
        from .r250_recovering_serial_tree import (
            R250RecoveringSerialTree,
            R250SerialRecoveryExhausted,
        )'''
    payload = _replace_once(
        payload, old_lane_import, new_lane_import, label="r250 serial process imports"
    )
    payload = _replace_once(
        payload,
        b'''        def arena_factory(lane_id: int) -> Any:
            return R225StockNativeSearchLane(lane_id, lib=sim.lib, api_module=api)''',
        b'''        def arena_factory(lane_id: int) -> Any:
            return R249ProcessSearchLane(lane_id, stage=self.stage)''',
        label="r250 serial process lane factory",
    )
    payload = _replace_once(
        payload,
        b"self._search = PersistentAsyncEightWorkerMCTS(",
        b"self._search = R250RecoveringSerialTree(",
        label="r250 recovering serial tree",
    )
    payload = _replace_once(
        payload,
        b'''            "history_previous_actions": list(self.policy.previous_action_history),
        }''',
        b'''            "history_previous_actions": list(self.policy.previous_action_history),
            "internal_value_boundary_count": 0,
            "internal_value_boundary_reasons": {},
            "max_internal_ordered_action_count": 0,
        }''',
        label="internal leaf boundary decision counters",
    )
    payload = _replace_once(
        payload,
        b'''        started = time.monotonic()
        try:
            receipt = self._search.run_decision(''',
        b'''        started = time.monotonic()
        print("R253_RESTARTING_SERIAL_SEARCH_BEGIN", flush=True)
        try:
            receipt = self._search.run_decision(''',
        label="serial search phase marker",
    )
    old_failure = b'''        except AsyncEightWorkerError as exc:
            # The core emits this exact post-cleanup failure only when the
            # deadline produced zero backups.  Structural/native failures are
            # deliberately not downgraded in this viability submission.
            if "completed no backups" not in str(exc):
                raise
            selected = tuple(
                int(item)
                for item in self.policy._factorized_greedy_prepared(
                    obs,
                    board,
                    target_source="r228_clean_deadline_fallback",
                )
            )
            if selected not in legal:
                raise R228GameplayError("clean-deadline frozen fallback was illegal")
            receipt = None
            mode = "clean_deadline_zero_backup_frozen_model_fallback"'''
    new_failure = b'''        except AsyncEightWorkerError as exc:
            if isinstance(exc, R250SerialRecoveryExhausted):
                # Both complete serial attempts failed at a contained native
                # boundary.  The root direct counterfactual was already
                # computed from this exact frozen policy state.
                selected = direct_action
                receipt = None
                mode = "bounded_lane_recovery_exhausted_direct_fallback"
            else:
                # Fewer than two independent roots cannot receive MCTS action
                # authority. Model/tree/identity failures remain hard failures.
                if "completed fewer than two independent root rollouts" in str(exc):
                    selected = direct_action
                    receipt = None
                    mode = "clean_deadline_insufficient_rollouts_frozen_model_fallback"
                elif "completed no backups" not in str(exc):
                    raise
                else:
                    selected = tuple(
                        int(item)
                        for item in self.policy._factorized_greedy_prepared(
                            obs,
                            board,
                            target_source="r228_clean_deadline_fallback",
                        )
                    )
                    if selected not in legal:
                        raise R228GameplayError("clean-deadline frozen fallback was illegal")
                    receipt = None
                    mode = "clean_deadline_zero_backup_frozen_model_fallback"'''
    payload = _replace_once(
        payload, old_failure, new_failure, label="r250 bounded recovery fallback"
    )
    payload = _replace_once(
        payload,
        b'''        finally:
            self._decision = None''',
        b'''        finally:
            active_context = self._decision or {}
            search_boundary_metrics = {
                "internal_value_boundary_count": int(
                    active_context.get("internal_value_boundary_count", 0)
                ),
                "internal_value_boundary_reasons": dict(
                    active_context.get("internal_value_boundary_reasons", {})
                ),
                "max_internal_ordered_action_count": int(
                    active_context.get("max_internal_ordered_action_count", 0)
                ),
                "internal_ordered_action_expansion_ceiling": 64,
                "explicit_chance_probability_distribution_assumed": False,
                "explicit_chance_always_stops_before_random_resolution": True,
                "internal_boundary_has_action_or_child_authority": False,
            }
            self._decision = None''',
        label="internal value-boundary metrics survive decision cleanup",
    )
    payload = _replace_once(
        payload,
        b'''            "action_changed": selected != direct_action,
        }''',
        b'''            "action_changed": selected != direct_action,
            "lane_process_recovery": dict(self._search.last_decision_recovery),
            **search_boundary_metrics,
        }''',
        label="r250 decision recovery telemetry",
    )
    path.write_bytes(payload)


def repair_r233_queue_for_r250_serial(path: Path) -> None:
    """Set one lane and contain consumed worker errors without a second wait."""

    payload = path.read_bytes()
    payload = _replace_once(
        payload,
        b'''"""Minimal persistent asynchronous eight-worker shared-tree search.

This module is deliberately small.  Eight thread-affine simulator arenas keep
their native search states alive while a coordinator repeatedly:

1. reserves a legal edge from one shared tree;
2. lets the owning simulator worker advance exactly one step;
3. microbatches whichever frontier states are ready;
4. backs those values into the same tree; and
5. immediately queues the next edge for those lanes.

It is a viability implementation, not a production-strength MCTS variant.
"""''',
        b'''"""Minimal persistent serial shared-tree search for r252.

One thread-affine proxy keeps one process-owned native search state alive while
the coordinator repeatedly reserves an edge, advances one simulator step,
evaluates the single frontier leaf, backs it into one logical tree, and queues
the next edge.  The legacy class names remain package-compatibility details.
"""''',
        label="serial queue description",
    )
    replacements = (
        (b"LANES = 8", b"LANES = 1", "serial lane count"),
        (
            b'exactly eight search-input rows are required',
            b'exactly one search-input row is required',
            "serial search input contract",
        ),
        (
            b'decision deadline expired before eight arenas opened',
            b'decision deadline expired before one arena opened',
            "serial arena-open contract",
        ),
        (
            b'asynchronous eight-worker decision failed',
            b'asynchronous serial-process decision failed',
            "serial failure marker",
        ),
    )
    for old, new, label in replacements:
        payload = _replace_once(payload, old, new, label=label)
    payload = _replace_once(
        payload,
        b'''class AsyncEightWorkerError(RuntimeError):
    """The eight-worker search could not return a trustworthy decision."""''',
        b'''class AsyncEightWorkerError(RuntimeError):
    """The serial search could not return a trustworthy decision."""''',
        label="serial error description",
    )
    payload = _replace_once(
        payload,
        b'''class PersistentAsyncEightWorkerMCTS:
    """Eight persistent simulator workers feeding one coordinator-owned tree."""''',
        b'''class PersistentAsyncEightWorkerMCTS:
    """One persistent process-owned simulator lane feeding one logical tree."""''',
        label="serial coordinator description",
    )
    payload = _replace_once(
        payload,
        b'''    unique_handle_count: int
    search_begin_calls: int''',
        b'''    unique_handle_count: int
    per_lane_handle_identities: tuple[int | str, ...]
    search_begin_calls: int''',
        label="handle-scoped serial search identity receipt",
    )
    payload = _replace_once(
        payload,
        b'''            unique_handle_count=len({worker.handle_identity for worker in self._workers}),
            search_begin_calls=LANES,''',
        b'''            unique_handle_count=len({worker.handle_identity for worker in self._workers}),
            per_lane_handle_identities=tuple(
                worker.handle_identity for worker in self._workers
            ),
            search_begin_calls=LANES,''',
        label="serial handle identity",
    )
    payload = _replace_once(
        payload,
        b'''            if command is None:
                return''',
        b'''            if command is None:
                close_owned_process = getattr(self._arena, "close", None)
                if callable(close_owned_process):
                    close_owned_process()
                return''',
        label="bounded serial process-lane shutdown",
    )
    payload = _replace_once(
        payload,
        b'''                for row in ready:
                    if row.error is not None or row.kind != "step" or row.search_id is None:
                        raise AsyncEightWorkerError(
                            f"lane {row.lane_id} SearchStep failed: {row.error or row.kind}"
                        )
                    if row.lane_id not in in_flight:
                        raise AsyncEightWorkerError("received an untracked simulator result")
                    step_rows.append(row)''',
        b'''                for row in ready:
                    if row.error is not None or row.kind != "step" or row.search_id is None:
                        # This row is the one and only completion for the
                        # issued command.  Remove its reservation before the
                        # finally-block drains genuinely outstanding work;
                        # otherwise the coordinator waits forever for a
                        # second response that cannot exist.
                        tracked = in_flight.pop(row.lane_id, None)
                        if tracked is not None:
                            _context, failed_edge = tracked
                            _context.in_flight = False
                            if failed_edge.virtual_loss > 0:
                                failed_edge.virtual_loss -= 1
                        raise AsyncEightWorkerError(
                            f"lane {row.lane_id} SearchStep failed: {row.error or row.kind}"
                        )
                    if row.lane_id not in in_flight:
                        raise AsyncEightWorkerError("received an untracked simulator result")
                    step_rows.append(row)''',
        label="consumed serial worker error clears in-flight reservation",
    )
    payload = _replace_once(
        payload,
        b'''            while len(close_rows) < LANES:
                row = self._completions.get()
                if row.kind == "close":
                    close_rows[row.lane_id] = row
                else:
                    structural_error = structural_error or row.error or AsyncEightWorkerError(
                        f"unexpected result during cleanup: {row.kind}"
                    )''',
        b'''            while len(close_rows) < LANES:
                row = self._completions.get()
                # An error is still the terminal cleanup response for this
                # lane.  Count it, preserve it as structural evidence, and
                # finish boundedly instead of waiting for a nonexistent
                # additional close row.
                close_rows[row.lane_id] = row
                if row.kind != "close" or row.error is not None:
                    structural_error = structural_error or row.error or AsyncEightWorkerError(
                        f"unexpected result during cleanup: {row.kind}"
                    )''',
        label="serial cleanup error is terminal lane response",
    )
    payload = _replace_once(
        payload,
        b'''                    step_rows.append(row)
                packets = [self._make_packet(row.lane_id, row.observation) for row in step_rows]
                leaves = tuple(self._evaluate_batch(packets))
                if len(leaves) != len(step_rows):
                    raise AsyncEightWorkerError("GPU evaluator returned a partial microbatch")
                microbatches.append(len(leaves))
                for row, leaf in zip(step_rows, leaves):
                    leaf.validate()
                    context, edge = in_flight.pop(row.lane_id)
                    context.in_flight = False''',
        b'''                    step_rows.append(row)
                # Every row above is already the terminal response to its
                # issued native command.  Resolve those reservations before
                # packet construction/model evaluation, both of which may
                # raise.  The final drain then waits only for commands whose
                # worker response has not actually arrived.
                resolved_rows: list[tuple[_WorkerResult, _LaneContext, _Edge]] = []
                for row in step_rows:
                    context, edge = in_flight.pop(row.lane_id)
                    context.in_flight = False
                    resolved_rows.append((row, context, edge))
                try:
                    packets = [
                        self._make_packet(row.lane_id, row.observation)
                        for row, _context, _edge in resolved_rows
                    ]
                    leaves = tuple(self._evaluate_batch(packets))
                    if len(leaves) != len(resolved_rows):
                        raise AsyncEightWorkerError(
                            "GPU evaluator returned a partial microbatch"
                        )
                    for leaf in leaves:
                        leaf.validate()
                except Exception:
                    for _row, _context, failed_edge in resolved_rows:
                        if failed_edge.virtual_loss > 0:
                            failed_edge.virtual_loss -= 1
                    raise
                microbatches.append(len(leaves))
                for (row, context, edge), leaf in zip(resolved_rows, leaves):''',
        label="received serial steps resolve before parent leaf evaluation",
    )
    path.write_bytes(payload)


def repair_r233_main_for_r250_serial(path: Path) -> None:
    """Bind serial markers into the pre-r234 entrypoint."""

    payload = path.read_bytes()
    replacements = (
        (
            b"r228 asynchronous eight-worker viability smoke",
            b"r253 restarting-serial MCTS fleet mirror",
            "entrypoint description",
        ),
        (
            b"R228_ASYNC_EIGHT_WORKER_FULL_GAMEPLAY_SUCCESS",
            b"R253_RESTARTING_SERIAL_FULL_GAMEPLAY_SUCCESS",
            "full-game marker",
        ),
        (
            b"poke_bot.r228_async_eight_worker_kaggle_viability/v1",
            b"poke_bot.r253_restarting_serial_mcts_fleet_mirror/v1",
            "full-game schema",
        ),
        (
            b"R228_ASYNC_EIGHT_WORKER_HARD_FAILURE",
            b"R253_RESTARTING_SERIAL_HARD_FAILURE",
            "hard-failure marker",
        ),
    )
    for old, new, label in replacements:
        payload = _replace_once(payload, old, new, label=label)
    path.write_bytes(payload)


def overlay_r233_runtime(*, source: Path, destination: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative, expected in R233_RUNTIME_COMPONENTS.items():
        source_path = source / relative
        if not source_path.is_file() or sha(source_path) != expected:
            raise StageError(f"r233 runtime source drifted: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)
        hashes[relative] = expected
    repair_r233_main_for_r250_serial(destination / "main.py")
    repair_r233_queue_for_r250_serial(
        destination / "poke_bot/r228_async_shared_tree_queue.py"
    )
    repair_r233_runtime_for_r229(destination / "poke_bot/r228_kaggle_async_runtime.py")
    hashes["main.py"] = sha(destination / "main.py")
    hashes["poke_bot/r228_async_shared_tree_queue.py"] = sha(
        destination / "poke_bot/r228_async_shared_tree_queue.py"
    )
    hashes["poke_bot/r228_kaggle_async_runtime.py"] = sha(
        destination / "poke_bot/r228_kaggle_async_runtime.py"
    )
    return hashes


def verify_canonical_native_set(root: Path) -> dict[str, dict[str, object]]:
    expected_paths = {row["package_relative_path"] for row in CANONICAL_LIBRARIES.values()}
    observed_paths = {
        path.relative_to(root).as_posix()
        for path in (root / "cg").iterdir()
        if path.is_file() and (path.name.startswith("libcg") or path.name == "cg.dll")
    }
    if observed_paths != expected_paths:
        raise StageError("package does not contain exactly the canonical four-member libcg set")
    receipt: dict[str, dict[str, object]] = {}
    for platform_name, row in CANONICAL_LIBRARIES.items():
        path = root / str(row["package_relative_path"])
        size = path.stat().st_size
        digest = sha(path)
        if size != row["size_bytes"] or digest != row["sha256"]:
            raise StageError(f"canonical libcg member drifted: {row['package_relative_path']}")
        receipt[platform_name] = {
            "path": row["package_relative_path"],
            "sha256": digest,
            "size_bytes": size,
        }
    return receipt


def overlay_canonical_native_set(*, wheel: Path, destination: Path) -> dict[str, dict[str, object]]:
    if wheel.stat().st_size != WHEEL_SIZE_BYTES or sha(wheel) != WHEEL_SHA256:
        raise StageError("input is not the exact official Kaggle Environments 1.32.6 wheel")
    with zipfile.ZipFile(wheel) as archive:
        rows = archive.infolist()
        for row in CANONICAL_LIBRARIES.values():
            matches = [info for info in rows if info.filename == row["wheel_member"]]
            if len(matches) != 1:
                raise StageError(f"official wheel member is missing or duplicated: {row['wheel_member']}")
            info = matches[0]
            mode = info.external_attr >> 16
            if info.is_dir() or stat.S_IFMT(mode) == stat.S_IFLNK or info.file_size != row["size_bytes"]:
                raise StageError(f"official wheel member metadata drifted: {row['wheel_member']}")
            target = destination / str(row["package_relative_path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source_stream, target.open("wb") as sink:
                shutil.copyfileobj(source_stream, sink)
    return verify_canonical_native_set(destination)


def safe_extract(archive: Path, destination: Path) -> None:
    with tarfile.open(archive, "r:gz") as tar:
        seen: set[str] = set()
        for member in tar.getmembers():
            clean = member.name.removeprefix("./")
            path = Path(clean)
            if not clean or path.is_absolute() or ".." in path.parts or clean in seen:
                raise StageError("r228 archive contains an unsafe or duplicate path")
            if not (member.isfile() or member.isdir()):
                raise StageError("r228 archive contains a link or special member")
            seen.add(clean)
            target = destination / path
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise StageError("r228 archive member is unreadable")
                with target.open("xb") as sink:
                    shutil.copyfileobj(source, sink)


def stage(*, source_root: Path, archive: Path, r233_runtime_source: Path, wheel: Path, output: Path) -> dict:
    if sha(archive) != R228_ARCHIVE:
        raise StageError("input is not the exact historical r228 archive")
    if output.exists():
        raise StageError("output already exists; refusing overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="r229-stage-", dir=output.parent) as raw:
        temporary = Path(raw) / "package"
        temporary.mkdir()
        safe_extract(archive, temporary)
        if sha(temporary / "model.pt") != MODEL or sha(temporary / "matchup_tree.json") != TREE:
            raise StageError("r228 frozen r195 identity drifted")
        overlay_hashes = overlay_r233_runtime(source=r233_runtime_source, destination=temporary)
        feature_path = temporary / "poke_bot/features.py"
        raise_packaged_action_cap(feature_path)
        overlay_hashes["poke_bot/features.py"] = sha(feature_path)
        for destination, source in OVERLAYS.items():
            source_path = source_root / source
            if not source_path.is_file():
                raise StageError(f"missing overlay source: {source}")
            shutil.copy2(source_path, temporary / destination)
            overlay_hashes[destination] = sha(temporary / destination)
        native_libraries = overlay_canonical_native_set(wheel=wheel, destination=temporary)
        manifest = {
            "schema": "poke_bot.alakazam_r228_vs_r195_no_mcts_fleet_bo1000_r253_package/v1",
            "status": "sealed_restarting_serial_mcts_leaf_bounded_evaluation_only",
            "owner_goal_revision": 253,
            "bo_lifecycle_revision": 233,
            "canonical_libcg_revision": 236,
            "superseded_two_lane_topology_revision": 239,
            "owner_handle_scoped_search_id_revision": 244,
            "owner_process_lane_recovery_revision": 249,
            "owner_serial_mcts_revision": 250,
            "owner_internal_leaf_boundary_revision": 252,
            "owner_restarting_serial_rollout_revision": 253,
            "native_simulator_worker_process_count": 1,
            "shared_tree_and_frozen_model_remain_in_parent": True,
            "native_search_calls_in_parent_worker_threads": False,
            "concurrent_libcg_search_calls_allowed": False,
            "complete_serial_retry_count_after_fault": 1,
            "failed_partial_tree_reuse_allowed": False,
            "consumed_worker_error_clears_in_flight_before_drain": True,
            "cleanup_error_counts_as_terminal_lane_response": True,
            "received_step_resolves_in_flight_before_parent_leaf_evaluation": True,
            "coordinator_post_error_wait_is_bounded": True,
            "search_seconds_per_attempt": 8.0,
            "exhausted_recovery_direct_fallback_is_degraded": True,
            "clean_full_game_preflight_max_exhausted_recovery_fallbacks": 0,
            "simulator_lane_count": 1,
            "internal_agent_start_arena_count": 1,
            "minimum_search_begin_call_count_per_searched_decision": 2,
            "search_begin_call_count_equals_completed_rollout_count": True,
            "required_handle_identity_count": 1,
            "required_handle_scoped_search_id_chain_count": 1,
            "required_handle_first_search_id_composite_count": 1,
            "handle_scoped_first_search_id_composite_state_array_field": (
                "handle_scoped_first_search_id_composite_states"
            ),
            "handle_scoped_first_search_id_composite_state_entry_exact_keys_in_order": [
                "lane_id",
                "handle_identity",
                "first_search_id",
            ],
            "search_begin_identity_scope": "arena_handle_plus_handle_local_search_id",
            "raw_search_id_global_uniqueness_required": False,
            "logical_frontier_leaf_count_per_frozen_model_batch": 1,
            "partial_frontier_batches_allowed": False,
            "serial_one_lane_continuation_required": False,
            "independent_exact_root_restart_per_rollout_required": True,
            "one_new_leaf_or_value_boundary_maximum_per_rollout": True,
            "rollout_search_id_chain_count_equals_rollout_count": True,
            "bounded_release_and_search_end_per_rollout_required": True,
            "minimum_completed_rollouts_for_mcts_action_authority": 2,
            "maximum_rollouts_per_decision": 1000,
            "one_shared_logical_mcts_tree_required": True,
            "process_parallel_node_evaluation_included": False,
            "parallel_node_evaluation_requires_clean_serial_full_game_receipt": True,
            "base_r228_archive_sha256": R228_ARCHIVE,
            "checkpoint_sha256": MODEL,
            "matchup_tree_sha256": TREE,
            "complete_ordered_action_ceiling": 65536,
            "internal_ordered_action_expansion_ceiling": 64,
            "every_explicit_chance_context_is_pre_random_value_boundary": True,
            "explicit_chance_probability_distribution_assumed": False,
            "deterministic_internal_fanout_over_64_is_value_only_boundary": True,
            "internal_boundary_representative_action_has_no_tree_authority": True,
            "internal_boundary_has_action_or_child_authority": False,
            "internal_value_boundary_telemetry_required": True,
            "kaggle_environments_version": "1.32.6",
            "canonical_libcg_wheel": {
                "filename": WHEEL_FILENAME,
                "sha256": WHEEL_SHA256,
                "size_bytes": WHEEL_SIZE_BYTES,
                "native_library_update_commit": NATIVE_LIBRARY_UPDATE_COMMIT,
            },
            "canonical_native_libraries": native_libraries,
            "r234_kaggle_broker_or_queue_lifecycle_included": False,
            "kaggle_search_policy_changes_included": False,
            "r249_bo_process_lane_boundary_included": True,
            "r250_serial_process_lane_topology_included": True,
            "r252_internal_leaf_boundary_included": True,
            "r253_restarting_serial_rollout_included": True,
            "continuous_single_trajectory_action_authority_allowed": False,
            "overlays": overlay_hashes,
            "package_payload_tree_sha256": tree_sha(temporary),
            "training_eligible": False,
        }
        (temporary / "r253_fleet_evaluation_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, indent=2) + "\n"
        )
        os.replace(temporary, output)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--r228-archive", type=Path, required=True)
    parser.add_argument("--r233-runtime-source", type=Path, required=True)
    parser.add_argument("--canonical-libcg-wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = stage(
        source_root=args.source_root.resolve(),
        archive=args.r228_archive.resolve(),
        r233_runtime_source=args.r233_runtime_source.resolve(),
        wheel=args.canonical_libcg_wheel.resolve(),
        output=args.output.resolve(),
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
