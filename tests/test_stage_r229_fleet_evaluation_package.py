from __future__ import annotations

import importlib.util
import io
import sys
import tarfile
import threading
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/stage_r229_fleet_evaluation_package.py"
spec = importlib.util.spec_from_file_location("r229_stage", SCRIPT)
assert spec and spec.loader
stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stage)


def test_game_runner_and_watchdog_are_sealed_package_overlays():
    assert stage.OVERLAYS == {
        "run_r229_process_watchdog.py": "scripts/run_r229_process_watchdog.py",
        "run_r229_mirror_game.py": "scripts/run_r229_mirror_game.py",
        "poke_bot/r249_process_search_lane.py": "poke_bot/r249_process_search_lane.py",
        "poke_bot/r250_recovering_serial_tree.py": "poke_bot/r250_recovering_serial_tree.py",
        "poke_bot/r252_search_leaf_boundary.py": "poke_bot/r252_search_leaf_boundary.py",
        "poke_bot/r253_restarting_serial_mcts.py": "poke_bot/r253_restarting_serial_mcts.py",
    }


def test_manifest_binds_public_handle_scoped_composite_shape():
    # The complete stage path is covered by the exact-package test below; this
    # focused assertion protects the canonical field names from accidental drift.
    source = Path(stage.__file__).read_text()
    assert '"handle_scoped_first_search_id_composite_state_array_field"' in source
    assert '"handle_scoped_first_search_id_composite_states"' in source
    assert (
        '"handle_scoped_first_search_id_composite_state_entry_exact_keys_in_order"'
        in source
    )


def test_payload_tree_digest_binds_paths_and_bytes(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "a").write_bytes(b"one")
    first = stage.tree_sha(root)
    assert stage.tree_sha(root) == first
    (root / "a").write_bytes(b"two")
    assert stage.tree_sha(root) != first
    (root / "a").write_bytes(b"one")
    (root / "b").write_bytes(b"")
    assert stage.tree_sha(root) != first


def test_safe_extract_rejects_links(tmp_path):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        row = tarfile.TarInfo("safe")
        payload = b"safe"
        row.size = len(payload)
        tar.addfile(row, io.BytesIO(payload))
        link = tarfile.TarInfo("linked")
        link.type = tarfile.SYMTYPE
        link.linkname = "safe"
        tar.addfile(link)
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(stage.StageError, match="link or special"):
        stage.safe_extract(archive, destination)


def test_action_cap_transform_is_exact_and_single_use(tmp_path):
    path = tmp_path / "features.py"
    path.write_text("before\nMAX_ACTION_COMBOS: int = 4096\nafter\n")
    stage.raise_packaged_action_cap(path)
    assert "MAX_ACTION_COMBOS: int = 65536" in path.read_text()
    with pytest.raises(stage.StageError, match="exact 4,096"):
        stage.raise_packaged_action_cap(path)


def _fake_wheel(monkeypatch, tmp_path: Path, *, omit: str | None = None) -> Path:
    wheel = tmp_path / stage.WHEEL_FILENAME
    payloads = {}
    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, row in stage.CANONICAL_LIBRARIES.items():
            if name == omit:
                continue
            payload = (name.encode() + b"-") * 3
            payloads[name] = payload
            archive.writestr(row["wheel_member"], payload)
    monkeypatch.setattr(stage, "WHEEL_SIZE_BYTES", wheel.stat().st_size)
    monkeypatch.setattr(stage, "WHEEL_SHA256", stage.sha(wheel))
    monkeypatch.setattr(
        stage,
        "CANONICAL_LIBRARIES",
        {
            name: {
                **row,
                "sha256": "sha256:" + __import__("hashlib").sha256(payloads.get(name, b"")).hexdigest(),
                "size_bytes": len(payloads.get(name, b"")),
            }
            for name, row in stage.CANONICAL_LIBRARIES.items()
        },
    )
    return wheel


def test_official_wheel_overlay_replaces_complete_native_set(monkeypatch, tmp_path):
    wheel = _fake_wheel(monkeypatch, tmp_path)
    package = tmp_path / "package"
    (package / "cg").mkdir(parents=True)
    for row in stage.CANONICAL_LIBRARIES.values():
        (package / row["package_relative_path"]).write_bytes(b"historical")
    receipt = stage.overlay_canonical_native_set(wheel=wheel, destination=package)
    assert set(receipt) == set(stage.CANONICAL_LIBRARIES)
    assert stage.verify_canonical_native_set(package) == receipt


def test_official_wheel_overlay_rejects_missing_sibling(monkeypatch, tmp_path):
    wheel = _fake_wheel(monkeypatch, tmp_path, omit="windows_x86_64")
    package = tmp_path / "package"
    (package / "cg").mkdir(parents=True)
    with pytest.raises(stage.StageError, match="missing or duplicated"):
        stage.overlay_canonical_native_set(wheel=wheel, destination=package)


def test_r233_runtime_source_is_checksum_pinned_and_actor_repaired(monkeypatch, tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    runtime = source / "poke_bot/r228_kaggle_async_runtime.py"
    queue = source / "poke_bot/r228_async_shared_tree_queue.py"
    runtime.parent.mkdir(parents=True)
    destination.mkdir()
    (source / "main.py").write_bytes(b'''r228 asynchronous eight-worker viability smoke
R228_ASYNC_EIGHT_WORKER_FULL_GAMEPLAY_SUCCESS
poke_bot.r228_async_eight_worker_kaggle_viability/v1
R228_ASYNC_EIGHT_WORKER_HARD_FAILURE''')
    queue.write_bytes(b'''"""Minimal persistent asynchronous eight-worker shared-tree search.

This module is deliberately small.  Eight thread-affine simulator arenas keep
their native search states alive while a coordinator repeatedly:

1. reserves a legal edge from one shared tree;
2. lets the owning simulator worker advance exactly one step;
3. microbatches whichever frontier states are ready;
4. backs those values into the same tree; and
5. immediately queues the next edge for those lanes.

It is a viability implementation, not a production-strength MCTS variant.
"""
LANES = 8
class AsyncEightWorkerError(RuntimeError):
    """The eight-worker search could not return a trustworthy decision."""
class PersistentAsyncEightWorkerMCTS:
    """Eight persistent simulator workers feeding one coordinator-owned tree."""
            if command is None:
                return
exactly eight search-input rows are required
decision deadline expired before eight arenas opened
asynchronous eight-worker decision failed
    unique_handle_count: int
    search_begin_calls: int
            unique_handle_count=len({worker.handle_identity for worker in self._workers}),
            search_begin_calls=LANES,
                coalesce_until = min(
                    float(deadline_monotonic), time.monotonic() + self._coalesce_seconds
                )
                step_rows: list[_WorkerResult] = []
                for row in ready:
                    if row.error is not None or row.kind != "step" or row.search_id is None:
                        raise AsyncEightWorkerError(
                            f"lane {row.lane_id} SearchStep failed: {row.error or row.kind}"
                        )
                    if row.lane_id not in in_flight:
                        raise AsyncEightWorkerError("received an untracked simulator result")
                    step_rows.append(row)
                packets = [self._make_packet(row.lane_id, row.observation) for row in step_rows]
                leaves = tuple(self._evaluate_batch(packets))
                if len(leaves) != len(step_rows):
                    raise AsyncEightWorkerError("GPU evaluator returned a partial microbatch")
                microbatches.append(len(leaves))
                for row, leaf in zip(step_rows, leaves):
                    leaf.validate()
                    context, edge = in_flight.pop(row.lane_id)
                    context.in_flight = False
                if smoke_min_depth_per_lane is not None and all(
                    len(context.action_path) >= int(smoke_min_depth_per_lane)
            while len(close_rows) < LANES:
                row = self._completions.get()
                if row.kind == "close":
                    close_rows[row.lane_id] = row
                else:
                    structural_error = structural_error or row.error or AsyncEightWorkerError(
                        f"unexpected result during cleanup: {row.kind}"
                    )''')
    runtime.write_bytes(b'''"""Stock-libcg eight-worker shared-tree action authority for the r228 smoke.

This is intentionally a small viability runtime.  One competition process owns
one frozen model and one shared search tree per decision.  Eight persistent
thread-affine ``AgentStart`` arenas advance independent simulator states and
feed ready leaves to the same model-backed coordinator queue.
"""
from .r228_async_shared_tree_queue import (
    AsyncEightWorkerError,
    DecodedLeaf,
    PersistentAsyncEightWorkerMCTS,
)

SCHEMA = "poke_bot.r228_async_eight_worker_kaggle_viability/v1"
        from .r225_stock_native_lane import (
            R225StockNativeSearchLane,
            prewarm_stock_cg,
        )
        def arena_factory(lane_id: int) -> Any:
            return R225StockNativeSearchLane(lane_id, lib=sim.lib, api_module=api)
        self._search = PersistentAsyncEightWorkerMCTS(
DECISION_PREFIX = "R228_ASYNC_EIGHT_WORKER_DECISION"
STOCK_LIBRARY_SHA256 = {
    "libcg.so": "ffd89bf923525a3e6feb5e6201e96a866c0f456895499ed5c4a566303caae67c",
    "libcg.dylib": "77bb978a8129b094452679e0daf0da69593afda7331685f4642c0d4a94d39d82",
    "libcg-arm64.so": "030b4728ce9fb9e90b75830b7cf7236f71859732a05ec4a377078eee0421bbe5",
    "cg.dll": "9ea2b0a751029689bff3ddccb5f29a98edd46961dad264490ed121ef704fb500",
}
search_inputs=tuple(dict(search_inputs) for _ in range(8))
        root_leaf = forward_leaf_batch(self.model, [root_packet])[0]
        root_priors = tuple(float(value) for value in root_leaf.priors)
        started = time.monotonic()
        try:
            receipt = self._search.run_decision(
                    "arena_count": receipt.arena_count,
                    "unique_handle_count": receipt.unique_handle_count,
                    "per_lane_depth": list(receipt.per_lane_depth),
                    "search_release_calls": receipt.search_release_calls,
            decoded[index] = DecodedLeaf(
                state_key=_state_key(lane_id=frontier.lane_id, raw=frontier.raw),
                value=float(leaf.value),
            combos = tuple(
                tuple(int(item) for item in action)
                for action in features.enumerate_action_combos(raw)
            )
            if not combos:
                raise R228GameplayError("nonterminal simulator leaf has no legal actions")
            boundary = _chance_boundary(raw)
            packets.append(self._leaf_packet(raw, combos=combos))
            pending.append((index, frontier, combos, boundary))
            "history_previous_actions": list(self.policy.previous_action_history),
        }
        except AsyncEightWorkerError as exc:
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
            mode = "clean_deadline_zero_backup_frozen_model_fallback"
        finally:
            self._decision = None
            "action_changed": selected != direct_action,
        }''')
    monkeypatch.setattr(stage, "R233_RUNTIME_COMPONENTS", {
        "main.py": stage.sha(source / "main.py"),
        "poke_bot/r228_async_shared_tree_queue.py": stage.sha(queue),
        "poke_bot/r228_kaggle_async_runtime.py": stage.sha(runtime),
    })
    hashes = stage.overlay_r233_runtime(source=source, destination=destination)
    repaired = (destination / "poke_bot/r228_kaggle_async_runtime.py").read_text()
    assert "d16244a3157fc55" in repaired
    assert 'actor = int(current.get("yourIndex", -1))' in repaired
    assert "range(1)" in repaired
    assert "per_lane_search_id_chains" in repaired
    assert "per_lane_handle_identities" in repaired
    assert "handle_scoped_first_search_id_composite_states" in repaired
    assert '"lane_id": lane_id' in repaired
    assert '"first_search_id": receipt.per_lane_search_id_chains[lane_id][0]' in repaired
    assert '"rollout_count": receipt.rollout_count' in repaired
    assert '"rollout_search_id_chains"' in repaired
    assert '"root_action_visit_counts"' in repaired
    assert "clean_deadline_insufficient_rollouts_frozen_model_fallback" in repaired
    assert "requested_simulator_lane_count" in repaired
    assert "R249ProcessSearchLane" in repaired
    assert "R250RecoveringSerialTree" in repaired
    assert "bounded_lane_recovery_exhausted_direct_fallback" in repaired
    assert '"lane_process_recovery"' in repaired
    assert "from .r252_search_leaf_boundary import classify_search_leaf" in repaired
    assert '"internal_ordered_action_expansion_ceiling": 64' in repaired
    assert '"internal_boundary_has_action_or_child_authority": False' in repaired
    assert "LANES = 1" in (destination / "poke_bot/r228_async_shared_tree_queue.py").read_text()
    assert "two-lane frontier batch was incomplete" not in (
        destination / "poke_bot/r228_async_shared_tree_queue.py"
    ).read_text()
    assert "close_owned_process" in (
        destination / "poke_bot/r228_async_shared_tree_queue.py"
    ).read_text()
    assert "second response that cannot exist" in (
        destination / "poke_bot/r228_async_shared_tree_queue.py"
    ).read_text()
    assert "terminal cleanup response" in (
        destination / "poke_bot/r228_async_shared_tree_queue.py"
    ).read_text()
    assert "R253_RESTARTING_SERIAL_FULL_GAMEPLAY_SUCCESS" in (
        destination / "main.py"
    ).read_text()
    assert hashes["main.py"] == stage.sha(destination / "main.py")
    assert hashes["main.py"] != stage.sha(source / "main.py")


def test_exact_pre_r234_baseline_produces_canonical_r253_bytes(tmp_path: Path):
    cache_root = Path("/Users/tsinzitari/.cache/pokebot/r229-r228-package")
    candidates = (
        cache_root / "package-ineligible-pre-r239-r228-59531249",
        cache_root / "package",
    )
    source = next(
        (
            candidate
            for candidate in candidates
            if all((candidate / relative).is_file() for relative in stage.R233_RUNTIME_COMPONENTS)
            and all(
                stage.sha(candidate / relative) == expected
                for relative, expected in stage.R233_RUNTIME_COMPONENTS.items()
            )
        ),
        candidates[0],
    )
    if not all((source / relative).is_file() for relative in stage.R233_RUNTIME_COMPONENTS):
        pytest.skip("canonical pre-r234 cache is unavailable on this host")
    if any(
        stage.sha(source / relative) != expected
        for relative, expected in stage.R233_RUNTIME_COMPONENTS.items()
    ):
        pytest.skip("local cache does not contain the canonical pre-r234 bytes")

    destination = tmp_path / "r253"
    destination.mkdir()
    hashes = stage.overlay_r233_runtime(source=source, destination=destination)

    assert hashes == {
        "main.py": "sha256:bc14cc472817f802bb357e48ecf7494bb4d66b308d6cfecf2cae87cd7531167a",
        "poke_bot/r228_async_shared_tree_queue.py": (
            "sha256:aee55da9e40a7f9345f1c899a1019e3fa1fe71bfb3d71c6bf5d85ca44b87f516"
        ),
        "poke_bot/r228_kaggle_async_runtime.py": (
            "sha256:26b745da2e99661b6bf84022319cf7b557443c9c0468e5c56761ab063aa72cdb"
        ),
    }
    for relative in hashes:
        compile((destination / relative).read_bytes(), relative, "exec")


@pytest.mark.parametrize("fault_phase", ["step", "evaluate", "cleanup"])
def test_exact_transformed_queue_contains_consumed_error_rows(
    tmp_path: Path, fault_phase: str
):
    cache_root = Path("/Users/tsinzitari/.cache/pokebot/r229-r228-package")
    candidates = (
        cache_root / "package-ineligible-pre-r239-r228-59531249",
        cache_root / "package",
    )
    source = next(
        (
            candidate
            for candidate in candidates
            if all(
                (candidate / relative).is_file()
                and stage.sha(candidate / relative) == expected
                for relative, expected in stage.R233_RUNTIME_COMPONENTS.items()
            )
        ),
        None,
    )
    if source is None:
        pytest.skip("canonical pre-r234 cache is unavailable on this host")

    destination = tmp_path / "r252"
    destination.mkdir()
    stage.overlay_r233_runtime(source=source, destination=destination)
    queue_path = destination / "poke_bot/r228_async_shared_tree_queue.py"
    module_name = f"r252_queue_fault_{fault_phase}"
    module_spec = importlib.util.spec_from_file_location(module_name, queue_path)
    assert module_spec and module_spec.loader
    queue_module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = queue_module
    try:
        module_spec.loader.exec_module(queue_module)
    finally:
        sys.modules.pop(module_name, None)

    class FaultArena:
        handle_identity = f"fault-{fault_phase}"

        def search_begin(self, observation, search_inputs, *, manual_coin=True):
            return SimpleNamespace(searchId=0, observation={"phase": "open"})

        def search_step(self, search_id, action):
            if fault_phase == "step":
                raise RuntimeError("injected native step fault")
            return SimpleNamespace(searchId=1, observation={"phase": "leaf"})

        def search_release(self, search_id):
            if fault_phase == "cleanup":
                raise RuntimeError("injected native cleanup fault")

        def search_end(self):
            return None

        def close(self):
            return None

    def evaluate(packets):
        if fault_phase == "evaluate":
            raise RuntimeError("injected leaf evaluate fault")
        return (
            queue_module.DecodedLeaf(
                state_key="leaf",
                value=0.0,
                legal_actions=(),
                priors=(),
                boundary=True,
                actor_seat=None,
            ),
        )

    core = queue_module.PersistentAsyncEightWorkerMCTS(
        arena_factory=lambda _lane_id: FaultArena(),
        make_packet=lambda _lane_id, observation: observation,
        evaluate_batch=evaluate,
    )
    outcome: dict[str, object] = {}

    def invoke() -> None:
        try:
            core.run_decision(
                root_observation={},
                search_inputs=({},),
                root_state_key="root",
                root_actions=((0,), (1,)),
                root_priors=(0.5, 0.5),
                root_seat=0,
                deadline_monotonic=time.monotonic() + 0.5,
            )
        except Exception as exc:  # noqa: BLE001 - capture fault-injection result
            outcome["error"] = exc

    worker = threading.Thread(target=invoke, daemon=True)
    worker.start()
    worker.join(timeout=2.0)
    assert not worker.is_alive(), f"{fault_phase} error left coordinator waiting forever"
    assert isinstance(outcome.get("error"), queue_module.AsyncEightWorkerError)
    assert fault_phase in str(outcome["error"]).lower()
    core.close()
