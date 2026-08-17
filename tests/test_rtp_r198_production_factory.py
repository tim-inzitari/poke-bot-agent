"""Focused invariant tests for the sealed r198 production evaluation factory.

These tests deliberately avoid parent checkpoints, a live evaluator, and any
production authority.  They exercise the factory-owned pure invariants that
must hold before the hermetic native-fixture E2E test runs.
"""

from __future__ import annotations

import json
import os
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot import rtp_r198_production_factory as factory


def test_r198_profiles_are_exact_and_non_authoritative() -> None:
    expected_flags = {
        "no_rtp": (False, False, False, False),
        "direct_bridge_recursive_disabled": (True, True, True, False),
        "recursive_rtp": (True, True, False, True),
    }
    for arm, flags in expected_flags.items():
        profile = factory.r198_runtime_profile_payload(arm)
        assert profile["evaluation_arm"] == arm
        assert profile["sizing_profile"] == "pure_rl_r197"
        assert profile["max_neural_passes"] == 256
        assert profile["max_action_combos"] == 1024
        assert profile["num_plan_candidates"] == 4
        assert profile["max_recursion_depth"] == 2
        assert (
            profile["recursive_turn_planner_enabled"],
            profile["direct_bridge_enabled"],
            profile["force_direct_bridge_only"],
            profile["recursive_repairs_enabled"],
        ) == flags
        assert profile["training_eligible"] is False
        assert profile["replay_eligible"] is False
        assert profile["serving_eligible"] is False
        assert profile["action_authority_enabled"] is False


def test_factory_authority_is_fixed_negative_scope() -> None:
    authority = factory.r198_factory_evaluation_authority_payload()
    assert authority == {
        "schema": factory.EVALUATION_AUTHORITY_SCHEMA,
        "status": "authorized_evaluation_only",
        "scope": "r198_factory_preparation_and_evaluation_only",
        "evaluation_only": True,
        "training_eligible": False,
        "replay_eligible": False,
        "serving_change_authorized": False,
        "selector_change_authorized": False,
        "action_authority_authorized": False,
        "kaggle_submission_authorized": False,
    }


def test_r197_completion_receipt_snapshot_contract_is_exact() -> None:
    assert factory.R197_COMPLETION_RECEIPT_FILENAME == "r197-completion-receipt.json"
    assert factory.R197_COMPLETION_RECEIPT_SCHEMA == (
        "poke_bot.alakazam_rtp_r197_shadow_candidate/v1"
    )
    assert factory.R197_COMPLETION_RECEIPT_SHA256 == (
        "sha256:b0c209257ed401bf9c5fe5a1ee17be1d1cdc01a1f9780e3e0d23ce8fa5f80737"
    )
    assert factory.R197_COMPLETION_RECEIPT_BYTES == 113_366


def test_matchup_adapter_registry_snapshot_contract_is_exact() -> None:
    assert factory.MATCHUP_ADAPTER_REGISTRY_RELATIVE == Path(
        "state/matchup_adapter_roster.json"
    )
    assert factory.MATCHUP_ADAPTER_REGISTRY_SHA256 == (
        "sha256:08322efe30c0f8b75d922aae8b882b4e78a20df03a63ed997ec8288165bfd1bc"
    )
    assert factory.MATCHUP_ADAPTER_REGISTRY_BYTES == 11_899
    assert factory.MATCHUP_ADAPTER_REGISTRY_CANONICAL_SHA256 == (
        "sha256:444c42c1235c19d3d95b10e80a12a84f35c9fb803967096736446eac1a5e225a"
    )


def test_final_closure_contract_keeps_runtime_copy_separate_from_provenance(
    tmp_path: Path,
) -> None:
    """A closure's copy is 0444; capability provenance may independently be 0555."""

    runtime_copy = _write_readonly(tmp_path / "cg" / "libcg.so", "closure DSO")
    identity = {
        "path": str(runtime_copy),
        "sha256": factory._sha256_file(runtime_copy),
        "bytes": runtime_copy.stat().st_size,
    }
    assert factory._closure_provenance_engine_identity(identity)["mode"] == 0o444
    runtime_copy.chmod(0o555)
    with pytest.raises(factory.R198ProductionFactoryError, match="must use mode 0o444"):
        factory._closure_provenance_engine_identity(identity)
    assert factory.EVALUATION_CG_CLOSURE_FILENAME == "eval-cg-closure.json"
    assert factory.R198_EVAL_CG_CLOSURE_RECEIPT_BYTES == 2399
    assert factory.R198_EVAL_CG_CLOSURE_RECEIPT_SHA256 == (
        "sha256:419ad46a9b31b9fdc040b851b553108b1bd038b68acadccb4dc9c38bfd35bbe0"
    )


def _cg_tree_identity_with_first_byte_count(
    tmp_path: Path, byte_count: object = 0, *, omit: bool = False
) -> dict[str, object]:
    schema = "test.r198_eval_cg_tree/v1"
    files: list[dict[str, object]] = []
    for index, relative_path in enumerate(factory._EVALUATION_CG_CLOSURE_TREE_PATHS):
        contents = b"" if index == 0 else relative_path.encode("utf-8")
        row: dict[str, object] = {
            "relative_path": relative_path,
            "sha256": factory._sha256_bytes(contents),
            "bytes": len(contents),
        }
        if index == 0:
            if omit:
                row.pop("bytes")
            else:
                row["bytes"] = byte_count
        files.append(row)
    material = {"schema": schema, "file_count": len(files), "files": files}
    path = _write_readonly_json(
        tmp_path / "cg-tree.json",
        {**material, "tree_sha256": factory._canonical_digest(material)},
    )
    return {**_file_identity(path), "schema": schema}


def test_cg_tree_digest_accepts_checksum_bound_zero_byte_package_marker(
    tmp_path: Path,
) -> None:
    identity = _cg_tree_identity_with_first_byte_count(tmp_path)
    tree_sha256, files = factory._verified_cg_tree_digest(
        identity,
        label="test CG tree",
        schema=str(identity["schema"]),
    )
    assert tree_sha256 == json.loads(
        Path(str(identity["path"])).read_text(encoding="utf-8")
    )["tree_sha256"]
    assert files[0] == {
        "relative_path": "__init__.py",
        "sha256": factory._sha256_bytes(b""),
        "bytes": 0,
    }


@pytest.mark.parametrize("case", ("negative", "boolean", "missing"))
def test_cg_tree_digest_rejects_invalid_byte_counts(
    tmp_path: Path, case: str
) -> None:
    identity = _cg_tree_identity_with_first_byte_count(
        tmp_path,
        -1 if case == "negative" else True,
        omit=case == "missing",
    )
    with pytest.raises(factory.R198ProductionFactoryError, match=r"files\[0\]\.bytes is invalid"):
        factory._verified_cg_tree_digest(
            identity,
            label="test CG tree",
            schema=str(identity["schema"]),
        )


def test_manifest_closure_binds_exact_snapshot_runtime_library(tmp_path: Path) -> None:
    source = tmp_path / "sealed-source"
    runtime = source / "kaggle" / "input" / "rtp-eval-cg"
    receipt_path = _write_readonly(source / "eval-cg-closure.json", "receipt")
    library_path = _write_readonly(runtime / "cg" / "libcg.so", "runtime DSO")
    receipt = {
        "path": str(receipt_path),
        "sha256": factory._sha256_file(receipt_path),
        "bytes": receipt_path.stat().st_size,
    }
    runtime_library = {
        "path": str(library_path),
        "sha256": factory._sha256_file(library_path),
        "bytes": library_path.stat().st_size,
    }
    cg = factory._CGAssets(
        runtime_root=runtime,
        closure_manifest=receipt,
        library=runtime_library,
        closure_payload={},
        closure_evidence={},
    )
    manifest = {
        "evaluation_cg_closure": {
            "receipt": receipt,
            "runtime_library": runtime_library,
        }
    }
    factory._verify_manifest_evaluation_cg_closure(
        manifest, cg=cg, source_root=source
    )
    manifest["evaluation_cg_closure"]["unexpected_alias"] = runtime_library
    with pytest.raises(
        factory.R198ProductionFactoryError,
        match="exact base pair or full evaluator-normalized record",
    ):
        factory._verify_manifest_evaluation_cg_closure(
            manifest, cg=cg, source_root=source
        )


class _FakeRouter:
    def __init__(self) -> None:
        self.public_state = {"route": 0, "observations": ["initial"]}


class _FakeExecutor:
    def __init__(self) -> None:
        self.active_program = {"steps": ["a"]}
        self.cursor = 0
        self.repairs_used = 0
        self.steps_executed = 1
        self.repair_fn = lambda: None


class _FakeBridge:
    def __init__(self) -> None:
        self.memory = {"encoded": [1, 2]}
        self.active_turn_key = (5, 7)
        self.active_turn_complexity_intent = {"would_recurse": True}
        self.last_diagnostics = {"mode": "recursive_plan"}
        self.executor = _FakeExecutor()


class _FakeCandidate:
    def __init__(self) -> None:
        self.board_history = [{"board": "initial"}]
        self.previous_action_history = [{"action": "initial"}]
        self._previous_action_token = {"token": "initial"}
        self._kv_cache = {"layers": [1, 2]}
        self._matchup_adapter_shadow_router = _FakeRouter()
        self._rtp_bridge = _FakeBridge()
        self.last_rtp_diagnostics = {"mode": "recursive_plan"}
        self.rng = random.Random(37)

    def matchup_adapter_shadow_snapshot(self) -> dict[str, object]:
        return {
            "public_state": dict(self._matchup_adapter_shadow_router.public_state),
        }


@pytest.fixture
def fake_candidate() -> _FakeCandidate:
    original_global_rng = random.getstate()
    random.seed(9182)
    try:
        yield _FakeCandidate()
    finally:
        random.setstate(original_global_rng)


def test_complexity_probe_digest_is_stable_without_mutation(
    fake_candidate: _FakeCandidate,
) -> None:
    before = factory._candidate_probe_state_fingerprint(fake_candidate)
    assert factory._require_unchanged_complexity_probe_state(fake_candidate, before) == before


def test_logical_over_cap_fingerprint_uses_only_causal_policy_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bridge identities/audit counters cannot create false parity failures."""

    from poke_bot import features

    class GameRouter:
        def __init__(self, *, route: int = 0, offset: int = 0) -> None:
            self.route = route
            self.offset = offset

        def observe(self, observation: dict[str, object]) -> None:
            self.route = int(observation["logical_route"]) + self.offset

        def fork(self) -> "GameRouter":
            return type(self)(route=self.route, offset=self.offset)

    class Router:
        def __init__(self, *, game_router: GameRouter | None = None, audit: object | None = None) -> None:
            self.game_router = game_router or GameRouter()
            self.audit = audit if audit is not None else {"observations": 7, "events": ["live"]}
            self.tree = SimpleNamespace(digest="sha256:" + "d" * 64)

        @property
        def candidate_model_route(self) -> int:
            return self.game_router.route

        def fork(self) -> "Router":
            # Match the production shadow router: game state forks, audit is
            # intentionally shared and must remain untouched by fingerprinting.
            return type(self)(game_router=self.game_router.fork(), audit=self.audit)

    class Candidate:
        def __init__(self) -> None:
            self.board_history = [{"board": "before"}]
            self.previous_action_history = [{"action": "before"}]
            self._previous_action_token = {"token": "before"}
            self._kv_cache = None
            self._matchup_adapter_shadow_router = Router()
            self._history_context_limit = lambda: 4
            self.sample_actions = False
            self.rng = random.Random(19)
            self.action_temperature = 1.0
            self.deck = [1, 2, 3]
            self.model = SimpleNamespace(d_model=96, max_context=320)
            self.checkpoint_digest = "sha256:" + "e" * 64
            # Deliberately divergent non-causal bridge/diagnostic object
            # identities must not enter the logical factorized-policy key.
            self._rtp_bridge = SimpleNamespace(executor=object(), memory=object())
            self.last_rtp_diagnostics = {"mode": "fallback", "opaque": object()}

    board = features.SparseVector()
    board.word_start()
    board.add(3, 1.0)
    monkeypatch.setattr(features, "build_board_tokens", lambda *_args: board)
    observation: dict[str, object] = {
        "logical_route": 2,
        "select": {"minCount": 1, "maxCount": 5, "option": [{}, {}, {}]},
    }
    left = Candidate()
    right = Candidate()
    # Different implementation-only identities and diagnostics are expected
    # across A/B/C and must not affect a causal parity key.
    right._rtp_bridge = SimpleNamespace(executor=object(), memory=object())
    right.last_rtp_diagnostics = {"mode": "recursive_plan", "opaque": object()}
    audit_before = dict(left._matchup_adapter_shadow_router.audit)  # type: ignore[arg-type]
    left_digest = factory._candidate_logical_policy_input_fingerprint(left, observation)
    right_digest = factory._candidate_logical_policy_input_fingerprint(right, observation)
    assert left_digest == right_digest
    assert left._matchup_adapter_shadow_router.audit == audit_before

    right.board_history.append({"board": "different-causal-history"})
    assert factory._candidate_logical_policy_input_fingerprint(right, observation) != left_digest
    right.board_history.pop()
    right._kv_cache = {"cache": "different-causal-content"}
    assert factory._candidate_logical_policy_input_fingerprint(right, observation) != left_digest
    right._kv_cache = None
    right._previous_action_token = {"token": "different-causal-token"}
    assert factory._candidate_logical_policy_input_fingerprint(right, observation) != left_digest
    right._previous_action_token = {"token": "before"}
    right._matchup_adapter_shadow_router.game_router.offset = 1
    assert factory._candidate_logical_policy_input_fingerprint(right, observation) != left_digest


def test_complexity_probe_short_circuits_external_turn_order_before_sidecar(
    fake_candidate: _FakeCandidate,
) -> None:
    """The IsFirst control must not require an RTP bridge/model sidecar."""

    probe = object.__new__(factory._ComplexityIntentProbe)
    probe._candidate = fake_candidate
    before = factory._candidate_probe_state_fingerprint(fake_candidate)

    result = probe(
        {
            "select": {
                "context": 41,
                "minCount": 1,
                "maxCount": 1,
                "option": [{"type": 1}, {"type": 2}],
            }
        }
    )

    assert result == {
        "intended_complex": False,
        "planner_reason": "forced_go_first_contract",
    }
    assert factory._candidate_probe_state_fingerprint(fake_candidate) == before
    # ``_bridge`` is intentionally absent: any encoder/router/sidecar use
    # before the forced-control short circuit would have raised AttributeError.
    assert not hasattr(probe, "_bridge")


def test_complexity_probe_rejects_unpreclassified_over_cap(
    fake_candidate: _FakeCandidate,
) -> None:
    """A direct probe call cannot relabel an unassessed space as simple."""

    probe = object.__new__(factory._ComplexityIntentProbe)
    probe._candidate = fake_candidate
    before = factory._candidate_probe_state_fingerprint(fake_candidate)
    with pytest.raises(
        factory.R198ProductionFactoryError,
        match="runner must preclassify over-cap factorized selection",
    ):
        probe(
            {
                "select": {
                    "minCount": 1,
                    "maxCount": 5,
                    "option": [{"type": 2} for _ in range(9)],
                }
            }
        )
    assert factory._candidate_probe_state_fingerprint(fake_candidate) == before


@pytest.mark.parametrize(
    "field",
    (
        "board_history",
        "previous_action_history",
        "previous_action_token",
        "kv_cache_identity_and_content",
        "router",
        "bridge_memory",
        "bridge_turn_key_and_intent",
        "bridge_executor",
        "bridge_diagnostics",
        "candidate_diagnostics",
        "candidate_rng",
        "global_rng",
    ),
)
def test_complexity_probe_guard_covers_candidate_causal_state(
    fake_candidate: _FakeCandidate, field: str
) -> None:
    before = factory._candidate_probe_state_fingerprint(fake_candidate)
    bridge = fake_candidate._rtp_bridge
    if field == "board_history":
        fake_candidate.board_history.append({"board": "after"})
    elif field == "previous_action_history":
        fake_candidate.previous_action_history.append({"action": "after"})
    elif field == "previous_action_token":
        fake_candidate._previous_action_token = {"token": "after"}
    elif field == "kv_cache_identity_and_content":
        fake_candidate._kv_cache = {"layers": [1, 2, 3]}
    elif field == "router":
        fake_candidate._matchup_adapter_shadow_router.public_state["route"] = 1
    elif field == "bridge_memory":
        bridge.memory = {"encoded": [9]}
    elif field == "bridge_turn_key_and_intent":
        bridge.active_turn_key = (9, 9)
        bridge.active_turn_complexity_intent = {"would_recurse": False}
    elif field == "bridge_executor":
        bridge.executor.cursor = 1
        bridge.executor.active_program = {"steps": ["changed"]}
    elif field == "bridge_diagnostics":
        bridge.last_diagnostics = {"mode": "fallback"}
    elif field == "candidate_diagnostics":
        fake_candidate.last_rtp_diagnostics = {"mode": "fallback"}
    elif field == "candidate_rng":
        fake_candidate.rng.random()
    elif field == "global_rng":
        random.random()
    else:  # pragma: no cover - pytest parametrization is literal above.
        raise AssertionError(field)
    with pytest.raises(factory.R198ProductionFactoryError, match="mutated candidate causal state"):
        factory._require_unchanged_complexity_probe_state(fake_candidate, before)


def _write_readonly(path: Path, content: str, mode: int = 0o444) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(mode)
    return path


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": factory._sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _write_readonly_json(path: Path, payload: dict[str, object]) -> Path:
    return _write_readonly(path, json.dumps(payload, sort_keys=True) + "\n")


def _freeze_tree(root: Path) -> None:
    for current_raw, _directories, files in os.walk(root, topdown=False):
        current = Path(current_raw)
        for name in files:
            (current / name).chmod(0o444)
        current.chmod(0o555)


def _production_shaped_base_spec_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    deck_entry_path: str = "deck.csv",
) -> tuple[Path, dict[str, object], dict[str, dict[str, object]]]:
    """Build immutable snapshot bytes around the base-spec package boundary.

    The candidate/CG/pairing internals are patched only after the builder has
    independently validated their file identities.  Official package trees
    remain physical and are checked by the production code without a mock.
    """

    source = tmp_path / "source-snapshot"
    checkout_root = Path(factory.__file__).resolve().parents[1]
    factory_module = _write_readonly(
        source / "poke_bot" / "rtp_r198_production_factory.py", "# snapshot factory\n"
    )
    registry = _write_readonly(
        source / factory.MATCHUP_ADAPTER_REGISTRY_RELATIVE,
        (checkout_root / factory.MATCHUP_ADAPTER_REGISTRY_RELATIVE).read_text(
            encoding="utf-8"
        ),
    )
    registry_identity = _file_identity(registry)
    assert registry_identity["sha256"] == factory.MATCHUP_ADAPTER_REGISTRY_SHA256
    assert registry_identity["bytes"] == factory.MATCHUP_ADAPTER_REGISTRY_BYTES
    candidate_root = source / "evaluation-artifacts" / "r197-candidate"
    parent = _write_readonly(candidate_root / "parent.pt", "parent")
    sidecar = _write_readonly(candidate_root / "sidecar.pt", "sidecar")
    sidecar_receipt = _write_readonly(candidate_root / "sidecar-receipt.json", "receipt")
    completion_receipt = _write_readonly_json(
        candidate_root / factory.R197_COMPLETION_RECEIPT_FILENAME,
        {
            "schema": factory.R197_COMPLETION_RECEIPT_SCHEMA,
            "status": "completed_shadow_only",
            "candidate_id": factory.R198_CANDIDATE_ID,
            "candidate_contract_sha256": factory.R198_CANDIDATE_CONTRACT_SHA256,
            "authority": {
                "shadow_only": True,
                "serving_eligible": False,
                "action_authority_enabled": False,
                "selector_authority": False,
                "live_checkpoint_publication": False,
                "submission_eligible": False,
            },
        },
    )
    candidate_deck = _write_readonly(candidate_root / "deck.csv", "candidate deck\n")
    matchup_tree = _write_readonly(candidate_root / "matchup-tree.json", "{}\n")
    candidate_artifacts = {
        "parent_checkpoint": _file_identity(parent),
        "sidecar": _file_identity(sidecar),
        "sidecar_receipt": _file_identity(sidecar_receipt),
        "completion_receipt": _file_identity(completion_receipt),
        "deck": _file_identity(candidate_deck),
        "matchup_tree": _file_identity(matchup_tree),
    }
    _write_readonly_json(
        candidate_root / "manifest.json",
        {
            "schema": factory.CANDIDATE_SNAPSHOT_SCHEMA,
            "status": "sealed",
            "no_symlinks": True,
            "all_paths_read_only": True,
            "candidate_id": factory.R198_CANDIDATE_ID,
            "candidate_contract_sha256": factory.R198_CANDIDATE_CONTRACT_SHA256,
            "package_root": str(candidate_root),
            "artifacts": candidate_artifacts,
        },
    )

    official_decks: dict[str, dict[str, object]] = {}
    for opponent_id, content_digest in factory.OFFICIAL_CONTROL_DIGESTS.items():
        package_root = source / "baselines" / "official" / opponent_id
        main_py = _write_readonly(package_root / "main.py", "def agent(_observation):\n    return [0]\n")
        deck = _write_readonly(package_root / "deck.csv", f"{opponent_id} deck\n")
        deck_identity = _file_identity(deck)
        entries = sorted(
            [
                {
                    "path": deck_entry_path,
                    "sha256": deck_identity["sha256"],
                    "bytes": deck_identity["bytes"],
                },
                {
                    "path": "main.py",
                    "sha256": _file_identity(main_py)["sha256"],
                    "bytes": _file_identity(main_py)["bytes"],
                },
            ],
            key=lambda row: str(row["path"]),
        )
        _write_readonly_json(
            source
            / "evaluation-artifacts"
            / "official-control-manifests"
            / f"{opponent_id}.json",
            {
                "schema": factory.PACKAGE_SNAPSHOT_SCHEMA,
                "status": "sealed",
                "opponent_id": opponent_id,
                "content_digest": content_digest,
                "no_symlinks": True,
                "all_paths_read_only": True,
                "package_root": str(package_root),
                "entries": entries,
                "tree_entries_sha256": factory._canonical_digest(entries),
                "deck_sha256": deck_identity["sha256"],
                "deck_order_sha256": deck_identity["sha256"],
            },
        )
        official_decks[opponent_id] = deck_identity

    cg_root = source / "kaggle" / "input" / "rtp-eval-cg"
    closure = _write_readonly(cg_root / factory.EVALUATION_CG_CLOSURE_FILENAME, "closure")
    library = _write_readonly(cg_root / "cg" / "libcg.so", "library")
    source_tree_sha256 = factory._sha256_bytes(b"source tree")
    _write_readonly_json(
        source / factory.SOURCE_SNAPSHOT_MANIFEST_NAME,
        {
            "schema": factory.SOURCE_SNAPSHOT_SCHEMA,
            "source_tree_sha256": source_tree_sha256,
            "required_relative_files": [
                factory.MATCHUP_ADAPTER_REGISTRY_RELATIVE.as_posix()
            ],
            "source_entries": [
                {
                    "path": factory.MATCHUP_ADAPTER_REGISTRY_RELATIVE.as_posix(),
                    "type": "file",
                    "mode": 0o444,
                    "size": registry_identity["bytes"],
                    "sha256": registry_identity["sha256"],
                }
            ],
            "eval_cg_closure": {
                "closure_manifest": _file_identity(closure),
                "library": _file_identity(library),
            },
        },
    )

    private = tmp_path / "private-pairing"
    engine = _write_readonly(private / "engine.so", "engine", mode=0o555)
    pairing_source = _write_readonly(private / "source.json", "source")
    patch = _write_readonly(private / "patch.cpp", "patch")
    build = _write_readonly(private / "build.json", "build")
    capability = _write_readonly_json(
        private / "capability.json",
        {
            "schema": factory.PAIRING_CAPABILITY_SCHEMA,
            "status": "available",
            "engine_artifact": _file_identity(engine),
            "source_artifact": _file_identity(pairing_source),
            "patch_artifact": _file_identity(patch),
            "build_artifact": _file_identity(build),
        },
    )
    _freeze_tree(source)
    monkeypatch.setattr(factory, "__file__", str(factory_module))
    monkeypatch.setattr(
        factory,
        "_candidate_assets",
        lambda *_args, **_kwargs: SimpleNamespace(
            parent_checkpoint=candidate_artifacts["parent_checkpoint"],
            deck=candidate_artifacts["deck"],
            matchup_tree=candidate_artifacts["matchup_tree"],
            completion_receipt={
                **candidate_artifacts["completion_receipt"],
                "mode": 0o444,
            },
        ),
    )
    monkeypatch.setattr(factory, "_cg_assets", lambda *_args, **_kwargs: SimpleNamespace())
    monkeypatch.setattr(
        factory, "_pairing_artifacts", lambda *_args, **_kwargs: (SimpleNamespace(), {})
    )
    monkeypatch.setattr(factory, "_crosscheck_cg_against_pairing", lambda *_args: None)
    base_spec = factory.build_r198_evaluator_base_spec(
        source_snapshot_root=source,
        source_tree_sha256=source_tree_sha256,
        pairing_capability=_file_identity(capability),
    )
    return source, base_spec, official_decks


def test_base_spec_resolves_relative_official_decks_inside_snapshot_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, base_spec, official_decks = _production_shaped_base_spec_fixture(
        tmp_path, monkeypatch
    )
    opponents = {row["id"]: row for row in base_spec["opponents"]}
    assert set(opponents) == set(factory.OFFICIAL_CONTROL_DIGESTS)
    for opponent_id, expected_deck in official_decks.items():
        opponent = opponents[opponent_id]
        expected_root = source / "baselines" / "official" / opponent_id
        assert opponent["package_root"] == str(expected_root)
        for key, expected in expected_deck.items():
            assert opponent["deck"][key] == expected
        assert opponent["deck"]["path"] == str(expected_root / "deck.csv")

        runtime_package = factory._official_package(
            {"opponents": list(base_spec["opponents"])}, opponent_id, source
        )
        assert runtime_package.package_root == expected_root
        for key, expected in expected_deck.items():
            assert runtime_package.deck[key] == expected


def test_base_spec_binds_snapshot_local_completion_receipt_as_sixth_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, base_spec, _official_decks = _production_shaped_base_spec_fixture(
        tmp_path, monkeypatch
    )
    artifacts = base_spec["production_factory"]["artifacts"]
    assert set(artifacts) == {
        "parent_checkpoint",
        "sidecar",
        "sidecar_receipt",
        "completion_receipt",
        "deck",
        "matchup_tree",
    }
    completion = artifacts["completion_receipt"]
    assert completion["path"] == str(
        source
        / "evaluation-artifacts"
        / "r197-candidate"
        / factory.R197_COMPLETION_RECEIPT_FILENAME
    )
    assert completion["mode"] == 0o444
    assert base_spec["shared_artifacts"]["r197_completion_receipt"] == completion


def test_base_spec_binds_snapshot_inventory_matchup_adapter_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, base_spec, _official_decks = _production_shaped_base_spec_fixture(
        tmp_path, monkeypatch
    )
    registry = base_spec["production_factory"]["matchup_adapter_registry"]
    assert registry == {
        "path": str(source / factory.MATCHUP_ADAPTER_REGISTRY_RELATIVE),
        "sha256": factory.MATCHUP_ADAPTER_REGISTRY_SHA256,
        "bytes": factory.MATCHUP_ADAPTER_REGISTRY_BYTES,
        "mode": 0o444,
    }


def test_runtime_registry_loader_and_policy_router_are_snapshot_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "sealed-source"
    checkout_root = Path(factory.__file__).resolve().parents[1]
    registry_path = _write_readonly(
        source / factory.MATCHUP_ADAPTER_REGISTRY_RELATIVE,
        (checkout_root / factory.MATCHUP_ADAPTER_REGISTRY_RELATIVE).read_text(
            encoding="utf-8"
        ),
    )
    module_path = _write_readonly(
        source / "poke_bot" / "matchup_adapters_v6.py", "# sealed module\n"
    )
    _freeze_tree(source)
    registry_identity = {
        **_file_identity(registry_path),
        "mode": 0o444,
    }
    payload = json.loads(registry_path.read_text(encoding="utf-8"))

    fake_module = types.ModuleType("poke_bot.matchup_adapters_v6")
    fake_module.__file__ = str(module_path)
    fake_module.DEFAULT_REGISTRY_PATH = registry_path

    def load_slot_registry(path: Path | str = registry_path) -> dict[str, object]:
        assert Path(path) == registry_path
        return dict(payload)

    fake_module.load_slot_registry = load_slot_registry
    fake_module.registry_digest = factory._canonical_digest
    monkeypatch.setitem(sys.modules, "poke_bot.matchup_adapters_v6", fake_module)
    import poke_bot

    monkeypatch.setattr(poke_bot, "matchup_adapters_v6", fake_module, raising=False)
    inputs = SimpleNamespace(
        candidate=SimpleNamespace(source_root=source),
        matchup_adapter_registry=registry_identity,
        matchup_adapter_registry_digest=(
            factory.MATCHUP_ADAPTER_REGISTRY_CANONICAL_SHA256
        ),
    )
    observed = factory._require_snapshot_matchup_adapter_registry(inputs)
    assert observed == factory.MATCHUP_ADAPTER_REGISTRY_CANONICAL_SHA256

    model_config = {
        "format": "poke-bot-matchup-adapter-bank-v6",
        "slot_registry_digest": observed,
        "slot_registry": payload,
    }
    model = SimpleNamespace(
        matchup_adapter_bank=SimpleNamespace(
            config_dict=lambda: dict(model_config)
        )
    )
    factory._require_model_matchup_adapter_registry(model, inputs)
    model_config["slot_registry_digest"] = factory._sha256_bytes(b"other roster")
    with pytest.raises(
        factory.R198ProductionFactoryError,
        match="registry digest differs from the snapshot roster",
    ):
        factory._require_model_matchup_adapter_registry(model, inputs)
    model_config["slot_registry_digest"] = observed
    model_config["slot_registry"] = {**payload, "revision": -1}
    with pytest.raises(
        factory.R198ProductionFactoryError,
        match="serialized registry differs from the snapshot roster",
    ):
        factory._require_model_matchup_adapter_registry(model, inputs)

    candidate = SimpleNamespace(
        _matchup_adapter_shadow_router=SimpleNamespace(
            tree=SimpleNamespace(slot_registry_digest=observed)
        )
    )
    factory._require_policy_router_registry(candidate, observed)
    candidate._matchup_adapter_shadow_router.tree.slot_registry_digest = (
        factory._sha256_bytes(b"ambient registry")
    )
    with pytest.raises(
        factory.R198ProductionFactoryError,
        match="did not retain the sealed matchup adapter registry",
    ):
        factory._require_policy_router_registry(candidate, observed)

    ambient = _write_readonly(tmp_path / "ambient" / "matchup_adapter_roster.json", "{}\n")
    load_slot_registry.__defaults__ = (ambient,)
    with pytest.raises(
        factory.R198ProductionFactoryError,
        match="default is not snapshot-local",
    ):
        factory._require_snapshot_matchup_adapter_registry(inputs)


def test_all_three_runtime_profiles_bind_exact_seven_shared_artifacts(
    tmp_path: Path,
) -> None:
    """Exercise the production factory output against the real runner contract."""

    from poke_bot import rtp_three_arm_evaluation_runner as runner

    source = tmp_path / "sealed-source"
    candidate_root = source / "evaluation-artifacts" / "r197-candidate"
    parent = _write_readonly(candidate_root / "parent.pt", "parent\n")
    deck = _write_readonly(candidate_root / "deck.csv", "deck\n")
    matchup_tree = _write_readonly(candidate_root / "matchup-tree.json", "{}\n")
    completion = _write_readonly(candidate_root / "r197-completion-receipt.json", "{}\n")
    checkout_root = Path(factory.__file__).resolve().parents[1]
    registry = _write_readonly(
        source / "ops" / "research_control_registry_v1.json",
        (checkout_root / "ops" / "research_control_registry_v1.json").read_text(
            encoding="utf-8"
        ),
    )
    assert factory._sha256_file(registry) == factory.RESEARCH_CONTROL_REGISTRY_SHA256
    assert registry.stat().st_size == factory.RESEARCH_CONTROL_REGISTRY_BYTES
    _freeze_tree(source)

    evaluation_root = tmp_path / "r198-evaluation-inputs-production-shaped"
    fixture_root = evaluation_root / "preflight-fixture-inputs"
    fixture_root.mkdir(parents=True)
    cohort = _write_readonly_json(
        evaluation_root / "cohort" / "evaluation-only-cohort.json",
        {
            "schema": "poke_bot.recursive_turn_planner.r197_evaluation_only_cohort/v1",
            "status": "frozen",
        },
    )
    preflight = _write_readonly_json(
        evaluation_root / "preflight" / "planner-pass-preflight.json",
        {"schema": factory.PREFLIGHT_RECEIPT_SCHEMA, "status": "passed"},
    )
    _freeze_tree(evaluation_root)

    candidate = SimpleNamespace(
        source_root=source,
        parent_checkpoint=_file_identity(parent),
        deck=_file_identity(deck),
        matchup_tree=_file_identity(matchup_tree),
        completion_receipt=_file_identity(completion),
    )
    inputs = SimpleNamespace(
        candidate=candidate,
        evaluation_inputs_root=fixture_root,
    )
    shared_artifacts = {
        "deck": _file_identity(deck),
        "evaluation_only_cohort": _file_identity(cohort),
        "matchup_tree": _file_identity(matchup_tree),
        "parent_checkpoint": _file_identity(parent),
        "planner_preflight_receipt": _file_identity(preflight),
        "r197_completion_receipt": _file_identity(completion),
        "research_control_registry": _file_identity(registry),
    }
    shared = factory._shared_assets(
        {"shared_artifacts": shared_artifacts}, inputs
    )
    shared_runtime = factory._runtime_shared_artifact_sha256s(shared)
    assert shared_runtime == {
        f"{name}_sha256": shared_artifacts[name]["sha256"]
        for name in factory.R198_SHARED_ARTIFACT_NAMES
    }

    runtime_artifact = _write_readonly(
        tmp_path / "sealed-arm-runtime" / "factory.py", "# runtime\n"
    )
    sidecar = _write_readonly(
        tmp_path / "sealed-arm-runtime" / "sidecar.pt", "sidecar\n"
    )
    arm_specs: dict[str, dict[str, object]] = {}
    for arm in factory.CANONICAL_ARMS:
        profile_payload = factory.r198_runtime_profile_payload(arm)
        profile = _write_readonly_json(
            tmp_path / "sealed-arm-runtime" / f"{arm}.json",
            profile_payload,
        )
        arm_specs[arm] = {
            "runtime_artifact": _file_identity(runtime_artifact),
            "runtime_profile": _file_identity(profile),
            "rtp_sidecar": None if arm == "no_rtp" else _file_identity(sidecar),
        }
    manifest = {
        "arms": arm_specs,
        "shared_artifacts": shared_artifacts,
    }
    runtime_by_arm: dict[str, dict[str, object]] = {}
    for arm in factory.CANONICAL_ARMS:
        profile_payload = factory.r198_runtime_profile_payload(arm)
        runtime = {
            "arm": arm,
            "runtime_artifact_sha256": _file_identity(runtime_artifact)["sha256"],
            "runtime_profile_sha256": arm_specs[arm]["runtime_profile"]["sha256"],  # type: ignore[index]
            "action_attached_rtp_sidecar_sha256": (
                None if arm == "no_rtp" else _file_identity(sidecar)["sha256"]
            ),
            "complexity_probe_sidecar_sha256": _file_identity(sidecar)["sha256"],
            "complexity_probe_sidecar_instrumentation_only": True,
            "complexity_probe_latency_excluded": True,
            "rtp_action_attachment_enabled": arm != "no_rtp",
            "rtp_action_authority_enabled": False,
            **shared_runtime,
            **{
                key: profile_payload[key]
                for key in (
                    "recursive_turn_planner_enabled",
                    "direct_bridge_enabled",
                    "force_direct_bridge_only",
                    "max_neural_passes",
                    "max_action_combos",
                )
            },
        }
        runtime_by_arm[arm] = runtime
        checked = runner._runtime_profile_contract(manifest, arm, runtime)
        for key, expected in shared_runtime.items():
            assert checked[key] == expected

    omitted = dict(runtime_by_arm["no_rtp"])
    omitted.pop("evaluation_only_cohort_sha256")
    with pytest.raises(
        runner.RTPThreeArmRunnerError,
        match="runtime identity mismatch at evaluation_only_cohort_sha256",
    ):
        runner._runtime_profile_contract(manifest, "no_rtp", omitted)
    tampered = {
        **runtime_by_arm["recursive_rtp"],
        "research_control_registry_sha256": factory._sha256_bytes(b"tampered"),
    }
    with pytest.raises(
        runner.RTPThreeArmRunnerError,
        match="runtime identity mismatch at research_control_registry_sha256",
    ):
        runner._runtime_profile_contract(manifest, "recursive_rtp", tampered)

    missing_manifest = dict(shared_artifacts)
    missing_manifest.pop("planner_preflight_receipt")
    with pytest.raises(
        factory.R198ProductionFactoryError,
        match="not the exact r198 identity set",
    ):
        factory._shared_assets({"shared_artifacts": missing_manifest}, inputs)
    tampered_manifest = {
        **shared_artifacts,
        "evaluation_only_cohort": {
            **shared_artifacts["evaluation_only_cohort"],
            "sha256": factory._sha256_bytes(b"tampered cohort"),
        },
    }
    with pytest.raises(factory.R198ProductionFactoryError, match="checksum mismatch"):
        factory._shared_assets({"shared_artifacts": tampered_manifest}, inputs)


def _real_candidate_assets_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    completion_filename: str = factory.R197_COMPLETION_RECEIPT_FILENAME,
    authority_overrides: dict[str, object] | None = None,
) -> factory._CandidateAssets:
    source = tmp_path / "source-snapshot"
    factory_module = _write_readonly(
        source / "poke_bot" / "rtp_r198_production_factory.py", "# frozen factory\n"
    )
    candidate_root = source / "evaluation-artifacts" / "r197-candidate"
    parent = _write_readonly(candidate_root / "parent-checkpoint.pt", "parent")
    sidecar = _write_readonly(candidate_root / "rtp-shadow-planner.pt", "sidecar")
    sidecar_receipt = _write_readonly(
        candidate_root / "rtp-shadow-planner.pt.receipt.json", "sidecar receipt"
    )
    authority = {
        "shadow_only": True,
        "serving_eligible": False,
        "action_authority_enabled": False,
        "selector_authority": False,
        "live_checkpoint_publication": False,
        "submission_eligible": False,
    }
    authority.update(authority_overrides or {})
    completion_receipt = _write_readonly_json(
        candidate_root / completion_filename,
        {
            "schema": factory.R197_COMPLETION_RECEIPT_SCHEMA,
            "status": "completed_shadow_only",
            "candidate_id": factory.R198_CANDIDATE_ID,
            "candidate_contract_sha256": factory.R198_CANDIDATE_CONTRACT_SHA256,
            "authority": authority,
        },
    )
    deck = _write_readonly(candidate_root / "deck.csv", "candidate deck\n")
    matchup_tree = _write_readonly(candidate_root / "matchup-tree.json", "{}\n")
    artifacts = {
        "parent_checkpoint": _file_identity(parent),
        "sidecar": _file_identity(sidecar),
        "sidecar_receipt": _file_identity(sidecar_receipt),
        "completion_receipt": _file_identity(completion_receipt),
        "deck": _file_identity(deck),
        "matchup_tree": _file_identity(matchup_tree),
    }
    manifest = _write_readonly_json(
        candidate_root / "manifest.json",
        {
            "schema": factory.CANDIDATE_SNAPSHOT_SCHEMA,
            "status": "sealed",
            "no_symlinks": True,
            "all_paths_read_only": True,
            "candidate_id": factory.R198_CANDIDATE_ID,
            "candidate_contract_sha256": factory.R198_CANDIDATE_CONTRACT_SHA256,
            "package_root": str(candidate_root),
            "artifacts": artifacts,
        },
    )
    _freeze_tree(candidate_root)
    monkeypatch.setattr(factory, "__file__", str(factory_module))
    monkeypatch.setattr(factory, "R195_PARENT_SHA256", artifacts["parent_checkpoint"]["sha256"])
    monkeypatch.setattr(factory, "R197_SIDECAR_SHA256", artifacts["sidecar"]["sha256"])
    monkeypatch.setattr(
        factory, "R197_SIDECAR_RECEIPT_SHA256", artifacts["sidecar_receipt"]["sha256"]
    )
    monkeypatch.setattr(
        factory,
        "R197_COMPLETION_RECEIPT_SHA256",
        artifacts["completion_receipt"]["sha256"],
    )
    monkeypatch.setattr(
        factory,
        "R197_COMPLETION_RECEIPT_BYTES",
        artifacts["completion_receipt"]["bytes"],
    )
    monkeypatch.setattr(factory, "R195_DECK_CSV_SHA256", artifacts["deck"]["sha256"])
    monkeypatch.setattr(
        factory, "R195_MATCHUP_TREE_SHA256", artifacts["matchup_tree"]["sha256"]
    )
    monkeypatch.setattr(factory, "_validate_candidate_deck", lambda _identity: [])
    spec = {
        "factory_module": _file_identity(factory_module),
        "candidate_snapshot": _file_identity(manifest),
        "artifacts": artifacts,
    }
    return factory._candidate_assets(
        spec,
        source,
        factory._sha256_bytes(b"source tree"),
        validate_sidecar_payload=False,
    )


def test_candidate_assets_revalidate_canonical_frozen_completion_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assets = _real_candidate_assets_fixture(tmp_path, monkeypatch)
    assert assets.completion_receipt["path"] == str(
        assets.package_root / factory.R197_COMPLETION_RECEIPT_FILENAME
    )
    assert assets.completion_receipt["mode"] == 0o444


def test_candidate_assets_reject_noncanonical_or_authoritative_completion_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(
        factory.R198ProductionFactoryError,
        match="canonical snapshot-local file",
    ):
        _real_candidate_assets_fixture(
            tmp_path / "alias", monkeypatch, completion_filename="receipt-copy.json"
        )

    with pytest.raises(
        factory.R198ProductionFactoryError,
        match="unexpectedly grants selector_authority",
    ):
        _real_candidate_assets_fixture(
            tmp_path / "authority",
            monkeypatch,
            authority_overrides={"selector_authority": True},
        )


def test_runtime_derives_official_deck_from_real_prepared_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The evaluator-normalized opponent omits deck without widening authority."""

    import poke_bot.rtp_three_arm_evaluation as evaluator

    source, base_spec, official_decks = _production_shaped_base_spec_fixture(
        tmp_path, monkeypatch
    )
    profile = {
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "num_plan_candidates": 4,
        "max_recursion_depth": 2,
        "max_plan_length": 12,
        "d_model": 96,
        "dynamics_width": 192,
        "complexity_option_threshold": 8,
        "complexity_entropy_threshold": 1.5,
        "repair_budget": 1,
    }
    sidecar = {
        "path": str(tmp_path / "sidecar.pt"),
        "sha256": factory._sha256_bytes(b"sidecar"),
        "bytes": 7,
    }
    arms = {
        "no_rtp": {"profile": dict(profile), "rtp_sidecar": None},
        "direct_bridge_recursive_disabled": {
            "profile": dict(profile),
            "rtp_sidecar": sidecar,
        },
        "recursive_rtp": {"profile": dict(profile), "rtp_sidecar": sidecar},
    }
    cases = {
        (opponent_id, seat, replicate): {
            "cell_id": f"{opponent_id}-{seat}-{replicate}",
            "case_id": f"case:{opponent_id}:{seat}:{replicate}",
            "content_digest": factory._sha256_bytes(
                f"case:{opponent_id}:{seat}:{replicate}".encode()
            ),
        }
        for opponent_id in evaluator.R198_OFFICIAL_CONTROL_ORDER
        for seat in (0, 1)
        for replicate in range(evaluator.OFFICIAL_CONTROL_REPLICATES)
    }
    pairing = {
        "abi": {
            "capture_boundary": "post_battle_start_first_external_selection",
            "boundary_tag": 1,
        }
    }

    monkeypatch.setattr(evaluator, "_normalize_gates", lambda _raw: {})
    monkeypatch.setattr(evaluator, "_normalize_latency_slo", lambda _raw, _gates: {})
    monkeypatch.setattr(
        evaluator,
        "_normalize_shared_artifacts",
        lambda _raw: {"deck": {"sha256": factory._sha256_bytes(b"candidate deck")}},
    )
    monkeypatch.setattr(evaluator, "_normalize_arm_input", lambda raw: raw)
    monkeypatch.setattr(evaluator, "_normalize_arm", lambda _arm, raw: raw)
    monkeypatch.setattr(
        evaluator, "_normalize_candidate_evaluation_binding", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(evaluator, "_normalize_planner_pass_preflight", lambda *_args: {})
    # Deliberately do not patch _normalize_opponents: it must produce the
    # real prepared manifest shape that omits the redundant deck identity.
    monkeypatch.setattr(evaluator, "_validate_official_control_panel", lambda *_args: {})
    monkeypatch.setattr(
        evaluator,
        "_normalize_r197_source_exclusion_binding",
        lambda *_args, **_kwargs: {
            "evaluation_case_bindings": [],
            "evaluation_case_bindings_sha256": factory._sha256_bytes(b"case bindings"),
            "evaluation_only_cohort": {"sha256": factory._sha256_bytes(b"cohort")},
            "source_exclusion_proof": {"sha256": factory._sha256_bytes(b"source proof")},
        },
    )
    monkeypatch.setattr(evaluator, "_case_bindings_with_cell_ids", lambda *_args: cases)
    monkeypatch.setattr(evaluator, "_normalize_pairing_capability", lambda _raw: pairing)
    monkeypatch.setattr(
        evaluator,
        "_normalize_evaluation_cg_closure",
        lambda _raw, _pairing: {"normalized": "unrelated CG proof"},
    )
    monkeypatch.setattr(
        evaluator, "_validate_production_factory_runtime_library", lambda *_args: None
    )

    def normalized_rng(*_args: object, **kwargs: object) -> list[dict[str, object]]:
        expected_cases = kwargs["expected_cases"]
        assert isinstance(expected_cases, dict)
        return [
            {
                "id": f"rng:{opponent_id}:{seat}:{replicate}",
                "kind": "snapshot",
                "opponent_id": opponent_id,
                "candidate_seat": seat,
                "replicate": replicate,
                "snapshot_artifact": {
                    "sha256": factory._sha256_bytes(
                        f"snapshot:{opponent_id}:{seat}:{replicate}".encode()
                    ),
                    "bytes": 1,
                },
                "seal": {"sha256": factory._sha256_bytes(b"seal")},
                "requested_seed_audit_only": None,
            }
            for opponent_id, seat, replicate in expected_cases
        ]

    monkeypatch.setattr(evaluator, "_normalize_rng_materials", normalized_rng)
    prepared_path = evaluator.prepare_three_arm_manifest(
        output_path=tmp_path / "prepared-manifest-without-deck.json",
        production_factory={"sealed": True},
        shared_artifacts={},
        arms=arms,
        candidate_evaluation_binding={},
        opponents=base_spec["opponents"],
        rng_materials=[],
        pairing_capability={"receipt": "normalized by test seam"},
        evaluation_cg_closure={},
        source_exclusion_proof={},
        replicates_per_seat=evaluator.OFFICIAL_CONTROL_REPLICATES,
    )
    prepared_manifest = json.loads(prepared_path.read_text(encoding="utf-8"))
    assert all("deck" not in opponent for opponent in prepared_manifest["opponents"])
    for opponent_id, expected_deck in official_decks.items():
        runtime_package = factory._official_package(prepared_manifest, opponent_id, source)
        for key, expected in expected_deck.items():
            assert runtime_package.deck[key] == expected
        assert runtime_package.deck["path"] == str(
            source / "baselines" / "official" / opponent_id / "deck.csv"
        )


@pytest.mark.parametrize("unsafe_path", ("../deck.csv", "/tmp/deck.csv", "./deck.csv"))
def test_base_spec_rejects_unsafe_relative_official_package_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_path: str
) -> None:
    with pytest.raises(factory.R198ProductionFactoryError, match="unsafe|unexpected importable tree"):
        _production_shaped_base_spec_fixture(
            tmp_path,
            monkeypatch,
            deck_entry_path=unsafe_path,
        )


def _normalized_closure_fixture(
    tmp_path: Path,
) -> tuple[Path, factory._CGAssets, dict[str, object], dict[str, object]]:
    """Build just enough real sealed bytes for evaluator closure normalization.

    The evaluator's closure normalizer is intentionally left unpatched by the
    integration test below.  Other manifest sections are patched only because
    the factory owns neither the 1,000-cell schedule nor its input producer.
    """

    source = tmp_path / "sealed-source"
    runtime = source / "kaggle" / "input" / "rtp-eval-cg"
    library = _write_readonly(runtime / "cg" / "libcg.so", "sealed DSO")
    for name in ("__init__.py", "api.py", "game.py", "sim.py", "utils.py"):
        _write_readonly(runtime / "cg" / name, f"# sealed {name}\n")
    runtime_library = _file_identity(library)
    private_engine = _write_readonly(
        tmp_path / "private-pairing" / "libcg.so", "sealed DSO", mode=0o555
    )
    private_engine_identity = _file_identity(private_engine)
    build = _write_readonly(tmp_path / "private-pairing" / "build.json", "build")
    build_identity = _file_identity(build)

    def tree_manifest(path: Path, schema: str) -> dict[str, object]:
        records: list[dict[str, object]] = []
        for name in factory._EVALUATION_CG_CLOSURE_TREE_PATHS:
            if name == "libcg.so":
                records.append(
                    {
                        "relative_path": name,
                        "sha256": runtime_library["sha256"],
                        "bytes": runtime_library["bytes"],
                    }
                )
            else:
                contents = f"{schema}:{name}".encode()
                records.append(
                    {
                        "relative_path": name,
                        "sha256": factory._sha256_bytes(contents),
                        "bytes": len(contents),
                    }
                )
        core: dict[str, object] = {
            "schema": schema,
            "file_count": len(records),
            "files": records,
        }
        tree_path = _write_readonly_json(
            path,
            {**core, "tree_sha256": factory._canonical_digest(core)},
        )
        return _file_identity(tree_path)

    source_tree = tree_manifest(
        runtime / "cg-source-manifest.json",
        "poke_bot.recursive_turn_planner.true_rng_pairing_eval_cg_source_manifest/v1",
    )
    closure_tree = tree_manifest(
        runtime / "cg-closure-manifest.json",
        factory.EVALUATION_CG_CLOSURE_MANIFEST_SCHEMA,
    )
    public_engine = _write_readonly(tmp_path / "public-cg" / "libcg.so", "public DSO")
    parity_path = _write_readonly_json(
        runtime / "cg-metadata-parity.json",
        {
            "schema": factory.EVALUATION_CG_METADATA_PARITY_SCHEMA,
            "status": "passed",
            "independent_processes": True,
            "public_initialized_before_pairing": True,
            "pairing_private_initialize_after_public_passed": True,
            "distinct_dso_handles": True,
            "public_cg_engine": _file_identity(public_engine),
            "pairing_engine": runtime_library,
            "all_card_canonical_sha256": factory._sha256_bytes(b"canonical cards"),
            "all_attack_canonical_sha256": factory._sha256_bytes(b"canonical attacks"),
            "public_all_card_raw_sha256": factory._sha256_bytes(b"raw cards"),
            "pairing_all_card_raw_sha256": factory._sha256_bytes(b"raw cards"),
            "public_all_attack_raw_sha256": factory._sha256_bytes(b"raw attacks"),
            "pairing_all_attack_raw_sha256": factory._sha256_bytes(b"raw attacks"),
        },
    )
    parity = _file_identity(parity_path)
    abi = factory._sha256_bytes(b"canonical ABI")
    receipt_path = _write_readonly_json(
        runtime / factory.EVALUATION_CG_CLOSURE_FILENAME,
        {
            "schema": factory.EVALUATION_CG_CLOSURE_SCHEMA,
            "status": "sealed",
            "engine_artifact": runtime_library,
            "pairing_build_artifact": build_identity,
            "cg_source_manifest": source_tree,
            "closure_manifest": closure_tree,
            "metadata_parity": parity,
            "canonical_abi_sha256": abi,
            "sim_initializer_symbol": "RtpPairingSnapshotInitialize",
            "snapshot_abi_version": 2,
        },
    )
    receipt = _file_identity(receipt_path)
    cg = factory._CGAssets(
        runtime_root=runtime,
        closure_manifest=receipt,
        library=runtime_library,
        closure_payload=json.loads(receipt_path.read_text(encoding="utf-8")),
        closure_evidence={
            "engine_artifact": runtime_library,
            "pairing_build_artifact": build_identity,
            "cg_source_manifest": source_tree,
            "closure_manifest": closure_tree,
            "metadata_parity": parity,
        },
    )
    pairing = {
        "engine_artifact": private_engine_identity,
        "build_artifact": build_identity,
        "canonical_abi_sha256": abi,
        "abi": {
            "capture_boundary": "post_battle_start_first_external_selection",
            "boundary_tag": 1,
        },
    }
    return source, cg, {"receipt": receipt, "runtime_library": runtime_library}, pairing


def test_actual_prepare_manifest_normalized_closure_is_accepted_by_factory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real evaluator expansion before the factory sees a child manifest."""

    import poke_bot.rtp_three_arm_evaluation as evaluator

    source, cg, base_closure, pairing = _normalized_closure_fixture(tmp_path)
    profile = {
        "max_neural_passes": 256,
        "max_action_combos": 1024,
        "num_plan_candidates": 4,
        "max_recursion_depth": 2,
        "max_plan_length": 12,
        "d_model": 96,
        "dynamics_width": 192,
        "complexity_option_threshold": 8,
        "complexity_entropy_threshold": 1.5,
        "repair_budget": 1,
    }
    sidecar = {
        "path": str(tmp_path / "sidecar.pt"),
        "sha256": factory._sha256_bytes(b"sidecar"),
        "bytes": 7,
    }
    normalized_arms = {
        "no_rtp": {"profile": dict(profile), "rtp_sidecar": None},
        "direct_bridge_recursive_disabled": {"profile": dict(profile), "rtp_sidecar": sidecar},
        "recursive_rtp": {"profile": dict(profile), "rtp_sidecar": sidecar},
    }
    opponents = [
        {"id": opponent_id, "content_digest": f"digest:{opponent_id}"}
        for opponent_id in evaluator.R198_OFFICIAL_CONTROL_ORDER
    ]
    cases = {
        (opponent_id, seat, replicate): {
            "cell_id": f"{opponent_id}-{seat}-{replicate}",
            "case_id": f"case:{opponent_id}:{seat}:{replicate}",
            "content_digest": factory._sha256_bytes(
                f"case:{opponent_id}:{seat}:{replicate}".encode()
            ),
        }
        for opponent_id in evaluator.R198_OFFICIAL_CONTROL_ORDER
        for seat in (0, 1)
        for replicate in range(evaluator.OFFICIAL_CONTROL_REPLICATES)
    }

    monkeypatch.setattr(evaluator, "_normalize_gates", lambda _raw: {})
    monkeypatch.setattr(evaluator, "_normalize_latency_slo", lambda _raw, _gates: {})
    monkeypatch.setattr(
        evaluator,
        "_normalize_shared_artifacts",
        lambda _raw: {"deck": {"sha256": factory._sha256_bytes(b"candidate deck")}},
    )
    monkeypatch.setattr(evaluator, "_normalize_arm_input", lambda raw: raw)
    monkeypatch.setattr(evaluator, "_normalize_arm", lambda _arm, raw: raw)
    monkeypatch.setattr(
        evaluator, "_normalize_candidate_evaluation_binding", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(evaluator, "_normalize_planner_pass_preflight", lambda *_args: {})
    monkeypatch.setattr(evaluator, "_normalize_opponents", lambda _raw: opponents)
    monkeypatch.setattr(evaluator, "_validate_official_control_panel", lambda *_args: {})
    monkeypatch.setattr(
        evaluator,
        "_normalize_r197_source_exclusion_binding",
        lambda *_args, **_kwargs: {
            "evaluation_case_bindings": [],
            "evaluation_case_bindings_sha256": factory._sha256_bytes(b"case bindings"),
            "evaluation_only_cohort": {"sha256": factory._sha256_bytes(b"cohort")},
            "source_exclusion_proof": {"sha256": factory._sha256_bytes(b"source proof")},
        },
    )
    monkeypatch.setattr(evaluator, "_case_bindings_with_cell_ids", lambda *_args: cases)
    monkeypatch.setattr(evaluator, "_normalize_pairing_capability", lambda _raw: pairing)
    monkeypatch.setattr(
        evaluator, "_validate_production_factory_runtime_library", lambda *_args: None
    )

    def normalized_rng(*_args: object, **kwargs: object) -> list[dict[str, object]]:
        expected_cases = kwargs["expected_cases"]
        assert isinstance(expected_cases, dict)
        return [
            {
                "id": f"rng:{opponent_id}:{seat}:{replicate}",
                "kind": "snapshot",
                "opponent_id": opponent_id,
                "candidate_seat": seat,
                "replicate": replicate,
                "snapshot_artifact": {
                    "sha256": factory._sha256_bytes(
                        f"snapshot:{opponent_id}:{seat}:{replicate}".encode()
                    ),
                    "bytes": 1,
                },
                "seal": {"sha256": factory._sha256_bytes(b"seal")},
                "requested_seed_audit_only": None,
            }
            for opponent_id, seat, replicate in expected_cases
        ]

    monkeypatch.setattr(evaluator, "_normalize_rng_materials", normalized_rng)
    manifest_path = evaluator.prepare_three_arm_manifest(
        output_path=tmp_path / "prepared-manifest.json",
        production_factory={"sealed": True},
        shared_artifacts={},
        arms=normalized_arms,
        candidate_evaluation_binding={},
        opponents=opponents,
        rng_materials=[],
        pairing_capability={"receipt": "ignored by patched capability normalizer"},
        evaluation_cg_closure=base_closure,
        source_exclusion_proof={},
        replicates_per_seat=evaluator.OFFICIAL_CONTROL_REPLICATES,
    )
    prepared_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    normalized = prepared_manifest["evaluation_cg_closure"]
    assert set(normalized) == factory._EVALUATION_CG_CLOSURE_NORMALIZED_KEYS
    factory._verify_manifest_evaluation_cg_closure(
        prepared_manifest,
        cg=cg,
        source_root=source,
    )

    raw_pair_final = {
        "schema": factory.THREE_ARM_EVALUATION_MANIFEST_SCHEMA,
        "evaluation_cg_closure": base_closure,
    }
    with pytest.raises(factory.R198ProductionFactoryError, match="full normalized"):
        factory._verify_manifest_evaluation_cg_closure(
            raw_pair_final,
            cg=cg,
            source_root=source,
        )
    normalized["all_attack_canonical_sha256"] = factory._sha256_bytes(b"tampered")
    with pytest.raises(factory.R198ProductionFactoryError, match="all_attack_canonical_sha256"):
        factory._verify_manifest_evaluation_cg_closure(
            prepared_manifest,
            cg=cg,
            source_root=source,
        )


def test_shared_cg_handle_check_requires_one_loader_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime = tmp_path / "rtp-eval-cg"
    sim_path = _write_readonly(runtime / "cg" / "sim.py", "# sealed test sim\n")
    (runtime / "cg").chmod(0o555)
    runtime.chmod(0o555)

    fake_cg = types.ModuleType("cg")
    fake_cg.__path__ = [str(runtime / "cg")]  # type: ignore[attr-defined]
    fake_sim = types.ModuleType("cg.sim")
    fake_sim.__file__ = str(sim_path)
    fake_sim.lib = SimpleNamespace(_handle=41)
    monkeypatch.setitem(sys.modules, "cg", fake_cg)
    monkeypatch.setitem(sys.modules, "cg.sim", fake_sim)
    from poke_bot import cg_env

    monkeypatch.setattr(cg_env, "ensure_cg_importable", lambda: runtime)
    assets = factory._CGAssets(
        runtime_root=runtime,
        closure_manifest={},
        library={},
        closure_payload={},
        closure_evidence={},
    )
    factory._require_shared_cg_dso_handle(
        SimpleNamespace(_library=SimpleNamespace(_handle=41)), assets
    )
    with pytest.raises(factory.R198ProductionFactoryError, match="different DSO handles"):
        factory._require_shared_cg_dso_handle(
            SimpleNamespace(_library=SimpleNamespace(_handle=42)), assets
        )
