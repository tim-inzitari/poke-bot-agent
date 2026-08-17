from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poke_bot.baselines_runtime import BaselineSpec
from poke_bot import baselines_runtime
from poke_bot import r241_marnie_direct_policy_adapter as adapter


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, payload: bytes | str) -> None:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    path.write_bytes(payload)


def _fake_h10_package(
    tmp_path: Path,
    monkeypatch,
) -> tuple[BaselineSpec, Path, Path, Path]:
    root = tmp_path / adapter.R241_H10_DIR_NAME
    root.mkdir()
    model = root / "model.pt"
    deck = root / "deck.csv"
    tree = root / "matchup_tree.json"
    old_cg = root / "cg" / "libcg.so"
    old_cg.parent.mkdir()
    _write(model, b"fake-r241-model")
    deck.write_text("\n".join(["741"] * 60) + "\n", encoding="utf-8")
    _write(tree, b'{"tree":"bound-data-only"}')
    _write(old_cg, b"known-old-libcg")
    (root / "search_config.json").write_text(
        json.dumps({"enabled": False}), encoding="utf-8"
    )
    # If this entry point is ever imported, the test fails immediately.  The
    # data-only adapter must not touch it while validating/loading the model.
    (root / "main.py").write_text(
        "raise RuntimeError('H10 package main.py was imported')\n",
        encoding="utf-8",
    )
    fake_content = "sha256:" + "1" * 64
    monkeypatch.setattr(adapter, "R241_H10_MODEL_SHA256", _sha256(model))
    monkeypatch.setattr(adapter, "R241_H10_MODEL_SIZE_BYTES", model.stat().st_size)
    monkeypatch.setattr(adapter, "R241_H10_CONTENT_SHA256", fake_content)
    monkeypatch.setattr(adapter, "R241_H10_MATCHUP_TREE_SHA256", _sha256(tree))
    monkeypatch.setattr(
        adapter, "R241_H10_MATCHUP_TREE_SIZE_BYTES", tree.stat().st_size
    )
    monkeypatch.setattr(adapter, "R241_OLD_EMBEDDED_LIBCG_SHA256", _sha256(old_cg))

    from poke_bot import baselines_runtime

    monkeypatch.setattr(
        baselines_runtime,
        "baseline_content_digest",
        lambda value: fake_content if Path(value).resolve() == root.resolve() else "",
    )
    cg_root = tmp_path / "official-cg-r236"
    cg_root.mkdir()
    monkeypatch.setattr(
        adapter,
        "validate_sealed_official_libcg",
        lambda path, *, environment: Path(path).resolve(),
    )
    spec = BaselineSpec(
        id=adapter.R241_H10_OPPONENT_ID,
        name="pinned H10 Marnie",
        dir_name=adapter.R241_H10_DIR_NAME,
        group="specialists",
        source="test",
        path=root,
    )
    receipt = tmp_path / "marnie-direct-policy-adapter.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": adapter.R241_MARNIE_ADAPTER_RECEIPT_SCHEMA,
                "revision": 241,
                "status": "passed",
                "passed": True,
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
                    "root_path": str(root.resolve()),
                    "content_sha256": fake_content,
                    "model": {
                        "relative_path": "model.pt",
                        "sha256": _sha256(model),
                        "size_bytes": model.stat().st_size,
                    },
                    "deck": {
                        "relative_path": "deck.csv",
                        "sha256": _sha256(deck),
                        "card_count": 60,
                    },
                    "matchup_tree": {
                        "relative_path": "matchup_tree.json",
                        "sha256": _sha256(tree),
                        "size_bytes": tree.stat().st_size,
                    },
                },
                "sealed_runtime": {
                    "cg_lib_path": str(cg_root.resolve()),
                    "linux_x86_64_sha256": adapter.R241_OFFICIAL_LINUX_LIBCG_SHA256,
                },
            }
        ),
        encoding="utf-8",
    )
    return spec, receipt, cg_root, tree


def _environment(receipt: Path, cg_root: Path) -> dict[str, str]:
    return {
        adapter.R241_DIRECT_POLICY_ONLY_ENV: "1",
        adapter.R241_DIRECT_POLICY_RECEIPT_ENV: str(receipt),
        "CG_LIB_PATH": str(cg_root),
        "POKEBOT_USE_RECURSIVE_TURN_PLANNER": "0",
        "POKEBOT_SEARCH_MODE": "policy",
        "POKEBOT_SUBMISSION_SEARCH_DISABLE": "1",
        # Matchup Adapter runtime remains expressly allowed in r241.
        "POKEBOT_MATCHUP_ADAPTER_RUNTIME": "1",
    }


def test_r241_h10_adapter_reads_only_bound_data_and_tree(
    tmp_path: Path, monkeypatch
) -> None:
    spec, receipt, cg_root, tree = _fake_h10_package(tmp_path, monkeypatch)

    model, deck, bound_tree, receipt_path = (
        adapter.validate_r241_marnie_direct_policy_adapter(
            spec, environment=_environment(receipt, cg_root)
        )
    )

    assert model.name == "model.pt"
    assert len(deck) == 60
    assert bound_tree == tree.resolve()
    assert receipt_path == receipt.resolve()


def test_r241_h10_adapter_passes_its_own_tree_to_direct_policy(
    tmp_path: Path, monkeypatch
) -> None:
    spec, receipt, cg_root, tree = _fake_h10_package(tmp_path, monkeypatch)
    observed: dict[str, object] = {}

    class FakePolicy:
        def reset_game(self) -> None:
            observed["reset"] = True

        def __call__(self, observation: dict) -> list[int]:
            observed["observation"] = observation
            return [0]

    def fake_build(model: Path, deck: list[int], matchup_tree: Path) -> FakePolicy:
        observed["model"] = model
        observed["deck"] = deck
        observed["tree"] = matchup_tree
        return FakePolicy()

    monkeypatch.setattr(adapter, "_build_direct_policy", fake_build)
    loaded = adapter.maybe_load_r241_direct_policy_agent(
        spec, environment=_environment(receipt, cg_root)
    )

    assert loaded is not None
    agent, deck = loaded
    assert len(deck) == 60
    assert observed["tree"] == tree.resolve()
    assert agent({"select": {"option": [1]}}) == [0]


def test_r241_filtering_h10_never_imports_its_legacy_entrypoint(
    tmp_path: Path, monkeypatch
) -> None:
    """The normal roster preflight must take the same data-only H10 path.

    ``filter_loadable_baselines`` probes every manifest entry before the
    active-gate roster is resolved.  It is therefore just as important as the
    later collection load: a direct H10 receipt must prevent this early probe
    from importing the package's deliberately failing ``main.py``.
    """

    spec, receipt, cg_root, _tree = _fake_h10_package(tmp_path, monkeypatch)
    monkeypatch.setenv(adapter.R241_DIRECT_POLICY_ONLY_ENV, "1")
    monkeypatch.setenv(adapter.R241_DIRECT_POLICY_RECEIPT_ENV, str(receipt))
    monkeypatch.setenv("CG_LIB_PATH", str(cg_root))
    monkeypatch.setenv("POKEBOT_USE_RECURSIVE_TURN_PLANNER", "0")
    monkeypatch.setenv("POKEBOT_SEARCH_MODE", "policy")
    monkeypatch.setenv("POKEBOT_SUBMISSION_SEARCH_DISABLE", "1")

    class FakePolicy:
        def __call__(self, observation: dict) -> list[int]:
            return [0]

    monkeypatch.setattr(
        adapter,
        "_build_direct_policy",
        lambda _model, _deck, _tree: FakePolicy(),
    )
    loadable, failed = baselines_runtime.filter_loadable_baselines(
        [spec], verbose=False
    )

    assert loadable == [spec]
    assert failed == []


def test_r241_receipted_mode_leaves_other_public_packages_on_normal_loader(
    tmp_path: Path, monkeypatch
) -> None:
    _spec, receipt, cg_root, _tree = _fake_h10_package(tmp_path, monkeypatch)
    other = BaselineSpec(
        id="ordinary-diverse-public-opponent",
        name="ordinary",
        dir_name="ordinary",
        group="community",
        source="test",
        path=tmp_path / "ordinary",
    )

    assert (
        adapter.maybe_load_r241_direct_policy_agent(
            other, environment=_environment(receipt, cg_root)
        )
        is None
    )


def test_r241_h10_adapter_rejects_a_cg_root_mixed_into_the_package(
    tmp_path: Path, monkeypatch
) -> None:
    spec, receipt, _cg_root, _tree = _fake_h10_package(tmp_path, monkeypatch)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["sealed_runtime"]["cg_lib_path"] = str(spec.path.resolve())
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(adapter.R241MarnieAdapterError, match="mixed with the H10 package"):
        adapter.validate_r241_marnie_direct_policy_adapter(
            spec, environment=_environment(receipt, spec.path)
        )


def test_r241_baseline_isolation_removes_planner_leaks_but_not_default_behavior(
    monkeypatch
) -> None:
    monkeypatch.setenv("POKEBOT_R241_DIRECT_POLICY_ONLY", "1")
    monkeypatch.setenv("POKEBOT_SEARCH_MODE", "policy")

    with baselines_runtime._isolated_baseline_environment():
        import os

        os.environ["POKEBOT_MCTS_SIMS"] = "64"
        os.environ["POKEBOT_RTP_CHECKPOINT"] = "/tmp/old-rtp.pt"
        os.environ["POKEBOT_SEARCH_MODE"] = "oracle_mcts"

    import os

    assert "POKEBOT_MCTS_SIMS" not in os.environ
    assert "POKEBOT_RTP_CHECKPOINT" not in os.environ
    assert os.environ["POKEBOT_SEARCH_MODE"] == "policy"

    monkeypatch.delenv("POKEBOT_R241_DIRECT_POLICY_ONLY")
    with baselines_runtime._isolated_baseline_environment():
        os.environ["POKEBOT_MCTS_SIMS"] = "64"
    assert os.environ["POKEBOT_MCTS_SIMS"] == "64"
