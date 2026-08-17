from __future__ import annotations

import ast
from pathlib import Path

from poke_bot.r215_bo1000_launch import (
    CANARY_GENUINE_SEARCH_FIELDS,
    R216_EVALUATION_ID,
    R216_BO1000_GPU_UUID,
    R215_RUNTIME_BRIDGE_RELATIVE_PATH,
    SERVICE_TEMPLATE_RELATIVE_PATH,
    SOURCE_TEMPLATE_RELATIVE_PATH,
    OUTPUT_TEMPLATE_RELATIVE_PATH,
    build_launch_plan,
    clean_r215_runtime_environment,
    materialize_templates,
    parse_schedule,
    runtime_environment_receipt,
    validate_approximate_canary_results,
)


ROOT = Path(__file__).resolve().parents[1]


def _genuine_turn() -> dict[str, object]:
    return {
        "direct_policy_fallback_used": False,
        "selected_action_from_policy_last_result": True,
        **{field: 1 for field in CANARY_GENUINE_SEARCH_FIELDS},
    }


def test_r216_canary_plan_is_content_addressed_and_seat_swapped(tmp_path: Path) -> None:
    plan = build_launch_plan(repo_root=ROOT, mode="canary", canary_pairs=1)

    assert plan["evaluation_id"] == R216_EVALUATION_ID
    assert plan["mode"] == "canary"
    assert plan["local_exploratory_bo1000_authorized"] is True
    assert plan["kaggle_submission_authorized"] is False
    assert len(plan["schedule"]) == 2
    schedule = parse_schedule(plan)
    assert {game.experimental_seat for game in schedule} == {0, 1}
    assert len({game.game_nonce_sha256 for game in schedule}) == 2
    assert len({game.engine_seed_u32 for game in schedule}) == 1
    assert len({game.deck_order_seed_u32 for game in schedule}) == 1
    source_files = plan["source"]["source_files"]
    for relative in (
        R215_RUNTIME_BRIDGE_RELATIVE_PATH,
        SOURCE_TEMPLATE_RELATIVE_PATH,
        OUTPUT_TEMPLATE_RELATIVE_PATH,
        SERVICE_TEMPLATE_RELATIVE_PATH,
    ):
        assert relative.as_posix() in source_files

    paths = materialize_templates(stage_root=tmp_path, plan=plan)
    assert paths["launch_plan"].is_file()
    assert paths["source_dir"].name.startswith("alakazam-r216-src-")
    assert paths["output_dir"].name.startswith("alakazam-r216-bo1000-")


def test_r216_bo1000_plan_has_the_complete_balanced_schedule() -> None:
    plan = build_launch_plan(repo_root=ROOT, mode="bo1000")

    assert plan["mode"] == "bo1000"
    assert len(plan["schedule"]) == 1000
    assert plan["expected_balance"]["experimental_as_seat_0"] == 500
    assert plan["expected_balance"]["experimental_as_seat_1"] == 500
    assert plan["source"]["runtime_policy"]["bo1000_gpu_binding"] == R216_BO1000_GPU_UUID
    assert plan["no_early_stop"] is True


def test_r216_canary_requires_one_real_mcts_turn_but_allows_late_fallback() -> None:
    plan = build_launch_plan(repo_root=ROOT, mode="canary", canary_pairs=1)
    schedule = parse_schedule(plan)
    results = [
        {
            "game_nonce_sha256": schedule[0].game_nonce_sha256,
            "terminal_status": "completed",
            "invalid_action": False,
            "crash": False,
            "experimental_turn_receipts": [_genuine_turn(), {"direct_policy_fallback_used": True}],
        },
        {
            "game_nonce_sha256": schedule[1].game_nonce_sha256,
            "terminal_status": "completed",
            "invalid_action": False,
            "crash": False,
            "experimental_turn_receipts": [{"direct_policy_fallback_used": True}],
        },
    ]
    acceptance = validate_approximate_canary_results(
        plan=plan,
        game_results=results,
        frozen_runtime={"package_content_sha256": "sha256:" + "0" * 64, "bundle_sha256": "sha256:" + "1" * 64},
        controller={"module_sha256": plan["source"]["controller"]["module_sha256"]},
    )

    assert acceptance["status"] == "accepted_local_approximate_canary"
    assert acceptance["genuine_mcts_turn_count"] == 1
    assert acceptance["direct_policy_fallback_turn_count"] == 2


def test_r216_runtime_environment_pins_no_rtp_features_and_package_libcg_root() -> None:
    package_root = ROOT / "frozen-r195-example"
    seeded_engine = package_root / "libcg_hidden_pristine_batch_b77afbd3.so"
    environment, scrubbed = clean_r215_runtime_environment(
        package_root=package_root,
        seeded_engine_lib=seeded_engine,
        inherited={
            "POKEBOT_RTP_CHECKPOINT": "bad",
            "POKEBOT_GUIDE2VEC_ENABLED": "1",
            "POKEBOT_SLOWKING_DISTILL_RUN": "1",
            "POKEBOT_LIBCG_PATH": "/training-only/libcg.so",
        },
    )
    receipt = runtime_environment_receipt(
        environment, scrubbed, seeded_engine_lib=seeded_engine
    )

    assert "POKEBOT_RTP_CHECKPOINT" in scrubbed
    assert "POKEBOT_GUIDE2VEC_ENABLED" in scrubbed
    assert "POKEBOT_SLOWKING_DISTILL_RUN" in scrubbed
    assert environment["POKEBOT_LIBCG_PATH"] == str(seeded_engine.resolve())
    assert environment["CG_LIB_PATH"] == str(package_root.resolve())
    assert receipt["rtp_enabled"] is False
    assert receipt["guide2vec_enabled"] is False
    assert receipt["matchup_adapter_enabled"] is True
    assert receipt["seeded_engine_override_path"] == str(seeded_engine.resolve())


def test_r216_launcher_files_have_no_external_submission_import_or_queue_path() -> None:
    files = (
        ROOT / "poke_bot/r215_bo1000_launch.py",
        ROOT / "poke_bot/r215_seeded_mirror_runtime.py",
        ROOT / "scripts/run_alakazam_full_turn_belief_mcts_bo1000_r215.py",
    )
    imported: set[str] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.lower() for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.lower())
    assert not {
        name
        for name in imported
        if "kaggle" in name or name == "queue" or name.startswith("queue.")
    }

    service = (ROOT / SERVICE_TEMPLATE_RELATIVE_PATH).read_text(encoding="utf-8")
    assert "--canary-acceptance" in service
    assert "--seeded-engine-lib @R216_SEEDED_ENGINE_LIB@" in service
    assert "--local-exploratory-override" in service
    assert "Restart=no" in service
    assert "@R216_PYTHON311_EXECUTABLE@" in service
    assert "ConditionPathIsDirectory=@R216_OUTPUT_ROOT@" in service
    assert f"Environment=CUDA_VISIBLE_DEVICES={R216_BO1000_GPU_UUID}" in service
    assert f"Environment=NVIDIA_VISIBLE_DEVICES={R216_BO1000_GPU_UUID}" in service
