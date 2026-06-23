#!/usr/bin/env python3
"""Validate a Kaggle competition submission tarball before upload.

Kaggle requires a .tar.gz with main.py and deck.csv at the archive root (not nested).
See: https://www.kaggle.com/competitions/pokemon-tcg-ai-battle/overview
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tarfile
import tempfile
from pathlib import Path


REQUIRED_ROOT_FILES = ("main.py", "deck.csv")
REQUIRED_PATHS = ("cg/libcg.so", "value_model.pt", "policy_runtime.py", "beam_search.py", "model.py", "features.py", "game_tracker.py", "rewards.py")
FORBIDDEN_PREFIXES = ("submission/", "./submission/", "dist/", "./dist/")


def normalize_member_path(name: str) -> str:
    return name.lstrip("./").replace("\\", "/")


def list_archive_paths(archive: Path) -> list[str]:
    with tarfile.open(archive, "r:gz") as tar:
        return [normalize_member_path(member.name) for member in tar.getmembers() if member.isfile()]


def validate_archive_layout(paths: list[str]) -> list[str]:
    errors: list[str] = []
    path_set = set(paths)

    if not paths:
        return ["archive contains no files"]

    for required in REQUIRED_ROOT_FILES:
        if required not in path_set:
            errors.append(f"missing required root file: {required}")

    for forbidden in FORBIDDEN_PREFIXES:
        if any(path.startswith(forbidden) for path in paths):
            errors.append(f"nested path forbidden by Kaggle layout: {forbidden}*")

    nested_main = [path for path in paths if path.endswith("/main.py") and path != "main.py"]
    if nested_main:
        errors.append(f"main.py must be at archive root, found nested: {nested_main}")

    nested_deck = [path for path in paths if path.endswith("/deck.csv") and path != "deck.csv"]
    if nested_deck:
        errors.append(f"deck.csv must be at archive root, found nested: {nested_deck}")

    for required in REQUIRED_PATHS:
        if required not in path_set:
            errors.append(f"missing required path: {required}")

    return errors


def validate_deck_csv(deck_text: str) -> list[str]:
    errors: list[str] = []
    values = [line.strip() for line in deck_text.splitlines() if line.strip()]
    if len(values) != 60:
        errors.append(f"deck.csv must contain exactly 60 card IDs, found {len(values)}")
        return errors
    for index, token in enumerate(values, start=1):
        try:
            card_id = int(token)
        except ValueError:
            errors.append(f"deck.csv line {index} is not an integer: {token!r}")
            continue
        if card_id < 0:
            errors.append(f"deck.csv line {index} has negative card id: {card_id}")
    return errors


def smoke_test_agent(extract_dir: Path, sample_observation: dict | None) -> list[str]:
    if sys.version_info < (3, 10):
        print(
            "warning: skipping agent smoke test on Python "
            f"{sys.version_info.major}.{sys.version_info.minor} "
            "(Kaggle runs Python 3.11+; layout checks still applied)",
            file=sys.stderr,
        )
        return []

    errors: list[str] = []
    main_path = extract_dir / "main.py"
    if not main_path.exists():
        return ["cannot smoke test: extracted main.py missing"]

    spec = importlib.util.spec_from_file_location("submission_main", main_path)
    if spec is None or spec.loader is None:
        return ["failed to load main.py module spec"]
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(extract_dir))
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return [f"importing main.py failed: {exc}"]
    finally:
        if str(extract_dir) in sys.path:
            sys.path.remove(str(extract_dir))

    agent = getattr(module, "agent", None)
    if not callable(agent):
        return ["main.py must define callable agent(obs_dict)"]

    if sample_observation is None:
        return errors

    try:
        action = agent(sample_observation)
    except Exception as exc:
        errors.append(f"agent(sample_observation) failed: {exc}")
        return errors

    if not isinstance(action, list) or not action:
        errors.append(f"agent(sample_observation) returned unexpected value: {action!r}")
    return errors


def load_sample_observation(path: Path | None, *, require_search_begin: bool = False) -> dict | None:
    if path is None or not path.exists():
        return None
    import json

    fallback: dict | None = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            obs = row.get("observation")
            if not isinstance(obs, dict):
                continue
            if obs.get("search_begin_input"):
                return obs
            if fallback is None:
                fallback = obs
    if require_search_begin:
        return None
    return fallback


def smoke_test_beam_search(extract_dir: Path, sample_observation: dict | None) -> list[str]:
    if sys.version_info < (3, 10):
        print(
            "warning: skipping beam smoke test on Python "
            f"{sys.version_info.major}.{sys.version_info.minor} "
            "(Kaggle runs Python 3.11+; layout checks still applied)",
            file=sys.stderr,
        )
        return []

    if sample_observation is None or not sample_observation.get("search_begin_input"):
        return ["beam smoke test: no observation with search_begin_input in sample data"]

    errors: list[str] = []
    sys.path.insert(0, str(extract_dir))
    try:
        from policy_runtime import get_policy_agent, legal_actions

        policy = get_policy_agent()
        policy.reset()
        deck_path = extract_dir / "deck.csv"
        deck = [int(line.strip()) for line in deck_path.read_text(encoding="utf-8").splitlines() if line.strip()][:60]

        obs = dict(sample_observation)
        obs["remainingOverageTime"] = max(float(obs.get("remainingOverageTime") or 0), 9999.0)

        select = obs.get("select") or {}
        options = select.get("option") or []
        min_count = int(select.get("minCount", 1))
        max_count = int(select.get("maxCount", 1))
        legal = legal_actions(len(options), min_count, max_count)

        action = policy.choose_action(obs, our_deck=deck)
        if not isinstance(action, list) or not action:
            errors.append(f"beam choose_action returned unexpected value: {action!r}")
            return errors
        if action not in legal:
            errors.append(f"beam choose_action returned illegal indices: {action!r}")
    except Exception as exc:
        errors.append(f"beam smoke test failed: {exc}")
    finally:
        if str(extract_dir) in sys.path:
            sys.path.remove(str(extract_dir))
    return errors


def validate_submission(archive: Path, *, smoke_test: bool, sample_data: Path | None) -> list[str]:
    errors: list[str] = []

    if not archive.exists():
        return [f"archive not found: {archive}"]
    if archive.suffixes[-2:] != [".tar", ".gz"] and archive.suffix != ".tgz":
        errors.append("submission must be a .tar.gz archive")

    try:
        paths = list_archive_paths(archive)
    except tarfile.TarError as exc:
        return [f"invalid tar.gz archive: {exc}"]

    errors.extend(validate_archive_layout(paths))

    with tarfile.open(archive, "r:gz") as tar:
        deck_member = tar.getmember("deck.csv")
        deck_text = tar.extractfile(deck_member).read().decode("utf-8")  # type: ignore[union-attr]
    errors.extend(validate_deck_csv(deck_text))

    if not smoke_test:
        return errors

    sample_observation = load_sample_observation(sample_data)
    beam_sample = load_sample_observation(sample_data, require_search_begin=True)
    with tempfile.TemporaryDirectory(prefix="submission-validate-") as tmp:
        extract_dir = Path(tmp)
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(extract_dir)
        errors.extend(smoke_test_agent(extract_dir, sample_observation))
        errors.extend(smoke_test_beam_search(extract_dir, beam_sample))

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Kaggle submission tarball layout and agent import.")
    parser.add_argument("archive", nargs="?", default="dist/submission.tar.gz")
    parser.add_argument("--no-smoke-test", action="store_true")
    parser.add_argument(
        "--require-smoke-test",
        action="store_true",
        help="fail if agent smoke test cannot run (needs Python 3.10+)",
    )
    parser.add_argument(
        "--sample-data",
        default="outputs/rollouts/notebook_rollouts.jsonl",
        help="JSONL file used for agent and beam smoke tests",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    archive = Path(args.archive)
    if not archive.is_absolute():
        archive = root / archive
    sample_data = Path(args.sample_data)
    if not sample_data.is_absolute():
        sample_data = root / sample_data

    errors = validate_submission(
        archive,
        smoke_test=not args.no_smoke_test,
        sample_data=sample_data,
    )
    if args.require_smoke_test and sys.version_info < (3, 10):
        errors.append("agent smoke test requires Python 3.10+ (Kaggle evaluation uses 3.11+)")
    if errors:
        print(f"submission validation failed: {archive}", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        raise SystemExit(1)

    print(f"submission validation passed: {archive}")
    print("layout: main.py and deck.csv at archive root (Kaggle requirement)")
    if not args.no_smoke_test and sys.version_info >= (3, 10):
        print("smoke test: imported main.py and ran agent() on one rollout observation")
        print("beam smoke test: choose_action with search_begin_input returned legal indices")
    elif not args.no_smoke_test:
        print("smoke test: skipped locally (run on Python 3.11+ before upload if possible)")
        print("beam smoke test: skipped locally (requires Python 3.10+)")


if __name__ == "__main__":
    main()
