from __future__ import annotations

import json
import re
import tempfile
import unittest
import zipfile
from contextlib import contextmanager
from pathlib import Path

from poke_bot.ladder_replay import canonical_deck_sha256
try:
    from scripts.retrain_public_matchup_tree_teal_mask_v43 import (
        _catalog,
        _drop_zero_nnz_rows,
        _public_calibration_day,
    )
except ModuleNotFoundError as exc:
    if exc.name != "sklearn":
        raise
    _catalog = None
    _drop_zero_nnz_rows = None
    _public_calibration_day = None


def _require_router_dependencies() -> None:
    if (
        _catalog is None
        or _drop_zero_nnz_rows is None
        or _public_calibration_day is None
    ):
        raise unittest.SkipTest(
            "router fitting dependencies are installed in the Elmo worker image"
        )


def _archive_receipt(path: Path, day: str) -> dict:
    return {
        "format": "pokebot-authoritative-visual-day-receipt",
        "source_date": day,
        "source_archive": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": "sha256:" + ("a" * 64),
        },
    }


@contextmanager
def _raises(error_type: type[Exception], match: str):
    try:
        yield
    except error_type as exc:
        assert re.search(match, str(exc))
    else:
        raise AssertionError(f"{error_type.__name__} was not raised")


def test_public_calibration_day_uses_exact_deck_and_facing_public_view(
) -> None:
    _require_router_dependencies()
    temporary = tempfile.TemporaryDirectory()
    tmp_path = Path(temporary.name)
    day = "2026-07-04"
    target_deck = [96] * 60
    other_deck = [7] * 60
    public_observation = {
        "current": {
            "yourIndex": 1,
            "players": [
                {
                    "active": [{"id": 96}],
                    "bench": [{"id": 108}],
                    "discard": [],
                    "hand": [{"id": 999}],
                    "deck": [{"id": 998}],
                    "prizes": [{"id": 997}],
                },
                {"active": [], "bench": [], "discard": []},
            ],
        }
    }
    temporarily_empty_observation = {
        "current": {
            "yourIndex": 1,
            "players": [
                {"active": [], "bench": [], "discard": []},
                {"active": [], "bench": [], "discard": []},
            ],
        }
    }
    payload = {
        "id": "episode-exact-teal",
        "steps": [
            [
                {"action": target_deck, "observation": {"current": None}},
                {"action": other_deck, "observation": {"current": None}},
            ],
            [{}, {"observation": public_observation}],
            [{}, {"observation": public_observation}],
            [{}, {"observation": temporarily_empty_observation}],
        ],
    }
    archive = (
        tmp_path / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
    )
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("episode-exact-teal.json", json.dumps(payload))
    receipt = tmp_path / f"{day}.receipt.json"
    receipt.write_text(
        json.dumps(_archive_receipt(archive, day)) + "\n",
        encoding="utf-8",
    )

    result = _public_calibration_day(
        (
            day,
            archive,
            receipt,
            frozenset({canonical_deck_sha256(target_deck)}),
            18,
            4095,
            42,
            20,
        )
    )

    assert result["matching_episodes"] == 1
    assert result["matching_acting_seats"] == 1
    assert result["observations"] == 3
    assert len(result["states"]) == 1
    state, count, split = result["states"][0]
    assert state == (96, 108)
    assert count == 3
    assert split in (0, 1)
    assert 999 not in state
    assert 998 not in state
    assert 997 not in state
    assert result["source"]["archive_validation_receipt"] == str(receipt)
    temporary.cleanup()


def test_historical_zero_feature_rows_are_removed_before_fitting() -> None:
    _require_router_dependencies()
    import numpy as np
    from scipy import sparse

    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
            dtype=np.uint8,
        )
    )
    labels = np.asarray([2, 3, 4], dtype=np.int32)
    weights = np.asarray([5.0, 7.0, 11.0], dtype=np.float64)

    filtered_x, filtered_y, filtered_w, removed = _drop_zero_nnz_rows(
        [matrix], [labels], [weights]
    )

    assert filtered_x[0].shape == (1, 3)
    assert filtered_x[0].nnz == 1
    assert filtered_y[0].tolist() == [3]
    assert filtered_w[0].tolist() == [7.0]
    assert removed == {"rows": 2, "weighted_observations": 16.0}


def test_public_calibration_day_rejects_receipt_for_different_archive(
) -> None:
    _require_router_dependencies()
    temporary = tempfile.TemporaryDirectory()
    tmp_path = Path(temporary.name)
    day = "2026-07-04"
    archive = (
        tmp_path / f"pokemon-tcg-ai-battle-episodes-{day}.zip"
    )
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("episode.json", "{}")
    receipt_payload = _archive_receipt(archive, day)
    receipt_payload["source_archive"]["bytes"] += 1
    receipt = tmp_path / f"{day}.receipt.json"
    receipt.write_text(
        json.dumps(receipt_payload) + "\n",
        encoding="utf-8",
    )

    with _raises(
        RuntimeError,
        "checksum-validated archive receipt changed",
    ):
        _public_calibration_day(
            (
                day,
                archive,
                receipt,
                frozenset({"sha256:" + ("b" * 64)}),
                18,
                4095,
                42,
                20,
            )
        )
    temporary.cleanup()


def test_catalog_requires_complete_daily_window_and_count() -> None:
    _require_router_dependencies()
    temporary = tempfile.TemporaryDirectory()
    tmp_path = Path(temporary.name)
    catalog_path = tmp_path / "catalog.json"
    payload = {
        "schema": "poke_bot.public_deck_archetype_catalog/v1",
        "specialist_id": "teal-mask-ogerpon-ex",
        "source_archetype": {"id": 151, "name": "Teal Mask Ogerpon ex"},
        "source_window": {
            "start": "2026-07-04",
            "end": "2026-07-05",
            "days": 2,
        },
        "observed_acting_seat_games": 3,
        "observed_by_day": {
            "2026-07-04": 1,
            "2026-07-05": 2,
        },
        "deck_fingerprints": ["sha256:" + ("c" * 64)],
    }
    catalog_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    catalog, fingerprints, dates = _catalog(catalog_path)
    assert catalog["observed_acting_seat_games"] == 3
    assert fingerprints == frozenset({"sha256:" + ("c" * 64)})
    assert dates == ["2026-07-04", "2026-07-05"]

    payload["observed_by_day"].pop("2026-07-05")
    catalog_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with _raises(RuntimeError, "catalog identity changed"):
        _catalog(catalog_path)
    temporary.cleanup()
