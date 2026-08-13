from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

import pytest

from scripts.stage_r274_bootstrap_submission import (
    _audit_archive,
    _deck_receipt,
)


def _write_bundle(path: Path, files: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            bundle.addfile(info, io.BytesIO(body))


def _sha(body: bytes) -> str:
    import hashlib

    return "sha256:" + hashlib.sha256(body).hexdigest()


def test_direct_bootstrap_archive_requires_search_assets_absent(tmp_path: Path) -> None:
    model = b"model"
    deck = b"deck"
    tree = b"tree"
    libcg = b"cg"
    archive = tmp_path / "submission.tar.gz"
    required = {
        "model.pt": model,
        "deck.csv": deck,
        "matchup_tree.json": tree,
        "cg/libcg.so": libcg,
        "main.py": b"pass\n",
        "runtime_profile.json": b"{}\n",
        "turn_order_profile.json": b"{}\n",
    }
    _write_bundle(archive, required)

    result = _audit_archive(
        archive,
        checkpoint_sha256=_sha(model),
        deck_sha256=_sha(deck),
        matchup_tree_sha256=_sha(tree),
        official_libcg_sha256=_sha(libcg),
    )
    assert result["passed"] is True
    assert result["checks"]["search_config_absent"] is True
    assert result["checks"]["belief_decks_absent"] is True

    required["search_config.json"] = b"{}\n"
    _write_bundle(archive, required)
    with pytest.raises(RuntimeError, match="search_config_absent"):
        _audit_archive(
            archive,
            checkpoint_sha256=_sha(model),
            deck_sha256=_sha(deck),
            matchup_tree_sha256=_sha(tree),
            official_libcg_sha256=_sha(libcg),
        )


def test_canonical_r274_deck_receipt_matches_typed_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    contract_path = root / "state/alakazam-new-list-direct-policy-r241.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))

    receipt = _deck_receipt(contract, contract_path)

    assert receipt["cards"] == 60
    assert receipt["file_sha256"] == contract["exact_deck"]["file_sha256"]
    assert receipt["cards_sha256"] == contract["exact_deck"]["ordered_cards_sha256"]
