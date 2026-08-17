from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.run_r274_rl_after_handoff import (
    DEFAULT_REMOTE_WORKER_ENDPOINTS,
    _adapter_training_contract,
    _remote_worker_endpoints,
)


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _receipt(tmp_path: Path) -> Path:
    authorization = tmp_path / "adapter-authorization.json"
    authorization.write_text('{"schema":"test"}\n', encoding="utf-8")
    receipt = tmp_path / "preservation.json"
    receipt.write_text(
        json.dumps(
            {
                "matchup_adapter": {
                    "epochs_per_rl_update": 1,
                    "activation_provenance": {
                        "matchup_adapter_bank_preserved": True,
                        "matchup_adapter_training_enabled": True,
                        "matchup_adapter_isolated_fit_continuation_required": True,
                        "matchup_adapter_isolated_bank_only_optimizer": True,
                    },
                    "training_activation": {
                        "path": str(authorization),
                        "sha256": _digest(authorization),
                        "size_bytes": authorization.stat().st_size,
                    },
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt


def test_r274_resolves_exact_one_epoch_adapter_continuation(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    epochs, authorization = _adapter_training_contract(receipt)
    assert epochs == 1
    assert authorization == tmp_path / "adapter-authorization.json"


def test_r274_rejects_changed_adapter_authorization(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    (tmp_path / "adapter-authorization.json").write_text(
        '{"schema":"changed"}\n', encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="continuation contract is invalid"):
        _adapter_training_contract(receipt)


def test_r274_rejects_disabled_adapter_continuation(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["matchup_adapter"]["epochs_per_rl_update"] = 0
    receipt.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="continuation contract is invalid"):
        _adapter_training_contract(receipt)


def test_r274_requires_additive_remote_worker_farms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PURE_RL_REMOTE_WORKER_ENDPOINTS", raising=False)
    assert _remote_worker_endpoints() == DEFAULT_REMOTE_WORKER_ENDPOINTS


def test_r274_rejects_empty_remote_worker_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PURE_RL_REMOTE_WORKER_ENDPOINTS", "  ")
    with pytest.raises(RuntimeError, match="requires nonempty remote worker"):
        _remote_worker_endpoints()
