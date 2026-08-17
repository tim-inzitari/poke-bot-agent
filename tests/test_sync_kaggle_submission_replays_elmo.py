from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_kaggle_submission_replays_elmo.py"


def _load_sync_module() -> ModuleType:
    """Load the standalone sync script without requiring real Kaggle credentials."""

    kaggle = ModuleType("kaggle")
    kaggle_api = ModuleType("kaggle.api")
    kaggle_extended = ModuleType("kaggle.api.kaggle_api_extended")
    kaggle_extended.KaggleApi = object
    requests = ModuleType("requests")

    class HTTPError(Exception):
        def __init__(self, *args, response=None):
            super().__init__(*args)
            self.response = response

    requests.HTTPError = HTTPError
    previous = {
        name: sys.modules.get(name)
        for name in (
            "kaggle",
            "kaggle.api",
            "kaggle.api.kaggle_api_extended",
            "requests",
        )
    }
    try:
        sys.modules["kaggle"] = kaggle
        sys.modules["kaggle.api"] = kaggle_api
        sys.modules["kaggle.api.kaggle_api_extended"] = kaggle_extended
        sys.modules["requests"] = requests
        spec = importlib.util.spec_from_file_location("submission_replay_sync", SCRIPT)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, value in previous.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def test_default_discovery_preserves_floor_and_unions_special_submission(
    monkeypatch,
) -> None:
    sync = _load_sync_module()
    calls: list[list[str]] = []

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        return SimpleNamespace(
            returncode=0,
            stderr="",
            stdout=(
                "ref,description\n"
                "55217604,historical special\n"
                "55315273,older ordinary submission\n"
                "55324802,current submission\n"
            ),
        )

    monkeypatch.setattr(sync.subprocess, "run", fake_run)

    selected = sync._list_team_submission_ids(
        sync.DEFAULT_COMPETITION,
        sync.DEFAULT_MIN_SUBMISSION_ID,
    )

    assert selected == [55217604, 55315274, 55324802]
    assert calls == [
        [
            "kaggle",
            "competitions",
            "submissions",
            sync.DEFAULT_COMPETITION,
            "-v",
            "--page-size",
            "200",
        ]
    ]


def test_default_selection_rechecks_special_case_on_each_discovery(monkeypatch) -> None:
    sync = _load_sync_module()
    responses = iter(
        (
            "ref,description\n55324802,current submission\n",
            "ref,description\n55324802,current submission\n55330000,new submission\n",
        )
    )
    calls = 0

    def fake_run(_argv, **_kwargs):
        nonlocal calls
        calls += 1
        return SimpleNamespace(returncode=0, stderr="", stdout=next(responses))

    monkeypatch.setattr(sync.subprocess, "run", fake_run)

    first = sync._select_submission_ids(
        sync.DEFAULT_COMPETITION,
        sync.DEFAULT_MIN_SUBMISSION_ID,
        None,
    )
    second = sync._select_submission_ids(
        sync.DEFAULT_COMPETITION,
        sync.DEFAULT_MIN_SUBMISSION_ID,
        None,
    )

    assert calls == 2
    assert 55217604 in first
    assert 55217604 in second
    assert 55330000 not in first
    assert 55330000 in second


def test_explicit_cli_override_does_not_auto_add_special_case(monkeypatch) -> None:
    sync = _load_sync_module()

    def unexpected_discovery(*_args, **_kwargs):
        raise AssertionError("explicit --submission must not invoke discovery")

    monkeypatch.setattr(sync, "_list_team_submission_ids", unexpected_discovery)

    assert sync._select_submission_ids(
        sync.DEFAULT_COMPETITION,
        sync.DEFAULT_MIN_SUBMISSION_ID,
        [55324802],
    ) == [55324802]
    assert (
        sync._select_submission_ids(
            sync.DEFAULT_COMPETITION,
            sync.DEFAULT_MIN_SUBMISSION_ID,
            [55217604],
        )
        == []
    )
    assert sync._select_submission_ids(
        sync.DEFAULT_COMPETITION,
        55217604,
        [55217604],
    ) == [55217604]


def test_kaggle_call_timeout_is_bounded_and_not_retried() -> None:
    sync = _load_sync_module()
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def blocked_call() -> None:
        nonlocal calls
        calls += 1
        started.set()
        release.wait(timeout=5)

    try:
        with pytest.raises(sync.KaggleCallTimeout, match="timed out after"):
            sync._retry_http(blocked_call, timeout_s=0.01)
        assert started.is_set()
        assert calls == 1
    finally:
        release.set()


def test_kaggle_call_returns_and_retries_only_http_429(monkeypatch) -> None:
    sync = _load_sync_module()
    sleeps: list[float] = []
    calls = 0

    def rate_limited_then_ready() -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            response = SimpleNamespace(status_code=429, headers={"Retry-After": "0"})
            raise sync.HTTPError(response=response)
        return "ready"

    monkeypatch.setattr(sync.time, "sleep", sleeps.append)

    assert (
        sync._retry_http(rate_limited_then_ready, attempts=2, timeout_s=0.1) == "ready"
    )
    assert calls == 2
    assert sleeps == [1.0]


def test_download_timeout_writes_partial_receipt_without_blocking_workers(
    tmp_path: Path,
) -> None:
    sync = _load_sync_module()
    started = threading.Event()
    release = threading.Event()

    class FakeApi:
        @staticmethod
        def competition_list_episodes(_submission_id: int):
            return [
                {
                    "id": 89781273,
                    "state": "COMPLETED",
                    "type": "PUBLIC",
                    "agents": [
                        {
                            "index": 0,
                            "reward": 1,
                            "submission_id": 55217604,
                            "team_id": 1,
                            "team_name": "owner",
                        },
                        {
                            "index": 1,
                            "reward": 0,
                            "submission_id": 999,
                            "team_id": 2,
                            "team_name": "opponent",
                        },
                    ],
                }
            ]

        @staticmethod
        def competition_episode_replay(_episode_id: int, _destination: str) -> None:
            started.set()
            release.wait(timeout=5)

    try:
        began = time.monotonic()
        summary = sync._sync_submission(
            FakeApi(),
            tmp_path,
            55217604,
            workers=1,
            loss_logs=False,
            api_timeout_s=0.01,
        )
        elapsed = time.monotonic() - began
    finally:
        release.set()

    receipt = json.loads(
        (tmp_path / "55217604" / "SYNC_RECEIPT.json").read_text(encoding="utf-8")
    )
    assert started.is_set()
    assert elapsed < 1.0
    assert summary["status"] == "partial"
    assert summary["error_count"] == 1
    assert receipt["status"] == "partial"
    assert len(receipt["errors"]) == 1
    assert "episode 89781273 replay download" in receipt["errors"][0]
    assert "timed out after" in receipt["errors"][0]
