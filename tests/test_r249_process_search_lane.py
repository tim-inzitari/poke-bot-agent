from __future__ import annotations

import os
import textwrap
import time
from pathlib import Path

import pytest

from poke_bot.r249_process_search_lane import (
    R249ProcessLaneError,
    R249ProcessSearchLane,
)


FAKE_CHILD = r'''
import argparse
import json
import os
import socket
import time

SCHEMA = "poke_bot.r249_process_search_lane/v1"

parser = argparse.ArgumentParser()
parser.add_argument("--child-fd", type=int, required=True)
parser.add_argument("--stage")
parser.add_argument("--lane-id", type=int, required=True)
args = parser.parse_args()
sock = socket.socket(fileno=args.child_fd)
stream = sock.makefile("rwb", buffering=0)
pid = os.getpid()

def send(row):
    stream.write((json.dumps(row, separators=(",", ":")) + "\n").encode())

send({
    "schema": SCHEMA,
    "type": "ready",
    "identity": {
        "pid": pid,
        "lane_id": args.lane_id,
        "platform": "test",
        "machine": "test",
        "native_member": "libcg.so",
        "native_sha256": "sha256:d16244a3157fc55c3314f08dcc7c5179168697d78c105b95c7debd556b764bb7",
        "raw_handle_identity": "fake",
        "handle_identity": f"process:{pid}:handle:fake",
    },
})
mode = os.environ.get("R249_FAKE_MODE", "normal")
next_id = 0
for raw in stream:
    request = json.loads(raw)
    kind = request["type"]
    if kind == "search_step" and mode == "hang_step":
        time.sleep(60)
    if kind == "search_step" and mode == "crash_step":
        os._exit(71)
    if kind == "search_release" and mode == "hang_release":
        time.sleep(60)
    row = {
        "schema": SCHEMA,
        "type": "result",
        "request_id": request["request_id"],
    }
    if kind in ("search_begin", "search_step"):
        row["state"] = {
            "search_id": next_id,
            "observation": {"select": {"option": [1, 2]}, "current": {"yourIndex": 0}},
        }
        next_id += 1
    send(row)
    if kind == "close":
        break
'''


def _stage(tmp_path: Path) -> Path:
    package = tmp_path / "stage"
    module = package / "fake_pkg/fake_lane_child.py"
    module.parent.mkdir(parents=True)
    (module.parent / "__init__.py").write_text("")
    module.write_text(textwrap.dedent(FAKE_CHILD))
    return package


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    return False


def _lane(stage: Path) -> R249ProcessSearchLane:
    return R249ProcessSearchLane(
        0,
        stage=stage,
        child_module="fake_pkg.fake_lane_child",
        startup_timeout_seconds=2.0,
        call_timeout_seconds=0.15,
        cleanup_timeout_seconds=0.10,
        reap_grace_seconds=0.10,
    )


def test_hung_native_step_reaps_exact_child_and_reopens_lane(monkeypatch, tmp_path):
    monkeypatch.setenv("R249_FAKE_MODE", "hang_step")
    lane = _lane(_stage(tmp_path))
    first_pid = lane.telemetry_snapshot()["child_identity"]["pid"]
    root = lane.search_begin(
        {"select": {"option": [1, 2]}, "current": {"yourIndex": 0}},
        {},
    )
    started = time.monotonic()
    with pytest.raises(R249ProcessLaneError, match="search_step failed"):
        lane.search_step(root.searchId, [0])
    assert time.monotonic() - started < 1.5
    assert _gone(first_pid)
    assert lane.faults[-1]["operation"] == "search_step"
    assert lane.faults[-1]["reap"]["reaped"] is True

    monkeypatch.setenv("R249_FAKE_MODE", "normal")
    reopened = lane.search_begin(
        {"select": {"option": [1, 2]}, "current": {"yourIndex": 0}},
        {},
    )
    second = lane.telemetry_snapshot()["child_identity"]
    assert second["pid"] != first_pid
    assert lane.handle_identity == f"process:{second['pid']}:handle:fake"
    assert reopened.searchId == 0
    lane.close()
    assert _gone(second["pid"])


def test_cleanup_hang_is_contained_and_cannot_block_close_row(monkeypatch, tmp_path):
    monkeypatch.setenv("R249_FAKE_MODE", "hang_release")
    lane = _lane(_stage(tmp_path))
    pid = lane.telemetry_snapshot()["child_identity"]["pid"]
    root = lane.search_begin(
        {"select": {"option": [1, 2]}, "current": {"yourIndex": 0}},
        {},
    )
    started = time.monotonic()
    lane.search_release(root.searchId)
    lane.search_end()
    assert time.monotonic() - started < 1.5
    assert _gone(pid)
    assert lane.faults[-1]["operation"] == "search_release"
    lane.close()


def test_native_child_crash_is_bounded_and_receipted(monkeypatch, tmp_path):
    monkeypatch.setenv("R249_FAKE_MODE", "crash_step")
    lane = _lane(_stage(tmp_path))
    pid = lane.telemetry_snapshot()["child_identity"]["pid"]
    root = lane.search_begin(
        {"select": {"option": [1, 2]}, "current": {"yourIndex": 0}},
        {},
    )
    with pytest.raises(R249ProcessLaneError, match="search_step failed"):
        lane.search_step(root.searchId, [0])
    assert _gone(pid)
    assert lane.faults[-1]["reap"]["reaped"] is True
    lane.close()
