#!/usr/bin/env python3
"""Resident GOAL.md / Crustle wake handler for AGENT_LOOP_WAKE_goalmd."""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

ROOT = Path("/Users/tsinzitari/Documents/poke-agent-codex")
STATE = ROOT / ".cursor-loop/goalmd_loop_state.json"
HB_LOG = ROOT / ".cursor-loop/goalmd_heartbeat.log"
WATCH_LOG = ROOT / ".cursor-loop/goalmd_watcher.log"
TERM_DIR = Path(
    "/Users/tsinzitari/.cursor/projects/"
    "Users-tsinzitari-Documents-poke-agent-codex/terminals"
)
# Updated when a new device login terminal is armed; may be overridden via state.
TERM_GH = TERM_DIR / "463088.txt"
SENTINEL = "AGENT_LOOP_WAKE_goalmd"
SELF_TERM_MARKERS = ("goalmd_resident_loop.py", "Run fixed resident GOAL.md loop")
ACTIVE_DEVICE_CODE = "CE54-CF8E"
PROMPT = (
    "Monitor GOAL.md Crustle: read revision/objective; audit crustle systemd "
    "units on train; keep dashboard aligned to runtime truth; leave healthy "
    "training alone; coordinate with sibling recovery via live state; do not "
    "rewrite GOAL.md or disrupt git/PR work."
)
HB_CMD = (
    "sleep 900; echo '"
    + SENTINEL
    + ' {"prompt":"'
    + PROMPT
    + '","reason":"fallback_heartbeat_900s"}\''
)
SSH = [
    "ssh",
    "-o",
    "BatchMode=yes",
    "-i",
    str(Path.home() / ".ssh/id_ed25519_poke_lan"),
    "-o",
    "IdentitiesOnly=yes",
    "train",
]


def alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def pgrep_af(pat: str) -> list[str]:
    try:
        return subprocess.check_output(["pgrep", "-af", pat], text=True).strip().splitlines()
    except subprocess.CalledProcessError:
        return []


def wake_count(path: Path) -> int:
    if not path.exists():
        return 0
    text = path.read_text(errors="replace")
    body = text.split("---", 2)[-1] if text.startswith("---") else text
    return sum(1 for ln in body.splitlines() if ln.startswith(SENTINEL))


def last_wake(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(errors="replace")
    body = text.split("---", 2)[-1] if text.startswith("---") else text
    lines = [ln for ln in body.splitlines() if ln.startswith(SENTINEL)]
    return lines[-1] if lines else ""


def load_state() -> dict:
    return json.loads(STATE.read_text())


def save_state(st: dict) -> None:
    st["updated_at_unix"] = int(time.time())
    STATE.write_text(json.dumps(st, indent=2) + "\n")


def gh_status() -> tuple[bool, str]:
    r = subprocess.run(
        ["gh", "auth", "status", "-h", "github.com"], capture_output=True, text=True
    )
    out = (r.stdout or "") + (r.stderr or "")
    ok = ("Logged in to github.com" in out) and ("Failed to log in" not in out)
    return ok, out


def _terminal_header(text: str) -> str:
    # Cursor terminal files: ---\nmeta\n---\nbody
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            return parts[1]
    return text[:800]


def _active_gh_term() -> Path:
    st = load_state()
    gh = st.get("gh_auth") or {}
    term = gh.get("started_terminal")
    if term:
        p = TERM_DIR / str(term)
        if p.exists():
            return p
    # Prefer newest running gh auth login terminal.
    newest = None
    newest_mt = -1.0
    if TERM_DIR.exists():
        for p in TERM_DIR.glob("*.txt"):
            try:
                t = p.read_text(errors="replace")
            except Exception:
                continue
            if "gh auth login" not in t or "one-time code" not in t:
                continue
            head = _terminal_header(t)
            if re.search(r"(?m)^exit_code:", head):
                continue
            mt = p.stat().st_mtime
            if mt > newest_mt:
                newest_mt = mt
                newest = p
    return newest or TERM_GH


def gh_device_still_pending() -> bool:
    """True while the active device flow is still waiting."""
    term = _active_gh_term()
    if not term.exists():
        return False
    t = term.read_text(errors="replace")
    head = _terminal_header(t)
    # Body only — the command field may mention "expired" from prior codes.
    body = t
    if t.startswith("---"):
        parts = t.split("---", 2)
        if len(parts) >= 3:
            body = parts[2]
    if re.search(r"(?i)Logged in as|Authentication complete", body):
        return False
    if re.search(r"(?i)context deadline exceeded|one-time code.*(expired|invalid)", body):
        return False
    if re.search(r"(?m)^exit_code:", head):
        return False
    running = ("status: running" in head) or ("running_for_ms" in head)
    code = (load_state().get("gh_auth") or {}).get("code") or ACTIVE_DEVICE_CODE
    return code in body and running


def finish_push_if_authed() -> bool:
    ok, _ = gh_status()
    if not ok:
        code = (load_state().get("gh_auth") or {}).get("code") or ACTIVE_DEVICE_CODE
        print(f"GH_STILL_PENDING code={code}", flush=True)
        return False
    print("GH_BECAME_OK", flush=True)
    r = subprocess.run(
        ["git", "push", "-u", "origin", "HEAD"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    print("PUSH_RC", r.returncode, flush=True)
    print((r.stdout or "")[-800:], flush=True)
    print((r.stderr or "")[-800:], flush=True)
    st = load_state()
    if r.returncode == 0:
        br = subprocess.run(
            ["git", "status", "-sb"], cwd=str(ROOT), capture_output=True, text=True
        )
        line = (br.stdout or "").splitlines()[0] if br.stdout else ""
        print("ORIGIN_TRACK", line, flush=True)
        code = (st.get("gh_auth") or {}).get("code") or ACTIVE_DEVICE_CODE
        st["gh_auth"] = {
            "status": "pushed",
            "code": code,
            "at": int(time.time()),
            "status_sb": line,
        }
        save_state(st)
        return True
    code = (st.get("gh_auth") or {}).get("code") or ACTIVE_DEVICE_CODE
    st["gh_auth"] = {
        "status": "auth_ok_push_failed",
        "code": code,
        "at": int(time.time()),
        "push_rc": r.returncode,
    }
    save_state(st)
    return False


def audit() -> dict:
    cmd = r"""python3 - <<'PY'
import json, shutil, subprocess, time
from pathlib import Path
free = shutil.disk_usage("/home/inzi").free / 1024**3
rl = subprocess.check_output(
    ["systemctl","--user","show","pokebot-final-format-crustle-r113-h10-rl.service",
     "-p","ActiveState","-p","SubState","-p","MainPID","-p","NRestarts","-p","Result","--no-pager"],
    text=True,
)
rtp = subprocess.check_output(
    ["systemctl","--user","show","pokebot-crustle-rtp-cotrain-r167.service",
     "-p","ActiveState","-p","MainPID","-p","NRestarts","--no-pager"],
    text=True,
)
ms = Path("/home/inzi/poke-bot-agent/outputs/pure_rl/final_format_crustle_r113_h10_i_v6_8k/monitor_state.json")
d = json.loads(ms.read_text())
line = ((d.get("progress") or {}).get("last_line") or "")[:200]
goal_rev = "?"
for ln in Path("/home/inzi/poke-bot-agent/GOAL.md").read_text().splitlines()[:10]:
    if ln.startswith("Revision:"):
        goal_rev = ln.split(":", 1)[1].strip().strip("`")
print(json.dumps({
    "free_GiB": round(free, 2),
    "margin_to_100": round(free - 100, 2),
    "rl": {k: v for k, v in (x.split("=", 1) for x in rl.strip().splitlines() if "=" in x)},
    "rtp": {k: v for k, v in (x.split("=", 1) for x in rtp.strip().splitlines() if "=" in x)},
    "line": line,
    "mon_alive": d.get("process_alive"),
    "goal_rev": goal_rev,
}, sort_keys=True))
PY"""
    r = subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=90)
    raw = (r.stdout or r.stderr or "").strip()
    print("AUDIT", raw[-1200:], flush=True)
    try:
        data = json.loads(raw.splitlines()[-1])
    except Exception:
        data = {"raw": raw[-400:]}
    st = load_state()
    rl = data.get("rl") or {}
    rtp = data.get("rtp") or {}
    st["last_pass_summary"] = {
        "healthy": rl.get("ActiveState") == "active",
        "rl": f"{rl.get('ActiveState')}/{rl.get('MainPID')}",
        "rtp": f"{rtp.get('ActiveState')}/{rtp.get('MainPID')}",
        "nrestarts": rl.get("NRestarts"),
        "disk_free_GiB": data.get("free_GiB"),
        "goal_rev": data.get("goal_rev"),
        "line": (data.get("line") or "")[:160],
        "action": "leave_alone",
        "at": int(time.time()),
    }
    st["pass_count"] = int(st.get("pass_count") or 0) + 1
    st["resident"] = True
    save_state(st)
    free = data.get("free_GiB")
    if isinstance(free, (int, float)) and free < 120:
        print("DISK_WARNING free_GiB=", free, flush=True)
    if isinstance(free, (int, float)) and free < 100:
        print("DISK_BLOCKER free_GiB=", free, flush=True)
    return data


def _ps_pid_cmd_map() -> dict[str, tuple[str, str]]:
    """pid -> (ppid, command)."""
    out: dict[str, tuple[str, str]] = {}
    for ln in subprocess.check_output(
        ["ps", "-ax", "-o", "pid=,ppid=,command="], text=True
    ).splitlines():
        parts = ln.strip().split(None, 2)
        if len(parts) < 3:
            continue
        out[parts[0]] = (parts[1], parts[2])
    return out


def sleep_900_alive() -> list[str]:
    """Return goalmd heartbeat `sleep 900` only (not other loops' sleeps)."""
    pid_map = _ps_pid_cmd_map()
    lines = []
    for pid, (ppid, cmd) in pid_map.items():
        c = cmd.strip()
        if not (c == "sleep 900" or c.startswith("sleep 900 ")):
            continue
        parent_cmd = pid_map.get(ppid, ("", ""))[1]
        # Dashboard/other loops also use exact `sleep 900`; bind to our sentinel.
        if "AGENT_LOOP_WAKE_goalmd" not in parent_cmd and "goalmd_heartbeat" not in parent_cmd:
            continue
        lines.append(f"{pid} {ppid} {c}")
    return lines


def rearm_heartbeat(st: dict) -> int | None:
    if sleep_900_alive():
        return None
    log = ROOT / ".cursor-loop/goalmd_heartbeat.log"
    with open(log, "ab") as lf:
        proc = subprocess.Popen(
            ["/bin/zsh", "-c", HB_CMD],
            stdout=lf,
            stderr=lf,
            start_new_session=True,
            cwd=str(ROOT),
        )
    st["heartbeat_shell_pid"] = proc.pid
    print("REARMED_HEARTBEAT", proc.pid, flush=True)
    return proc.pid


def ensure_loop() -> dict:
    st = load_state()
    if not pgrep_af("goalmd_crustle_watcher.py"):
        print("REARM_WATCHER", flush=True)
        sh = ROOT / ".cursor-loop/goalmd_crustle_watcher.sh"
        log = ROOT / ".cursor-loop/goalmd_watcher.log"
        with open(log, "ab") as lf:
            proc = subprocess.Popen(
                ["/bin/zsh", "-c", str(sh)],
                stdout=lf,
                stderr=lf,
                start_new_session=True,
                cwd=str(ROOT),
            )
        st["watcher_shell_pid"] = proc.pid
    else:
        pys = []
        for ln in pgrep_af("goalmd_crustle_watcher.py"):
            pid = ln.split(None, 1)[0]
            if pid.isdigit():
                pys.append(int(pid))
        if pys:
            st["watcher_shell_pid"] = pys[0]
            st["watcher_python_pids"] = pys
    rearm_heartbeat(st)
    # refresh hb pid from live sleeper if present
    for ln in sleep_900_alive():
        pid = int(ln.split(None, 1)[0])
        st["heartbeat_shell_pid"] = pid
        break
    st["resident"] = True
    st["parent_must_resume_on_wake"] = False
    st["fallback_heartbeat_seconds"] = 900
    save_state(st)
    return st


def baseline_counts() -> dict[str, int]:
    base: dict[str, int] = {}
    for p in [HB_LOG, WATCH_LOG]:
        if p.exists():
            base[str(p)] = wake_count(p)
    if TERM_DIR.exists():
        for p in TERM_DIR.glob("*.txt"):
            try:
                head = _terminal_header(p.read_text(errors="replace"))
            except Exception:
                head = ""
            # Still record count so later growth is measured, but wake handler skips self.
            base[str(p)] = wake_count(p)
            if any(m in head for m in SELF_TERM_MARKERS):
                continue
    return base


def main() -> None:
    st = load_state()
    st["resident"] = True
    st["parent_must_resume_on_wake"] = False
    st["rearm_note"] = (
        "resident goalmd_resident_loop.py handling AGENT_LOOP_WAKE_goalmd; "
        "no Crustle restarts; gh device 113C-D5CF reused until expiry"
    )
    save_state(st)
    ensure_loop()
    print("RESIDENT_START", flush=True)
    audit()
    finish_push_if_authed()

    base = baseline_counts()
    print(
        "BASE_INIT paths=",
        len(base),
        "with_wakes=",
        sum(1 for v in base.values() if v),
        flush=True,
    )

    last_gh = 0.0
    last_ensure = 0.0
    last_auth_note = 0.0
    # Stay resident indefinitely (or until killed). Outer notify watches stdout.
    while True:
        now = time.time()
        if now - last_ensure >= 60:
            last_ensure = now
            ensure_loop()
        if now - last_gh >= 60:
            last_gh = now
            ok, _ = gh_status()
            if ok:
                finish_push_if_authed()
            elif now - last_auth_note >= 600:
                # Leave device-login to the user; do not start new codes.
                last_auth_note = now
                st = load_state()
                gh = st.get("gh_auth") or {}
                print(
                    "AUTH_WAITING_USER status=",
                    gh.get("status", "unknown"),
                    "no_new_device_flow",
                    flush=True,
                )
                gh["status"] = "waiting_user"
                st["gh_auth"] = gh
                save_state(st)

        # Refresh known paths; only NEW increments count as wakes.
        if TERM_DIR.exists():
            for p in TERM_DIR.glob("*.txt"):
                sp = str(p)
                if sp not in base:
                    # Skip self/echo terminals so printing WAKE lines does not re-wake.
                    try:
                        head = _terminal_header(p.read_text(errors="replace"))
                    except Exception:
                        head = ""
                    if any(m in head for m in SELF_TERM_MARKERS):
                        base[sp] = wake_count(p)
                        continue
                    base[sp] = wake_count(p)
        for sp, prev in list(base.items()):
            path = Path(sp)
            try:
                head = _terminal_header(path.read_text(errors="replace"))
            except Exception:
                head = ""
            if any(m in head for m in SELF_TERM_MARKERS):
                base[sp] = wake_count(path)
                continue
            cur = wake_count(path)
            if cur > prev:
                # Avoid echoing the sentinel token (prevents self-trigger in this terminal).
                lw = last_wake(path)
                reason = "unknown"
                m = re.search(r'"reason"\s*:\s*"([^"]+)"', lw)
                if m:
                    reason = m.group(1)
                print("WAKE", path.name, f"{prev}->{cur}", "reason=", reason, flush=True)
                base[sp] = cur
                audit()
                # Re-arm heartbeat after consuming a wake (dynamic loop).
                st = load_state()
                rearm_heartbeat(st)
                save_state(st)
                print("WAKE_HANDLED_CONTINUE", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
