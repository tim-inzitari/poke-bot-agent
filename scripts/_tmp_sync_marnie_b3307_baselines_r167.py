#!/usr/bin/env python3
"""Sync specialist-marnie-final-format-h10-b3307cf1bd67 to Bert/Elmo manifests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

SRC = Path(
    "/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v9/baselines/"
    "specialists/marnie-final-format-h10-b3307cf1bd67"
)
BASE_ID = "specialist-marnie-final-format-h10-b3307cf1bd67"
SHORT = "marnie-final-format-h10-b3307cf1bd67"
OLD = "f20efb20f5c3"
NEW = "b3307cf1bd67"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def run(cmd: list[str]) -> str:
    print("+", " ".join(cmd), flush=True)
    out = subprocess.check_output(cmd, text=True)
    return out


def rsync_to(host_spec: str, dest: str) -> None:
    run(
        [
            "rsync",
            "-az",
            "--delete",
            f"{SRC}/",
            f"{host_spec}:{dest.rstrip('/')}/",
        ]
    )


def rewrite_obj(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = k.replace(OLD, NEW) if isinstance(k, str) else k
            out[nk] = rewrite_obj(v)
        return out
    if isinstance(obj, list):
        return [rewrite_obj(x) for x in obj]
    if isinstance(obj, str):
        return obj.replace(OLD, NEW)
    return obj


def patch_manifest_text(text: str, digest: str) -> str:
    m = json.loads(text)
    changed = False

    def walk_containers(node):
        nonlocal changed
        if not isinstance(node, dict):
            return
        # Clone any f20ef entry in this dict into a b3307 entry.
        for k, v in list(node.items()):
            if OLD in str(k) or (isinstance(v, (dict, list, str)) and OLD in json.dumps(v)):
                if OLD in str(k) and isinstance(v, dict):
                    nk = str(k).replace(OLD, NEW)
                    nv = rewrite_obj(v)
                    if isinstance(nv, dict):
                        if "id" in nv:
                            nv["id"] = BASE_ID
                        if "sha256" in nv:
                            nv["sha256"] = digest
                        if "digest" in nv:
                            nv["digest"] = digest
                    node[nk] = nv
                    changed = True
            if isinstance(v, dict):
                walk_containers(v)

    walk_containers(m)

    # Ensure BASE_ID is present somewhere searchable.
    raw = json.dumps(m)
    if NEW not in raw:
        m.setdefault("baselines", {})
        if isinstance(m["baselines"], dict):
            m["baselines"][BASE_ID] = {
                "id": BASE_ID,
                "path": f"specialists/{SHORT}/model.pt",
                "sha256": digest,
                "archetype": "marnie-s-grimmsnarl-ex",
            }
            changed = True
        else:
            raise RuntimeError("cannot insert baseline; baselines is not a dict")

    # Force digest on any b3307 dict entries we can find.
    def force_digest(node):
        if isinstance(node, dict):
            if any(NEW in str(x) for x in node.keys()) or (
                isinstance(node.get("id"), str) and NEW in node["id"]
            ):
                if "sha256" in node:
                    node["sha256"] = digest
                if "digest" in node:
                    node["digest"] = digest
            for v in node.values():
                force_digest(v)
        elif isinstance(node, list):
            for v in node:
                force_digest(v)

    force_digest(m)
    if not changed and NEW not in text:
        raise RuntimeError("manifest patch made no changes")
    return json.dumps(m, indent=2, sort_keys=True) + "\n"


def remote_read(host_spec: str, path: str) -> str:
    return run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host_spec, f"cat {path}"])


def remote_write(host_spec: str, path: str, content: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "manifest.json"
        local.write_text(content, encoding="utf-8")
        run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", str(local), f"{host_spec}:{path}"])


def main() -> int:
    if not (SRC / "model.pt").is_file():
        raise SystemExit(f"missing source package {SRC}")
    digest = sha256_file(SRC / "model.pt")
    print("digest", digest)

    # Bert
    bert = "tsinzitari@192.168.1.158"
    bert_pkg = (
        "/Users/tsinzitari/workspace/poke-bot-agent-h10-r79-stage/baselines/"
        f"specialists/{SHORT}"
    )
    bert_manifest = (
        "/Users/tsinzitari/workspace/poke-bot-agent-h10-r79-stage/baselines/manifest.json"
    )
    run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            bert,
            f"mkdir -p {bert_pkg}",
        ]
    )
    rsync_to(bert, bert_pkg)
    patched = patch_manifest_text(remote_read(bert, bert_manifest), digest)
    remote_write(bert, bert_manifest, patched)
    assert NEW in patched
    print("bert ok")

    # Elmo baseline-sync
    elmo = "admin@192.168.1.143"
    elmo_pkg = (
        "/mnt/Main/main/poke-bot-agent/containers/truenas-worker/baseline-sync/"
        f"specialists/{SHORT}"
    )
    elmo_manifest = (
        "/mnt/Main/main/poke-bot-agent/containers/truenas-worker/baseline-sync/manifest.json"
    )
    run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=20",
            elmo,
            f"mkdir -p {elmo_pkg}",
        ]
    )
    rsync_to(elmo, elmo_pkg)
    patched = patch_manifest_text(remote_read(elmo, elmo_manifest), digest)
    remote_write(elmo, elmo_manifest, patched)
    assert NEW in patched
    print("elmo ok")

    # Recreate Elmo worker so it remounts/reloads baseline-sync
    compose = (
        "cd /mnt/Main/main/poke-bot-agent/containers/truenas-worker && "
        "sudo docker compose up -d --force-recreate worker"
    )
    # Prefer known compose service name from prior ops; fall back to container recreate.
    try:
        run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                elmo,
                "sudo docker compose -f /mnt/Main/main/poke-bot-agent/containers/truenas-worker/docker-compose.yml up -d --force-recreate",
            ]
        )
    except subprocess.CalledProcessError:
        run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=20",
                elmo,
                "sudo docker restart poke-bot-truenas-worker",
            ]
        )
    print("elmo worker refreshed")

    receipt = {
        "schema": "poke_bot.crustle_marnie_b3307_baseline_fleet_sync/v1",
        "baseline_id": BASE_ID,
        "digest": digest,
        "hosts": {
            "bert.stage": {"package": bert_pkg, "manifest": bert_manifest},
            "elmo.baseline_sync": {"package": elmo_pkg, "manifest": elmo_manifest},
        },
        "reason": "Crustle r167 public mix requires dual-Marnie baselines; b3307 missing from remote manifests",
    }
    out = Path(
        "/home/inzi/poke-bot-agent/outputs/state/crustle-h10-marnie-b3307-baseline-fleet-sync-r167.json"
    )
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("receipt", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
