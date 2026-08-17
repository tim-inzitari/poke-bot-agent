#!/usr/bin/env python3
"""Archive every exact accepted Kaggle submission package on Elmo's NAS.

The upload queue is the submission-ID authority.  This tool copies only bundles
whose current bytes match the queue's SHA-256, then asks Elmo to extract and
verify the packaged model and matchup tree into content-addressed directories.
It never guesses identity from filenames and never overwrites conflicting
artifacts or evidence.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shlex
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

QUEUE_SCHEMA = "poke_bot.kaggle_submission_queue/v1"
DEFAULT_QUEUE = Path(
    "/home/pokebot/poke-bot-agent/outputs/state/kaggle-submission-queue.json"
)
DEFAULT_REMOTE_ROOT = Path(
    "/srv/poke-bot-agent/archive/replay-model-inspector"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _digest(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:") and len(text) == 71:
        try:
            bytes.fromhex(text[7:])
        except ValueError:
            return None
        return text
    return None


def accepted_entries(
    payload: Mapping[str, Any], *, minimum_id: int, maximum_id: int | None
) -> list[dict[str, Any]]:
    """Return exact accepted queue rows in the requested inclusive range."""

    if payload.get("schema") != QUEUE_SCHEMA:
        raise ValueError("submission queue schema changed")
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    for raw in payload.get("queue") or []:
        if not isinstance(raw, Mapping) or raw.get("queue_status") != "accepted":
            continue
        submission_id = raw.get("submission_id")
        if (
            isinstance(submission_id, bool)
            or not isinstance(submission_id, int)
            or submission_id < minimum_id
            or (maximum_id is not None and submission_id > maximum_id)
        ):
            continue
        if submission_id in seen:
            raise ValueError(f"duplicate accepted submission id {submission_id}")
        required = {
            "bundle": _digest(raw.get("file_sha256")),
            "checkpoint": _digest(raw.get("checkpoint_checksum")),
            "model": _digest(raw.get("model_checksum")),
            "matchup_tree": _digest(raw.get("matchup_tree_checksum")),
        }
        if any(value is None for value in required.values()):
            raise ValueError(f"submission {submission_id} lacks exact digests")
        if required["checkpoint"] != required["model"]:
            raise ValueError(f"submission {submission_id} model/checkpoint mismatch")
        file_path = Path(str(raw.get("file") or "")).expanduser().resolve()
        if not file_path.is_file():
            raise ValueError(f"submission {submission_id} bundle is missing")
        if _sha256(file_path) != required["bundle"]:
            raise ValueError(f"submission {submission_id} bundle digest mismatch")
        row = dict(raw)
        row["file"] = str(file_path)
        row["archive_digests"] = required
        selected.append(row)
        seen.add(submission_id)
    return sorted(selected, key=lambda row: int(row["submission_id"]))


REMOTE_INSTALL = r"""
import base64, hashlib, json, os, pathlib, shutil, sys, tarfile

payload=json.loads(base64.b64decode(sys.argv[1]).decode("utf-8"))
root=pathlib.Path(payload["root"]).resolve()
expected=pathlib.Path("/srv/poke-bot-agent/archive/replay-model-inspector")
if root != expected:
    raise SystemExit("unexpected archive root")

def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return "sha256:"+h.hexdigest()

sid=int(payload["submission_id"])
bundle_sha=payload["bundle_sha256"]
checkpoint_sha=payload["checkpoint_sha256"]
tree_sha=payload["matchup_tree_sha256"]
incoming=pathlib.Path(payload["incoming"]).resolve()
canonical_bundle=(root/"artifacts"/"bundles"/("sha256-"+bundle_sha[7:])/"submission.tar.gz").resolve()
if (incoming.parent != (root/".incoming").resolve() and incoming != canonical_bundle) or digest(incoming) != bundle_sha:
    raise SystemExit("incoming bundle failed containment or digest verification")

bundle_dir=root/"artifacts"/"bundles"/("sha256-"+bundle_sha[7:])
bundle_path=bundle_dir/"submission.tar.gz"
bundle_dir.mkdir(parents=True,exist_ok=True)
if bundle_path.exists():
    if digest(bundle_path) != bundle_sha:
        raise SystemExit("existing bundle conflicts with accepted queue identity")
    if incoming != bundle_path:
        incoming.unlink()
else:
    os.replace(incoming,bundle_path)
    os.chmod(bundle_path,0o444)

runtime_parent=root/"artifacts"/"runtimes"/("sha256-"+bundle_sha[7:])
runtime=runtime_parent/"package"
if not runtime.exists():
    runtime_parent.mkdir(parents=True,exist_ok=True)
    temporary=runtime_parent/(".package.%d.%d.tmp" % (sid,os.getpid()))
    temporary.mkdir(mode=0o755)
    with tarfile.open(bundle_path,"r:gz") as archive:
        for member in archive.getmembers():
            name=member.name.removeprefix("./")
            target=(temporary/name).resolve()
            contained=target == temporary.resolve() or temporary.resolve() in target.parents
            if member.issym() or member.islnk() or not contained:
                raise SystemExit("unsafe submitted bundle member")
        archive.extractall(temporary,filter="data")
    os.replace(temporary,runtime)

model=runtime/"model.pt"
tree=runtime/"matchup_tree.json"
if digest(model) != checkpoint_sha or digest(tree) != tree_sha:
    raise SystemExit("extracted runtime disagrees with accepted queue identity")

checkpoint_dir=root/"artifacts"/"checkpoints"/("sha256-"+checkpoint_sha[7:])
checkpoint=checkpoint_dir/"model.pt"
checkpoint_dir.mkdir(parents=True,exist_ok=True)
if checkpoint.exists():
    if digest(checkpoint) != checkpoint_sha:
        raise SystemExit("existing checkpoint conflicts with accepted queue identity")
else:
    try:
        os.link(model,checkpoint)
    except OSError:
        shutil.copy2(model,checkpoint)
    os.chmod(checkpoint,0o444)

evidence={
  "schema":"poke_bot.replay_model_inspector_submission_evidence/v1",
  "submission_id":sid,
  "label":payload["label"],
  "specialist_id":payload["specialist_id"],
  "checkpoint":{"path":"/data/inspector/artifacts/checkpoints/sha256-"+checkpoint_sha[7:]+"/model.pt","sha256":checkpoint_sha},
  "bundle":{"path":"/data/inspector/artifacts/bundles/sha256-"+bundle_sha[7:]+"/submission.tar.gz","sha256":bundle_sha},
  "matchup_tree":{"path":"/data/inspector/artifacts/runtimes/sha256-"+bundle_sha[7:]+"/package/matchup_tree.json","sha256":tree_sha},
  "runtime_package":{"path":"/data/inspector/artifacts/bundles/sha256-"+bundle_sha[7:]+"/submission.tar.gz","sha256":bundle_sha},
  "selection_authority":"accepted_queue_submission_id_plus_exact_uploaded_bundle",
  "training_eligible":False,
  "read_only":True,
}
evidence_dir=root/"provenance"/"submission-evidence"
evidence_dir.mkdir(parents=True,exist_ok=True)
evidence_path=evidence_dir/(str(sid)+".json")
encoded=(json.dumps(evidence,indent=2,sort_keys=True)+"\n").encode("utf-8")
if evidence_path.exists() and evidence_path.read_bytes()!=encoded:
    raise SystemExit("existing submission evidence conflicts with accepted queue")
if not evidence_path.exists():
    temporary=evidence_path.with_name("."+evidence_path.name+".tmp")
    temporary.write_bytes(encoded)
    os.chmod(temporary,0o444)
    os.replace(temporary,evidence_path)
print(json.dumps({"submission_id":sid,"bundle":str(bundle_path),"checkpoint":str(checkpoint),"runtime":str(runtime),"evidence":str(evidence_path)},sort_keys=True))
"""


def _run(argv: Iterable[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        list(argv), check=False, text=True, capture_output=True, **kwargs
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            f"archive command failed with exit {completed.returncode}: {detail}"
        )
    return completed


def archive_entry(row: Mapping[str, Any], *, host: str, root: Path) -> dict[str, Any]:
    submission_id = int(row["submission_id"])
    digests = dict(row["archive_digests"])
    bundle_hex = str(digests["bundle"])[7:]
    incoming_dir = root / ".incoming"
    incoming = incoming_dir / f"{submission_id}.{bundle_hex}.tar.gz"
    canonical_bundle = (
        root / "artifacts" / "bundles" / f"sha256-{bundle_hex}" / "submission.tar.gz"
    )
    remote_existing = (
        subprocess.run(
            ["ssh", "-o", "BatchMode=yes", host, "test", "-f", str(canonical_bundle)],
            check=False,
            text=True,
            capture_output=True,
        ).returncode
        == 0
    )
    if remote_existing:
        incoming = canonical_bundle
    else:
        _run(["ssh", "-o", "BatchMode=yes", host, "mkdir", "-p", str(incoming_dir)])
        incoming_exists = (
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", host, "test", "-f", str(incoming)],
                check=False,
                text=True,
                capture_output=True,
            ).returncode
            == 0
        )
        if not incoming_exists:
            _run(
                [
                    "scp",
                    "-q",
                    "-o",
                    "BatchMode=yes",
                    str(row["file"]),
                    f"{host}:{incoming}",
                ]
            )
    payload = {
        "root": str(root),
        "incoming": str(incoming),
        "submission_id": submission_id,
        "label": str(row.get("label") or f"Submission {submission_id}"),
        "specialist_id": str(row.get("specialist_id") or "unknown"),
        "bundle_sha256": digests["bundle"],
        "checkpoint_sha256": digests["checkpoint"],
        "matchup_tree_sha256": digests["matchup_tree"],
    }
    encoded = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    completed = _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            host,
            "python3",
            "-c",
            shlex.quote(REMOTE_INSTALL),
            shlex.quote(encoded),
        ]
    )
    return json.loads(completed.stdout)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--elmo-host", default="elmo")
    parser.add_argument("--remote-root", type=Path, default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--min-submission-id", type=int, required=True)
    parser.add_argument("--max-submission-id", type=int)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    payload = json.loads(args.queue.expanduser().read_text(encoding="utf-8"))
    rows = accepted_entries(
        payload,
        minimum_id=args.min_submission_id,
        maximum_id=args.max_submission_id,
    )
    if args.check:
        print(
            json.dumps(
                {
                    "status": "check",
                    "submission_ids": [row["submission_id"] for row in rows],
                }
            )
        )
        return 0
    results = []
    for row in rows:
        result = archive_entry(row, host=args.elmo_host, root=args.remote_root)
        results.append(result)
        print(json.dumps({"status": "archived_submission", **result}), flush=True)
    print(json.dumps({"status": "archived", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
