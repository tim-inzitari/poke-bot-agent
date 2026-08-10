#!/bin/bash
set -euo pipefail

SRC_DIR=/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v9/baselines/specialists/marnie-final-format-h10-f20efb20f5c3
DEST1=/Users/tsinzitari/workspace/poke-bot-agent-h10-r79-stage/baselines/specialists/marnie-final-format-h10-f20efb20f5c3
DEST2=/Users/tsinzitari/workspace/poke-bot-agent/baselines/specialists/marnie-final-format-h10-f20efb20f5c3

rm -rf "$DEST1" "$DEST2"
mkdir -p "$(dirname "$DEST1")" "$(dirname "$DEST2")"

ssh -o BatchMode=yes -o ConnectTimeout=30 train \
  "tar -C $(dirname "$SRC_DIR") -cf - $(basename "$SRC_DIR")" \
  | tar -C "$(dirname "$DEST1")" -xf -
echo "stage_extracted"
cp -a "$DEST1/." "$DEST2/"
echo "native_copied"
test -f "$DEST1/model.pt"
du -sh "$DEST1" "$DEST2"

python3 - <<'PY'
import json, shutil, datetime
from pathlib import Path
entry = {
    "dir": "marnie-final-format-h10-f20efb20f5c3",
    "group": "specialists",
    "id": "specialist-marnie-final-format-h10-f20efb20f5c3",
    "name": "Frozen final-format H10 Marnie's Grimmsnarl ex refresh",
    "source": "checksum-bound Marnie completion sha256:1c4f2f2ee464b32c960f65c0ff1fd469a732ab9e87fc031cd7f6fe1e3ac3989d",
}
for p in [
    Path("/Users/tsinzitari/workspace/poke-bot-agent-h10-r79-stage/baselines/manifest.json"),
    Path("/Users/tsinzitari/workspace/poke-bot-agent/baselines/manifest.json"),
]:
    bak = p.with_name(
        p.name
        + ".pre-h10-marnie-"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    shutil.copy2(p, bak)
    data = json.loads(p.read_text())
    agents = data.setdefault("agents", [])
    ids = {a.get("id") for a in agents if isinstance(a, dict)}
    if entry["id"] not in ids:
        agents.append(entry)
        p.write_text(json.dumps(data, indent=2) + "\n")
        print("added", p)
    else:
        print("already", p)
PY

# Push Bert-local copy to Elmo via train relay of the train-side package
ssh -o BatchMode=yes -o ConnectTimeout=30 train 'bash -s' <<'REMOTE'
set -euo pipefail
SRC=/home/inzi/poke-bot-agent-deployments/pure-rl-resident-v9/baselines/specialists/marnie-final-format-h10-f20efb20f5c3
ssh -o BatchMode=yes -o ConnectTimeout=20 elmo "mkdir -p /mnt/Main/main/poke-bot-agent/baselines/specialists"
tar -C "$(dirname "$SRC")" -cf - "$(basename "$SRC")" \
  | ssh -o BatchMode=yes -o ConnectTimeout=20 elmo \
      "tar -C /mnt/Main/main/poke-bot-agent/baselines/specialists -xf -"
ssh -o BatchMode=yes -o ConnectTimeout=20 elmo 'python3 - <<'"'"'PY'"'"'
import json, shutil, datetime
from pathlib import Path
p = Path("/mnt/Main/main/poke-bot-agent/baselines/manifest.json")
bak = p.with_name(
    p.name
    + ".pre-h10-marnie-"
    + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
)
shutil.copy2(p, bak)
data = json.loads(p.read_text())
agents = data.setdefault("agents", [])
entry = {
    "dir": "marnie-final-format-h10-f20efb20f5c3",
    "group": "specialists",
    "id": "specialist-marnie-final-format-h10-f20efb20f5c3",
    "name": "Frozen final-format H10 Marnie Grimmsnarl ex refresh",
    "source": "checksum-bound Marnie completion sha256:1c4f2f2ee464b32c960f65c0ff1fd469a732ab9e87fc031cd7f6fe1e3ac3989d",
}
ids = {a.get("id") for a in agents if isinstance(a, dict)}
if entry["id"] not in ids:
    agents.append(entry)
    p.write_text(json.dumps(data, indent=2) + "\n")
    print("added", p)
else:
    print("already", p)
print(
    "pkg",
    Path(
        "/mnt/Main/main/poke-bot-agent/baselines/specialists/"
        "marnie-final-format-h10-f20efb20f5c3/model.pt"
    ).exists(),
)
PY'
echo elmo_ok
REMOTE

python3 - <<'PY'
import json, datetime
from pathlib import Path
receipt = {
    "schema": "poke_bot.crustle_h10_marnie_baseline_fleet_sync/v1",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "baseline_id": "specialist-marnie-final-format-h10-f20efb20f5c3",
    "hosts": ["bert.stage", "bert.native", "elmo"],
    "reason": "Dual-Marnie r167 practice opponent missing on remotes",
}
out = Path(
    "/Users/tsinzitari/workspace/poke-bot-agent/outputs/state/"
    "crustle-h10-marnie-baseline-fleet-sync-r167.json"
)
out.write_text(json.dumps(receipt, indent=2) + "\n")
print(receipt)
PY
