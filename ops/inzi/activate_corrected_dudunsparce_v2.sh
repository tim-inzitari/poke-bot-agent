#!/usr/bin/env bash
set -euo pipefail

unit="pokebot-pure-rl-trevenant-staged.service"
staged_env="/home/inzi/.config/pokebot/specialist_runtime.corrected-v2.env"
active_env="/home/inzi/.config/pokebot/specialist_runtime.env"
runtime="/home/inzi/poke-bot-agent-deployments/dudunsparce-corrected-runtime-v2"
python="/home/inzi/miniconda3/envs/poke-bot-agent/bin/python"
ready="/home/inzi/poke-bot-agent/outputs/state/dudunsparce-expert-bootstrap-corrected-v2-ready.json"
registry="${runtime}/ops/specialist_runtime_registry_v1.json"
gate="${runtime}/ops/alakazam_gate_program_v1.json"
frozen="${runtime}/ops/frozen_specialist_registry_v1.json"
tree="/home/inzi/poke-bot-agent/outputs/state/dudunsparce-public-matchup-tree-v33.json"
receipt="/home/inzi/poke-bot-agent/outputs/state/dudunsparce-corrected-runtime-v2-activation.json"

test -s "${staged_env}"
test -s "${ready}"
test -s "${registry}"
test -s "${gate}"
test -s "${frozen}"
test -s "${tree}"
test -x "${python}"

old_selector_sha256="$(sha256sum "${active_env}" | awk '{print $1}')"
staged_selector_sha256="$(sha256sum "${staged_env}" | awk '{print $1}')"
ready_sha256="$(sha256sum "${ready}" | awk '{print $1}')"
registry_sha256="$(sha256sum "${registry}" | awk '{print $1}')"
gate_sha256="$(sha256sum "${gate}" | awk '{print $1}')"
frozen_sha256="$(sha256sum "${frozen}" | awk '{print $1}')"
tree_sha256="$(sha256sum "${tree}" | awk '{print $1}')"

# Validate the complete immutable selector package before waiting.  The current
# service is never stopped or signalled: its fail-closed controller owns its
# terminal lifecycle.
set -a
# shellcheck disable=SC1090
. "${staged_env}"
set +a
cd "${runtime}"
"${python}" -u scripts/launch_active_specialist.py --check

while systemctl --user --quiet is-active "${unit}"; do
  sleep 5
done

# Revalidate immediately before the atomic selector receipt boundary.
"${python}" -u scripts/launch_active_specialist.py --check
temporary="${active_env}.corrected-v2.$$"
install -m 0664 "${staged_env}" "${temporary}"
mv "${temporary}" "${active_env}"

systemctl --user reset-failed "${unit}"
systemctl --user start "${unit}"
systemctl --user --quiet is-active "${unit}"

"${python}" - "${receipt}" "${active_env}" "${registry}" "${ready}" \
  "${gate}" "${frozen}" "${tree}" \
  "${old_selector_sha256}" "${staged_selector_sha256}" \
  "${registry_sha256}" "${ready_sha256}" "${gate_sha256}" \
  "${frozen_sha256}" "${tree_sha256}" "${unit}" <<'PY'
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

(
    receipt_raw,
    selector_raw,
    registry_raw,
    ready_raw,
    gate_raw,
    frozen_raw,
    tree_raw,
    old_selector_sha256,
    staged_selector_sha256,
    registry_sha256,
    ready_sha256,
    gate_sha256,
    frozen_sha256,
    tree_sha256,
    unit,
) = sys.argv[1:]
receipt = Path(receipt_raw)
selector = Path(selector_raw)
registry = Path(registry_raw)
ready = Path(ready_raw)
gate = Path(gate_raw)
frozen = Path(frozen_raw)
tree = Path(tree_raw)

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

active_selector_sha256 = sha256(selector)
if (
    active_selector_sha256 != staged_selector_sha256
    or sha256(registry) != registry_sha256
    or sha256(ready) != ready_sha256
    or sha256(gate) != gate_sha256
    or sha256(frozen) != frozen_sha256
    or sha256(tree) != tree_sha256
):
    raise SystemExit("activation input changed across receipt boundary")

properties = subprocess.check_output(
    [
        "systemctl",
        "--user",
        "show",
        unit,
        "-p",
        "ActiveState",
        "-p",
        "SubState",
        "-p",
        "ExecMainPID",
    ],
    text=True,
)
service = dict(
    line.split("=", 1)
    for line in properties.splitlines()
    if "=" in line
)
if service.get("ActiveState") != "active":
    raise SystemExit("corrected trainer is not active")

payload = {
    "schema": "poke_bot.specialist_runtime_activation/v1",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "specialist_id": "dudunsparce",
    "run_name": "pure_rl_dudunsparce_temporal1_8k_corrected_v2_20260725",
    "old_selector_sha256": f"sha256:{old_selector_sha256}",
    "staged_selector_sha256": f"sha256:{staged_selector_sha256}",
    "active_selector_sha256": f"sha256:{active_selector_sha256}",
    "runtime_registry": str(registry),
    "runtime_registry_sha256": f"sha256:{registry_sha256}",
    "bootstrap_ready": str(ready),
    "bootstrap_ready_sha256": f"sha256:{ready_sha256}",
    "gate_contract": str(gate),
    "gate_contract_sha256": f"sha256:{gate_sha256}",
    "frozen_specialist_registry": str(frozen),
    "frozen_specialist_registry_sha256": f"sha256:{frozen_sha256}",
    "matchup_runtime_tree": str(tree),
    "matchup_runtime_tree_sha256": f"sha256:{tree_sha256}",
    "managed_service": unit,
    "service": service,
    "selector_committed": True,
    "corrected_checkpoint_only": True,
}
receipt.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_raw = tempfile.mkstemp(
    prefix=f".{receipt.name}.", dir=receipt.parent
)
temporary = Path(temporary_raw)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, receipt)
finally:
    temporary.unlink(missing_ok=True)
PY

echo "CORRECTED_DUDUNSPARCE_V2_ACTIVE"
