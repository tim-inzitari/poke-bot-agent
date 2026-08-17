#!/bin/bash
set -euo pipefail

repo="/Users/example/workspace/poke-bot-agent-h10-r79-stage"
stage="$repo"
label="com.pokebot.remote-worker-8766-h10-r80"
plist="/Users/example/Library/LaunchAgents/${label}.plist"
marker="$repo/outputs/state/bert-post-alakazam-iter0-rejoin-r89.json"
commit="/home/pokebot/poke-bot-agent-deployments/final-format-alakazam-h10-r79/outputs/pure_rl/final_format_alakazam_r79_h10_i_v6_8k/commits/iter_00000.json"
registry_local="$repo/outputs/runtime_recovery/specialist_runtime_registry_h10_r90_post_iter0_all_remotes.json"
registry_remote="/home/pokebot/poke-bot-agent/outputs/final_format_alakazam_r79/runtime/specialist_runtime_registry_h10_r90_post_iter0_all_remotes.json"
unit_local="$repo/outputs/runtime_recovery/pokebot-final-format-alakazam-r79-h10.post-iter0.service"
unit_remote="/home/pokebot/.config/systemd/user/pokebot-final-format-alakazam-r79-h10.service"
python="$stage/.venv/bin/python"
baseline_source="/home/pokebot/poke-bot-agent-deployments/pure-rl-resident-v9/baselines"
baseline_target="$repo/baselines"
baseline_manifest_sha256="94f3ae9b95b867780aebe3cc69cb372a218481ac5443c8e50c2781125801ec91"

ensure_baseline_library() {
  if [[ ! -f "$baseline_target/manifest.json" ]]; then
    partial="$repo/.baselines.partial.$$"
    rm -rf "$partial"
    mkdir -p "$partial"
    rsync -a --delete "train:$baseline_source/" "$partial/"
    actual="$(shasum -a 256 "$partial/manifest.json" | awk '{print $1}')"
    if [[ "$actual" != "$baseline_manifest_sha256" ]]; then
      rm -rf "$partial"
      echo "Bert baseline manifest mismatch: expected=$baseline_manifest_sha256 actual=$actual" >&2
      exit 1
    fi
    mv "$partial" "$baseline_target"
  fi

  actual="$(shasum -a 256 "$baseline_target/manifest.json" | awk '{print $1}')"
  [[ "$actual" == "$baseline_manifest_sha256" ]]
  (cd "$stage" && PYTHONPATH="$stage" "$python" - <<'PY'
from poke_bot.baselines_runtime import ensure_baselines_installed, load_manifest

specs = ensure_baselines_installed(load_manifest())
if len(specs) != 40:
    raise SystemExit(f"expected 40 portable baseline packages, found {len(specs)}")
PY
  )
}

# Portable public-opponent jobs resolve their package against Bert's own
# repository root.  Stage and validate that complete library before honoring
# an old rejoin marker or admitting Bert to production.
ensure_baseline_library

[[ -f "$marker" ]] && exit 0
ssh train "test -s '$commit'" || exit 0

if ! launchctl print "gui/$(id -u)/$label" >/dev/null 2>&1; then
  launchctl bootstrap "gui/$(id -u)" "$plist"
fi

check_health() {
  PYTHONPATH="$stage" "$python" - <<'PY'
from poke_bot.remote_jobs import RemoteJobClient
with RemoteJobClient("127.0.0.1", 8766, connect_timeout_s=15.0) as client:
    health = client.health()
required = (
    health.get("ok") is True,
    health.get("device") == "cpu",
    int(health.get("workers") or 0) == 16,
    int(health.get("leaf_servers") or 0) == 4,
    health.get("leaf_alive") is True,
    health.get("controller_healthy") is True,
    int(health.get("jobs_failed") or 0) == 0,
)
if not all(required):
    raise SystemExit(f"Bert optimized MPS worker is not stable: {health}")
PY
}

# Two separated successful probes keep a one-shot healthy response from
# triggering a production migration.
check_health
sleep 15
check_health

scp "$registry_local" "train:$registry_remote"
scp "$unit_local" "train:$unit_remote"
ssh train "systemctl --user stop pokebot-final-format-alakazam-r79-h10.service; systemctl --user daemon-reload; systemctl --user reset-failed pokebot-final-format-alakazam-r79-h10.service; systemctl --user start pokebot-final-format-alakazam-r79-h10.service; sleep 5; test \"\$(systemctl --user show pokebot-final-format-alakazam-r79-h10.service -p ActiveState --value)\" = active"

mkdir -p "$(dirname "$marker")"
printf '{"schema":"poke_bot.bert_post_iteration_rejoin/v1","status":"active","iteration":0,"backend":"cpu-4t","workers":16,"leaf_servers":4,"threads_per_leaf":4,"runtime_registry_sha256":"sha256:741bd8d1f90768e2fb0a42ca15527d6005beb6fa0b457777d59ba3c77de87f4a"}\n' >"$marker"
