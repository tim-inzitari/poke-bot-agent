# Replay Model Inspector

Replay Model Inspector is a **standalone, read-only localhost website** for
examining the checksum-bound model associated with a cached Kaggle submission. It is
not part of the training dashboard and it has no selector, trainer,
checkpoint-publication, submission, or service-control authority.

The product contract is staged in
[`state/replay-model-inspector-owner-design-r176.json`](../state/replay-model-inspector-owner-design-r176.json).
It is intended for post-training analysis and fine-tuning research without
changing the active training workload.

## What it inspects

The selector is deliberately precise:

1. Kaggle submission ID
2. Game / episode ID
3. Acting seat
4. Environment decision step and, when applicable, factorized action stage

For a verified submission, the inspector shows the causal replay observation,
legal options, recorded action, checkpoint and bundle identity, architecture
heads, raw and normalized head values where defined, fusion route reliabilities
and per-option contributions, base and final policy logits, probabilities, and
the recorded-versus-model action comparison.  It also exposes bounded parameter
summaries and bounded tensor slices; it never returns an entire tensor.

The raw replay cache does **not** contain model activations. The service
therefore reconstructs the selected decision in `eval()` /
`torch.inference_mode()` using the checksum-bound checkpoint and matchup tree.
It never substitutes the active training model or an unverified checkpoint.
Dynamic values are available only when an independently verified,
checksum-bound archived-runtime parity receipt is supplied and the service is
actually importing that exact extracted package.  Those values remain labelled
`recomputed_not_historical`: they are causal re-evaluation evidence, not
logits that were stored by Kaggle. A verified checkpoint/bundle identity alone
permits parameter inspection but never dynamic heads/logits or
`exact_reproduced`.

If exact provenance is unavailable or inconsistent, replay browsing may remain
available but model inspection must return an explicit unavailable reason.  It
must never fabricate a model result.

## Data roots and isolation

Elmo's existing cache convention is:

```text
/mnt/Main/main/poke-bot-agent/archive/submission-replays/
  <submission_id>/episodes.json
  <submission_id>/episode-<episode_id>-replay.json

/mnt/Main/main/poke-bot-agent/archive/submission-replay-rollouts/
  <submission_id>/episode-<episode_id>.jsonl
  <submission_id>/ROLLOUT_RECEIPT.json
```

The raw cache is downloaded by
[`scripts/sync_kaggle_submission_replays_elmo.py`](../scripts/sync_kaggle_submission_replays_elmo.py).
The rollout JSONL is produced by
[`scripts/rollout_kaggle_submission_replays.py`](../scripts/rollout_kaggle_submission_replays.py)
and contains causal, masked acting-seat records.  Prefer that masked view for
normal policy inspection.  Any omniscient/private replay data is diagnostic or
target-only and must be labelled as such; it must not be presented as a
deployment-visible model input.

Configure all source roots as absolute paths.  The server accepts only IDs and
bounded tensor selectors from clients, never a filesystem path.  For every
configured root it must resolve the candidate and prove it stays below that
root; symlinks that escape a root are rejected.  Source roots are read-only.
The service has a private state directory, but it has no reason to write below
any source root.

Evaluation/Kaggle replays remain evaluation-only.  The inspector must not
export them into training automatically, mark them training-eligible, tune
weights from them, or invoke any training service.

## Configuration

For a normal workstation launch, copy
[`config.example.json`](config.example.json) to `config.json`, then set
the absolute `replay_root`, `rollout_root`, `artifact_roots`,
`provenance_manifest`, and `runtime_source_root` paths for the host.
`runtime_source_root` is the immutable extracted root of the submitted package
and must be placed first on `PYTHONPATH` when launching the service. The flat JSON object contains only
the fields accepted by `replay_inspector.config.InspectorConfig`; keep
`config.json` host-local if those paths differ between hosts.

The service defaults are intentionally conservative:

- bind only to `127.0.0.1:8791`;
- CPU inference only, with GPU devices hidden from the process;
- lazy checkpoint loading with at most one resident model;
- no remote checkpoint reload or training invocation;
- no writes below replay, rollout, bundle, checkpoint, or provenance roots.

Use a Python environment containing the project, the competition `cg` runtime,
and CPU-capable PyTorch. The Elmo deployment instead uses the independent
container configuration in
[`ops/elmo/replay-model-inspector-config-r176.json`](../ops/elmo/replay-model-inspector-config-r176.json).
It clears GPU visibility and caps common CPU thread pools so it cannot contend
with the live GPU trainer.

## Exact provenance manifest

`provenance_manifest` must point to a checksum-bound manifest generated by the
artifact/receipt workflow.  This is the required version-1 shape:

```json
{
  "schema": "poke_bot.replay_model_inspector_provenance/v1",
  "version": 1,
  "generated_at_utc": "2026-08-07T00:00:00Z",
  "records": [
    {
      "submission_id": 55300000,
      "status": "verified",
      "checkpoint": {
        "path": "/mnt/Main/main/poke-bot-agent/archive/submission-bundles/55300000/extracted/model.pt",
        "sha256": "sha256:<64-lowercase-hex>"
      },
      "bundle": {
        "path": "/mnt/Main/main/poke-bot-agent/archive/submission-bundles/55300000/submission.tar.gz",
        "sha256": "sha256:<64-lowercase-hex>"
      },
      "matchup_tree": {
        "path": "/mnt/Main/main/poke-bot-agent/archive/submission-bundles/55300000/extracted/matchup_tree.json",
        "sha256": "sha256:<64-lowercase-hex>"
      },
      "runtime_package": {
        "path": "/mnt/Main/main/poke-bot-agent/archive/submission-bundles/55300000/submission.tar.gz",
        "sha256": "sha256:<64-lowercase-hex>"
      },
      "runtime_parity_receipt": {
        "path": "/mnt/Main/main/poke-bot-agent/archive/submission-bundles/55300000/runtime-parity.json",
        "sha256": "sha256:<64-lowercase-hex>"
      },
      "replay": {
        "games": [
          {
            "episode_id": 12345678,
            "replay_path": "/read-only/replays/55300000/episode-12345678-replay.json",
            "replay_sha256": "sha256:<64-lowercase-hex>"
          }
        ]
      }
    }
  ]
}
```

All `sha256:` values are full, lowercase SHA-256 digests.  The v1 loader
requires exactly one `verified` record for a submission, and verifies the
declared checkpoint, submission bundle, and matchup tree files before use.
Artifact paths may be relative to the manifest, but their resolved targets must
remain within `replay_root`, `rollout_root`, or an `artifact_roots` entry; an
escaping symlink is rejected.  A matching submission ID alone is not enough.

For a bundle that genuinely shipped no matchup tree, use the explicit null
form below instead of omitting the artifact:

```json
"runtime": {
  "matchup_tree_path": null,
  "matchup_tree_sha256": null
}
```

The checksum-bound bundle is the identity source for the submitted code, deck,
and packaged runtime assets. Extra descriptive fields are allowed, but the canonical v1
identity fields are `submission_id`, `status`, `checkpoint`, `bundle`, and
`matchup_tree` (or the explicit null runtime form).

For a dynamic trace, also declare `runtime_package` and
`runtime_parity_receipt`. The receipt is a separately checksum-bound JSON
artifact with this required shape:

```json
{
  "schema": "poke_bot.replay_model_inspector_runtime_parity_receipt/v1",
  "version": 1,
  "status": "verified",
  "submission_id": 55300000,
  "checkpoint_sha256": "sha256:<64-lowercase-hex>",
  "bundle_sha256": "sha256:<64-lowercase-hex>",
  "runtime_package_sha256": "sha256:<64-lowercase-hex>",
  "runtime_source_tree_sha256": "sha256:<64-lowercase-hex>",
  "verification": {
    "method": "independent_exact_runtime_parity",
    "verified_by": "named verifier",
    "verified_at_utc": "2026-08-07T00:00:00Z"
  }
}
```

Generate the source-tree digest and receipt after independently checking the
exact extracted package with
[`scripts/build_replay_inspector_runtime_parity_receipt.py`](../scripts/build_replay_inspector_runtime_parity_receipt.py).
At trace time the server checks the receipt bindings, rehashes
`runtime_source_root`, and rejects the request unless the imported
`poke_bot` module is inside that root. This prevents a current checkout from
being used merely because it can deserialize the checkpoint.

Every game eligible for a dynamic trace must also have exactly one
`replay.games[]` binding with its episode ID and replay SHA-256. The catalog
rehashes each file, and the request handler hashes the exact byte buffer it
parses again immediately before model loading. A new, missing, changed, or
ambiguous replay remains browseable but its model analysis fails closed. Run
[`scripts/build_replay_inspector_provenance.py`](../scripts/build_replay_inspector_provenance.py)
after a replay-cache sync to index newly downloaded games; pass the applicable
`--runtime-parity-receipt SUBMISSION_ID=PATH` argument for each attested
submission. The checked-in Elmo JSON is an identity bootstrap template with an
empty replay list, not a substitute for that generated live index.

The inspector verifies the declared artifacts before first use. A mismatch
invalidates the result rather than falling back to another model. The current
instrumentation runtime is always disclosed through the reproduction status;
bundle verification does not imply that historical execution timing, search
state, or exception/fallback behavior was reproduced.

## Local launch

Run the service only with a local config and a loopback host:

```bash
PY=/path/to/python-with-torch
PYTHONPATH=/read-only/extracted-submitted-package:$PYTHONPATH \
$PY scripts/start_replay_model_inspector.py \
  --config replay_inspector/config.json \
  --host 127.0.0.1 \
  --port 8791
```

Validate configuration without binding a port:

```bash
$PY scripts/start_replay_model_inspector.py \
  --config replay_inspector/config.json --check
```

Then open <http://127.0.0.1:8791>. The inspector remains its own service and
does not use the dashboard API or training process.

When the service runs on Elmo, retain loopback binding and tunnel from the
workstation instead of exposing it on the LAN:

```bash
ssh -N -o ExitOnForwardFailure=yes \
  -L 8791:127.0.0.1:8791 admin@elmo
```

Open <http://127.0.0.1:8791> on the workstation while that tunnel is active.

## Authenticated dashboard handoff

Revision 177 adds a presentation-only dashboard link to
<https://mc.tsinzitari.com/replay-inspector/>. The page is still served by the
separate Elmo inspector service. The existing dashboard Caddy edge handles
HTTPS and its normal external access policy, then sends only the fixed
`/replay-inspector/` path through a managed SSH local forward on Bert:

```text
browser -> Bert Caddy :443 -> Bert 127.0.0.1:8792
        -> encrypted SSH -> Elmo 127.0.0.1:8791
```

Neither Elmo port `8791` nor Bert port `8792` may bind a LAN/public address.
The gateway accepts GET only, strips the route prefix, replaces the upstream
Host with Elmo loopback, and removes browser authorization, cookies, origin,
referrer, and forwarding headers before the inspector sees the request. The
dashboard contains no inspector iframe, API polling, model state, or service
control.

The managed tunnel definition is
[`deploy/launchd/com.pokebot.replay-model-inspector-tunnel.plist`](../deploy/launchd/com.pokebot.replay-model-inspector-tunnel.plist).
The manual `ssh -L` flow above remains a local fallback if the external edge is
intentionally unavailable.

## Managed Elmo service

[`ops/elmo/pokebot-replay-model-inspector.service`](../ops/elmo/pokebot-replay-model-inspector.service)
is independent of all training and dashboard units. It does not start, stop,
reload, query, or alter another managed service. Elmo's host Python does not
contain Torch, so this unit launches a separate CPU-only, read-only container
from the already-installed worker image; it never execs into or modifies the
live worker container. The source deployment directory and every NAS mount are
read-only inside the inspector container.

The revision-176 Elmo files are:

- source: `/mnt/Main/main/poke-bot-agent/replay-model-inspector-r176`;
- replay cache: `/mnt/Main/main/poke-bot-agent/archive/submission-replays`;
- inspector artifact root:
  `/mnt/Main/main/poke-bot-agent/archive/replay-model-inspector`;
- exact extracted submitted runtime:
  `/mnt/Main/main/poke-bot-agent/archive/replay-model-inspector/artifacts/runtimes/sha256-335d91c0f3f239d885c153a154531c0a43f33a9199e63a285be15deed55c0c5b/package`;
- container config:
  [`ops/elmo/replay-model-inspector-config-r176.json`](../ops/elmo/replay-model-inspector-config-r176.json);
- checksum-bound manifest template:
  [`ops/elmo/replay-model-inspector-provenance-r176.json`](../ops/elmo/replay-model-inspector-provenance-r176.json).

An operator may manage this one service through the normal service manager:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now pokebot-replay-model-inspector.service
sudo systemctl status pokebot-replay-model-inspector.service
```

The service's hardening is intentionally scoped to this read-only tool:
read-only root and bind mounts, no Linux capabilities, no GPU visibility,
bounded CPU/memory/PIDs, `NoNewPrivileges`, and loopback-only application
configuration. It is not a mechanism for controlling interactive or training
sessions.

## Verification checklist

At minimum, validate:

```bash
python -m py_compile scripts/start_replay_model_inspector.py
python scripts/start_replay_model_inspector.py --config replay_inspector/config.json --check
pytest -q tests/test_replay_inspector_*.py
curl -fsS http://127.0.0.1:8791/healthz
```

Required test coverage includes cache indexing, provenance rejection, a
per-step/all-head/fusion fixture, parameter-summary and slice bounds,
configured-root and symlink-escape rejection, loopback binding, and a static
UI build or asset test.
