# Elastic remote simulation fleet (staged, not active)

The staged gateway gives the trainer one stable endpoint while one or many
remote simulator/GPU-leaf workers can be admitted behind it. Future membership
changes update one atomic JSON registry; they do not change trainer arguments,
training code, optimizer state, or the active checkpoint.

Nothing in this staging revision installs or starts a service, opens a port,
creates an AWS resource, changes the active trainer, or admits a new host.

## Why CPU and GPU are both required

The current workload is consistently simulator/CPU bound on Inzi, but each
remote still needs a co-located CUDA policy leaf. A CPU-only host adds simulator
capacity only if it can reach a leaf without turning per-decision WAN latency
into the new bottleneck. The preferred unit is therefore many fast CPU cores,
adequate RAM, and one modest inference GPU. More GPU than an L4/A10-class card
is unlikely to help until a host-level probe shows its policy leaf is saturated.

## Staged architecture

```text
trainer (future one-time adoption)
        |
        | existing length-prefixed JSON + negotiated zlib frames
        v
127.0.0.1:8770 stable fleet gateway
        |-- authenticated tunnel/private route --> Elmo worker
        |-- authenticated tunnel/private route --> Bert worker
        |-- authenticated tunnel/private route --> AWS worker 1
        `-- authenticated tunnel/private route --> AWS worker N
```

The gateway rejects unknown manifest fields, public bind addresses, duplicate
IDs/endpoints, mixed active checkpoint digests, capacity above the declared
fleet ceiling, missing worker capabilities, hostname mismatches, and
storage-local checkpoint digest mismatches. A candidate registry is admitted
as a complete snapshot. If a new remote fails preflight, the last known-good
snapshot remains in service. Removed or changed backends wait for active jobs
to drain.

Checkpoint reload and pin operations are fanned out. Every target checkpoint
is verified on every backend before the first mutating command is sent. A
partial backend failure makes the control call fail closed; the trainer's
existing between-iteration digest hard gate remains authoritative.

## Operator flow after an owner activation decision

For an updated standard worker, adding capacity is intentionally just:

```bash
sudo scripts/remote_fleet.py add 10.0.0.42
```

One command can add several workers in one all-or-nothing registry update:

```bash
sudo scripts/remote_fleet.py add 10.0.0.42 10.0.0.43 10.0.0.44
```

The command discovers each hostname, active checkpoint path and SHA-256,
advertised capacity, job kinds, and capabilities; checks worker/leaf health;
verifies the checkpoint from the worker's own storage; then validates, fsyncs,
and atomically replaces the registry. If any member of a batch fails, none are
written. The gateway will notice an accepted snapshot within its normal poll
interval. Use `--dry-run` to perform every probe and validation without
replacing the registry.

The default registry is `/etc/pokebot/remote-fleet-gateway.active.json`. For a
different path, either pass `--manifest PATH` before the command or set
`POKEBOT_REMOTE_FLEET_MANIFEST`. Useful commands are:

```bash
scripts/remote_fleet.py list
scripts/remote_fleet.py check
sudo scripts/remote_fleet.py remove aws-node-1
```

An older worker that predates checkpoint-path advertisement needs one extra
argument: `--checkpoint-path /absolute/remote/checkpoint.pt`. A nonstandard
worker checkout can use `--worker-root`; `--no-path-rewrite` is available when
trainer-visible spec paths are already identical.

1. Copy `config/remote_fleet_gateway.staged.json` outside the repository as
   `/etc/pokebot/remote-fleet-gateway.active.json`.
2. Replace placeholder identities with the exact checkpoint digest, backend
   hostname, private/tunnel endpoint, capacity probe result, and remote path.
3. Keep `enabled=false` while staging the worker image and checkpoint.
4. Validate syntax without network activity:

   ```bash
   python3 -m poke_bot.remote_fleet_gateway \
     --manifest /etc/pokebot/remote-fleet-gateway.active.json --check
   ```

5. Run a receipt-backed backend preflight, set `enabled=true`, and atomically
   replace the registry. Set `activation_allowed=true` only under the separate
   activation revision.
6. Install/start the staged gateway unit only at a clean collection boundary.
   At that same one-time boundary, change the trainer's remote endpoint to
   `127.0.0.1:8770` and set its gateway endpoint demand ceiling to 512. No later
   trainer restart is needed for ordinary fleet additions/removals.

The example AWS backend uses a local forwarded port (`127.0.0.1:18765`). It is
safe to use a VPC-private address instead if Inzi has an authenticated private
route. Raw worker port 8765 must never be internet-exposed.

## Selected AWS burst: two `g6.16xlarge`, optionally four

The staged operator deliberately supports only the selected shape. Each node
has 64 vCPU, 256 GiB RAM, and one 24 GiB L4. The simulator pool is 48 per node,
leaving CPU/RAM headroom while the co-located L4 services batched inference.

This table uses the us-east-1 Linux On-Demand planning rate of $3.3968 per
`g6.16xlarge` hour. Verify the exact account/region price before launch. GPS is
an engineering range, not a measured AWS result. The full-local reference is
the prior receipt-backed 10.31 GPS Blackwell-plus-Elmo exact collection, not a
promise that the current derivative has identical per-game cost.

| Fleet | Incremental cost/hour | 48-hour compute | Network-adjusted added GPS | Estimated total GPS vs 10.31 local | Approx. wall time for 4,000 games |
|---|---:|---:|---:|---:|---:|
| Full local fleet only | $0 | $0 | — | 10.31 measured reference | 6m 28s |
| 1× `g6.16xlarge` | $3.3968 | $163.05 | +5–7 | 15.3–17.3 | 3m 51s–4m 21s |
| 2× `g6.16xlarge` | $6.7936 | $326.09 | +9–13 | 19.3–23.3 | 2m 52s–3m 27s |
| 4× `g6.16xlarge` | $13.5872 | $652.19 | +17–24 | 27.3–34.3 | 1m 57s–2m 26s |
| 16× `g6.16xlarge` | $54.3488 | $2,608.74 | +45–70 | 55.3–80.3 | 50s–1m 13s |

The ranges already reduce scaling efficiency for the network. Policy leaves
are co-located on each AWS host, so per-decision inference does not cross the
WAN. The tunnel carries compressed whole-game requests and returned
trajectories. One or two nodes should usually remain compute-bound; at four,
Inzi's Internet path and the gateway begin to matter; 16 is explicitly
bandwidth/gateway-limited and is not expected to scale linearly. Launch
admission proves identity and health, but only a real 20–30 minute throughput
receipt can replace these estimates.

The 48-hour compute column excludes small gp3, PrivateLink endpoint, data
processing/egress, and tax charges. The launcher's `$500` default therefore
does not use the nominal $326.09 alone: it uses a conservative `$4/node-hour`
plus `$20` overhead ceiling, or `$404` for two nodes. Four nodes for all 48
hours are rejected (`$788` conservative; `$652.19` nominal compute). A later
`add-two` inherits the original absolute shutdown deadline and counts both the
original two-node horizon and the new nodes' remaining hours; under a `$500`
ceiling it becomes eligible only when no more than 12 whole hours remain.

AWS Budgets is useful as an alert or secondary action, but it is not a
real-time prepaid cap: AWS billing data can lag by many hours. The hard local
protections for this burst are pre-launch refusal, an exact CloudFormation
stack, and an instance-side absolute auto-termination deadline.

## Simple operator flow

Run the operator on Inzi, where the trainer-facing gateway lives. AWS
credentials are never placed in the JSON or copied to a worker. First create a
normal AWS CLI profile (access key/secret, or SSO) and install AWS CLI v2 plus
the Session Manager plugin. Then:

```bash
cd /home/inzi/poke-bot-agent
python3 scripts/aws_remote_fleet.py init ~/pokebot-aws.json \
  --profile pokebot-burst --region us-east-1
```

The generated file already selects two nodes, 48 hours, and `$500`. With the
default managed network, the only deliberate edit is:

```json
"activation_allowed": true
```

`vpc_id` and `subnet_id` remain `"auto"`. CloudFormation creates an isolated
VPC, a private subnet, and SSM/SSM Messages interface endpoints. Instances
receive no public IP and the security group has no inbound rule. If an
existing VPC is preferred, put both real IDs in the file; that subnet must
already reach SSM and SSM Messages over HTTPS.

Preflight and launch are two explicit commands:

```bash
python3 scripts/aws_remote_fleet.py check ~/pokebot-aws.json
python3 scripts/aws_remote_fleet.py launch ~/pokebot-aws.json \
  --confirm-cost-limit 500
python3 scripts/aws_remote_fleet.py status ~/pokebot-aws.json
```

The repository's gateway is currently staged and inactive. Before a billable
launch, the separately authorized one-time clean-boundary adoption must have
installed the active manifest/service, registered the existing local/LAN
backends, and changed the trainer's one remote endpoint to the loopback
gateway. `check` deliberately refuses while that receipt-backed adoption is
missing; it does not treat a filled AWS config as authority to touch training.

`check` verifies AWS identity, G-instance vCPU quota, regional/AZ availability,
AMI lookup, source worker-image identity, current learner-checkpoint identity,
required local tools, and the cost ceiling. `launch` creates the stack, opens
managed SSM tunnels, streams the Docker image directly from Elmo and the
checkpoint directly from Inzi, verifies both immutable identities on every
node, starts the workers, and atomically registers them. It uses no S3 object,
ECR repository, Tailscale network, public SSH, or public worker port.

Later expansion and cost stop are:

```bash
python3 scripts/aws_remote_fleet.py add-two ~/pokebot-aws.json \
  --confirm-cost-limit 500
python3 scripts/aws_remote_fleet.py stop ~/pokebot-aws.json
```

`add-two` updates the same stack from two to four and keeps the original fleet
expiry. It refuses if the cumulative conservative ceiling is above `$500` or
the account quota is below 256 G/VT vCPUs. `stop` requests deletion of only the
named CloudFormation stack and stops only its managed user tunnel units.

AWS commonly defaults the regional On-Demand G/VT quota to zero, so a new
account may need a quota increase to 128 vCPUs for two nodes (256 for four)
before any launch can succeed.

## Files

- Gateway: `poke_bot/remote_fleet_gateway.py`
- Simple add/list/remove command: `scripts/remote_fleet.py`
- Atomic registry controller: `poke_bot/remote_fleet_registry.py`
- Inert registry: `config/remote_fleet_gateway.staged.json`
- Service template: `deploy/systemd/pokebot-remote-fleet-gateway.service.template`
- AWS backend entry: `deploy/aws/remote-fleet/backend.example.json`
- AWS one-file operator: `scripts/aws_remote_fleet.py`
- AWS user config template: `deploy/aws/remote-fleet/aws-fleet.user.example.json`
- AWS CloudFormation stack: `deploy/aws/remote-fleet/two-g6-16xlarge.cloudformation.yaml`
- AWS worker environment: `deploy/aws/remote-fleet/remote-worker.env.example`
