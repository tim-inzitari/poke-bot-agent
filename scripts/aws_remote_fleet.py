#!/usr/bin/env python3
"""One-file operator for two (optionally four) private AWS GPU workers.

Normal use, on Inzi:

    scripts/aws_remote_fleet.py check aws.json
    scripts/aws_remote_fleet.py launch aws.json --confirm-cost-limit 500
    scripts/aws_remote_fleet.py status aws.json
    scripts/aws_remote_fleet.py add-two aws.json --confirm-cost-limit 500
    scripts/aws_remote_fleet.py stop aws.json

The launcher uses CloudFormation only for private EC2/SSM infrastructure.
Container and checkpoint bytes are streamed directly through authenticated SSM
SSH forwards.  S3, ECR, public SSH, and Tailscale are not used.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.aws_remote_fleet import (  # noqa: E402
    AwsCli,
    AwsFleetConfig,
    AwsFleetError,
    prerequisite_report,
    write_json_atomic,
)
from poke_bot.remote_fleet_gateway import FleetManifest  # noqa: E402
from poke_bot.remote_fleet_registry import (  # noqa: E402
    DiscoveredBackend,
    add_backends,
    discover_backend,
)


TEMPLATE = ROOT / "deploy/aws/remote-fleet/two-g6-16xlarge.cloudformation.yaml"
EXAMPLE = ROOT / "deploy/aws/remote-fleet/aws-fleet.user.example.json"
TRAINER_SERVICE = "pokebot-alakazam-rule-derivative-g5-rl.service"
PLANNING_PRICE_CEILING_PER_INSTANCE_HOUR = 4.00
FIXED_OVERHEAD_CEILING_USD = 20.00


def _run(
    command: Sequence[str],
    *,
    input_text: Optional[str] = None,
    timeout: Optional[float] = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise AwsFleetError(
            f"command failed ({' '.join(command[:8])} ...): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def _ssh(host: str, *command: str, timeout: float = 30.0) -> str:
    return _run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, *command],
        timeout=timeout,
    ).stdout


def _state_root(config: AwsFleetConfig) -> Path:
    return Path.home() / ".config/pokebot/aws-remote-fleet" / config.stack_name


def _cost_ceiling(config: AwsFleetConfig, count: Optional[int] = None) -> float:
    selected = int(count or config.instance_count)
    return (
        selected
        * config.max_runtime_hours
        * PLANNING_PRICE_CEILING_PER_INSTANCE_HOUR
        + FIXED_OVERHEAD_CEILING_USD
    )


def _expansion_cost_ceiling(
    config: AwsFleetConfig, state: Path, selected: int
) -> float:
    receipt_path = state / "launch-receipt.json"
    if not receipt_path.is_file():
        return _cost_ceiling(config, selected)
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        current = len(receipt["instance_ids"])
        expires = int(receipt["expire_at_epoch"])
        original_hours = int(receipt["planned_runtime_hours"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return _cost_ceiling(config, selected)
    if not 0 < current <= selected:
        return _cost_ceiling(config, selected)
    remaining_hours = max(0, math.ceil((expires - time.time()) / 3600.0))
    return (
        current * original_hours * PLANNING_PRICE_CEILING_PER_INSTANCE_HOUR
        + (selected - current)
        * remaining_hours
        * PLANNING_PRICE_CEILING_PER_INSTANCE_HOUR
        + FIXED_OVERHEAD_CEILING_USD
    )


def _assert_budget(
    config: AwsFleetConfig,
    confirmation: Optional[float],
    count: int,
    *,
    ceiling: Optional[float] = None,
) -> None:
    ceiling = float(ceiling if ceiling is not None else _cost_ceiling(config, count))
    if ceiling > config.spend_limit_usd:
        raise AwsFleetError(
            f"refusing: conservative {count}-instance ceiling ${ceiling:.2f} "
            f"exceeds configured spend_limit_usd=${config.spend_limit_usd:.2f}"
        )
    if confirmation is None or abs(confirmation - config.spend_limit_usd) > 0.005:
        raise AwsFleetError(
            "launch needs --confirm-cost-limit exactly matching configured "
            f"${config.spend_limit_usd:.2f}"
        )


def _ensure_key(state: Path) -> tuple[Path, str]:
    private = state / "id_ed25519"
    public = state / "id_ed25519.pub"
    state.mkdir(parents=True, exist_ok=True)
    if not private.exists():
        if public.exists():
            raise AwsFleetError(f"public key exists without private key: {public}")
        _run(
            [
                "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                f"pokebot-{state.name}",
                "-f",
                str(private),
            ]
        )
    os.chmod(private, 0o600)
    text = public.read_text(encoding="utf-8").strip()
    if not text.startswith("ssh-ed25519 "):
        raise AwsFleetError("deployment public key is not ed25519")
    return private, text


def _checkpoint_info(config: AwsFleetConfig) -> dict[str, Any]:
    host = config.source_checkpoint_host
    path = config.source_checkpoint_path
    if path == "auto":
        exec_start = _ssh(
            host,
            "systemctl",
            "--user",
            "show",
            TRAINER_SERVICE,
            "--property=ExecStart",
            "--value",
        )
        match = re.search(r"--run-name\s+([^\s;]+)", exec_start)
        if not match:
            raise AwsFleetError(f"cannot discover --run-name from {TRAINER_SERVICE}")
        run_name = match.group(1)
        loop_state = (
            f"/home/inzi/poke-bot-agent/outputs/pure_rl/{run_name}/loop_state.json"
        )
        raw = _ssh(host, "cat", "--", loop_state)
        try:
            state = json.loads(raw)
            learner = state["learner"]
            path = str(learner["path"])
            expected = str(learner["digest"])
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AwsFleetError(f"invalid learner identity in {loop_state}") from exc
    else:
        expected = ""
    proof = _ssh(host, "sha256sum", "--", path).strip().split()
    if len(proof) < 1 or not re.fullmatch(r"[0-9a-f]{64}", proof[0]):
        raise AwsFleetError("source checkpoint sha256sum returned invalid output")
    digest = "sha256:" + proof[0]
    if expected and digest != expected:
        raise AwsFleetError(f"loop-state checkpoint mismatch: {digest} != {expected}")
    size_text = _ssh(host, "stat", "-c", "%s", "--", path).strip()
    return {"host": host, "path": path, "digest": digest, "size": int(size_text)}


def _image_info(config: AwsFleetConfig) -> dict[str, str]:
    value = _ssh(
        config.source_image_host,
        "sudo",
        "-n",
        "docker",
        "image",
        "inspect",
        "--format",
        "{{.Id}}",
        config.source_image_ref,
    ).strip()
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise AwsFleetError("source worker image returned an invalid immutable ID")
    return {"host": config.source_image_host, "ref": config.source_image_ref, "id": value}


def _cloud_preflight(config: AwsFleetConfig, count: int) -> dict[str, Any]:
    aws = AwsCli(config.aws_profile, config.region)
    identity = aws.run_json("sts", "get-caller-identity")
    network: dict[str, Any]
    if config.vpc_id == "auto":
        network = {
            "mode": "managed-private",
            "detail": "stack creates an isolated VPC plus SSM interface endpoints",
        }
    else:
        vpcs = aws.run_json("ec2", "describe-vpcs", "--vpc-ids", config.vpc_id)
        subnets = aws.run_json(
            "ec2", "describe-subnets", "--subnet-ids", config.subnet_id
        )
        subnet_rows = list(subnets.get("Subnets") or [])
        if len(vpcs.get("Vpcs") or []) != 1 or len(subnet_rows) != 1:
            raise AwsFleetError("VPC/subnet lookup did not return exactly one result")
        if subnet_rows[0].get("VpcId") != config.vpc_id:
            raise AwsFleetError("configured subnet does not belong to configured VPC")
        network = {
            "mode": "existing",
            "subnet_availability_zone": subnet_rows[0].get("AvailabilityZone"),
            "requirement": "subnet must reach SSM and ssmmessages over HTTPS",
        }
    offerings = aws.run_json(
        "ec2",
        "describe-instance-type-offerings",
        "--location-type",
        "availability-zone",
        "--filters",
        f"Name=instance-type,Values={config.instance_type}",
    )
    offered_zones = sorted(
        {
            str(row.get("Location"))
            for row in offerings.get("InstanceTypeOfferings") or []
            if row.get("Location")
        }
    )
    if not offered_zones:
        raise AwsFleetError(f"{config.instance_type} is unavailable in {config.region}")
    if config.vpc_id == "auto":
        selected_zone = offered_zones[0]
    else:
        selected_zone = str(network["subnet_availability_zone"])
        if selected_zone not in offered_zones:
            raise AwsFleetError(
                f"{config.instance_type} is unavailable in subnet zone {selected_zone}"
            )
    ami = aws.run_json("ssm", "get-parameter", "--name", config.ami_ssm_parameter)
    quota_name = (
        "Running On-Demand G and VT instances"
        if config.market == "on-demand"
        else "All G and VT Spot Instance Requests"
    )
    quotas = aws.run_json(
        "service-quotas", "list-service-quotas", "--service-code", "ec2"
    )
    quota_rows = [
        row
        for row in quotas.get("Quotas") or []
        if row.get("QuotaName") == quota_name
    ]
    quota: dict[str, Any]
    if quota_rows:
        quota_value = float(quota_rows[0].get("Value") or 0.0)
        required_vcpus = 64 * count
        quota = {
            "name": quota_name,
            "available_vcpus": quota_value,
            "requested_vcpus": required_vcpus,
            "sufficient": quota_value >= required_vcpus,
        }
        if quota_value < required_vcpus:
            raise AwsFleetError(
                f"{quota_name} quota is {quota_value:g} vCPUs; "
                f"{count} g6.16xlarge instances require {required_vcpus}"
            )
    else:
        quota = {
            "name": quota_name,
            "available_vcpus": None,
            "requested_vcpus": 64 * count,
            "sufficient": None,
            "warning": "quota was not present in the first Service Quotas response",
        }
    return {
        "account": identity.get("Account"),
        "arn": identity.get("Arn"),
        "network": network,
        "selected_availability_zone": selected_zone,
        "offered_availability_zones": offered_zones,
        "quota": quota,
        "ami_id": dict(ami.get("Parameter") or {}).get("Value"),
    }


def _stack_parameters(
    config: AwsFleetConfig,
    public_key: str,
    count: int,
    availability_zone: str,
    expire_at_epoch: int,
) -> list[str]:
    return [
        f"UseManagedNetwork={'true' if config.vpc_id == 'auto' else 'false'}",
        f"ManagedAvailabilityZone={availability_zone}",
        f"VpcId={config.vpc_id}",
        f"SubnetId={config.subnet_id}",
        f"InstanceCount={count}",
        f"MarketType={config.market}",
        f"RootVolumeGiB={config.root_volume_gib}",
        f"MaxRuntimeHours={config.max_runtime_hours}",
        f"ExpireAtEpoch={expire_at_epoch}",
        f"DeploymentPublicKey={public_key}",
    ]


def _deploy(
    config: AwsFleetConfig,
    public_key: str,
    count: int,
    availability_zone: str,
    expire_at_epoch: int,
) -> None:
    _run(
        [
            "aws",
            "--profile",
            config.aws_profile,
            "--region",
            config.region,
            "cloudformation",
            "deploy",
            "--template-file",
            str(TEMPLATE),
            "--stack-name",
            config.stack_name,
            "--capabilities",
            "CAPABILITY_IAM",
            "--no-fail-on-empty-changeset",
            "--parameter-overrides",
            *_stack_parameters(
                config,
                public_key,
                count,
                availability_zone,
                expire_at_epoch,
            ),
        ],
        timeout=1800,
    )


def _stack_outputs(config: AwsFleetConfig) -> list[str]:
    value = AwsCli(config.aws_profile, config.region).run_json(
        "cloudformation", "describe-stacks", "--stack-name", config.stack_name
    )
    stacks = list(value.get("Stacks") or [])
    if len(stacks) != 1:
        raise AwsFleetError("stack lookup did not return exactly one stack")
    outputs = {
        str(row.get("OutputKey")): str(row.get("OutputValue"))
        for row in stacks[0].get("Outputs") or []
    }
    ids = [outputs.get(f"Worker{i}InstanceId", "") for i in range(1, 5)]
    ids = [item for item in ids if re.fullmatch(r"i-[0-9a-f]{8,17}", item)]
    if len(ids) not in {2, 4}:
        raise AwsFleetError(f"stack returned unexpected worker IDs: {ids}")
    return ids


def _unit_name(kind: str, index: int) -> str:
    return f"pokebot-aws-{kind}-{index + 1:02d}.service"


def _install_tunnel_units(
    config: AwsFleetConfig,
    instance_ids: list[str],
    state: Path,
) -> None:
    unit_root = Path.home() / ".config/systemd/user"
    unit_root.mkdir(parents=True, exist_ok=True)
    profile = config.aws_profile
    region = config.region
    for index, instance_id in enumerate(instance_ids):
        for kind, local_port, remote_port in (
            ("worker-tunnel", config.local_tunnel_ports[index], 8765),
            ("ssh-tunnel", config.ssh_tunnel_ports[index], 22),
        ):
            params = json.dumps(
                {"portNumber": [str(remote_port)], "localPortNumber": [str(local_port)]},
                separators=(",", ":"),
            )
            unit = unit_root / _unit_name(kind, index)
            unit_text = "\n".join(
                [
                    "[Unit]",
                    f"Description=Pokebot AWS {kind} {instance_id}",
                    "After=network-online.target",
                    "Wants=network-online.target",
                    "",
                    "[Service]",
                    "Type=simple",
                    "ExecStart="
                    + " ".join(
                        [
                            shutil.which("aws") or "/usr/local/bin/aws",
                            "--profile",
                            profile,
                            "--region",
                            region,
                            "ssm",
                            "start-session",
                            "--target",
                            instance_id,
                            "--document-name",
                            "AWS-StartPortForwardingSession",
                            "--parameters",
                            shlex.quote(params),
                        ]
                    ),
                    "Restart=always",
                    "RestartSec=5",
                    "",
                    "[Install]",
                    "WantedBy=default.target",
                    "",
                ]
            )
            temporary = unit.with_name(f".{unit.name}.tmp")
            temporary.write_text(unit_text, encoding="utf-8")
            os.replace(temporary, unit)

    _run(["systemctl", "--user", "daemon-reload"])
    for index in range(len(instance_ids)):
        for kind in ("worker-tunnel", "ssh-tunnel"):
            _run(["systemctl", "--user", "enable", "--now", _unit_name(kind, index)])
    write_json_atomic(state / "instances.json", {"instance_ids": instance_ids}, mode=0o600)


def _wait_port(port: int, timeout: float = 300.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise AwsFleetError(f"local tunnel port {port} did not become ready")


def _ssh_target_args(state: Path, port: int) -> list[str]:
    return [
        "-i",
        str(state / "id_ed25519"),
        "-p",
        str(port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={state / 'known_hosts'}",
        "ec2-user@127.0.0.1",
    ]


def _capture_host_keys(config: AwsFleetConfig, state: Path, count: int) -> None:
    rows: list[str] = []
    for port in config.ssh_tunnel_ports[:count]:
        _wait_port(port)
        result = _run(["ssh-keyscan", "-T", "10", "-p", str(port), "127.0.0.1"])
        rows.extend(line for line in result.stdout.splitlines() if line and not line.startswith("#"))
    if len(rows) < count:
        raise AwsFleetError("SSH host-key scan returned too few keys")
    known = state / "known_hosts"
    temporary = known.with_name(f".{known.name}.tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8")
    os.replace(temporary, known)
    os.chmod(known, 0o600)


def _pipe_commands(source: Sequence[str], destination: Sequence[str]) -> None:
    producer = subprocess.Popen(list(source), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert producer.stdout is not None
    consumer = subprocess.Popen(list(destination), stdin=producer.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    producer.stdout.close()
    consumer_out, consumer_err = consumer.communicate()
    producer_err = producer.stderr.read() if producer.stderr else b""
    producer_rc = producer.wait()
    if producer_rc != 0 or consumer.returncode != 0:
        raise AwsFleetError(
            "direct transfer failed: "
            + (producer_err + consumer_err + consumer_out).decode("utf-8", errors="replace")[-4000:]
        )


def _send_image(config: AwsFleetConfig, state: Path, index: int, image: dict[str, str]) -> None:
    port = config.ssh_tunnel_ports[index]
    destination = [
        "ssh",
        *_ssh_target_args(state, port),
        "sudo -n docker load",
    ]
    source = [
        "ssh",
        "-o",
        "BatchMode=yes",
        config.source_image_host,
        "sudo",
        "-n",
        "docker",
        "save",
        config.source_image_ref,
    ]
    probe = subprocess.run(
        ["ssh", *_ssh_target_args(state, port), "sudo -n docker image inspect --format '{{.Id}}' " + shlex.quote(config.source_image_ref)],
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode == 0 and probe.stdout.strip() == image["id"]:
        return
    _pipe_commands(source, destination)
    got = _run(
        ["ssh", *_ssh_target_args(state, port), "sudo -n docker image inspect --format '{{.Id}}' " + shlex.quote(config.source_image_ref)]
    ).stdout.strip()
    if got != image["id"]:
        raise AwsFleetError(f"worker {index + 1} image identity mismatch: {got}")


def _send_checkpoint(
    config: AwsFleetConfig,
    state: Path,
    index: int,
    checkpoint: dict[str, Any],
) -> str:
    digest_hex = checkpoint["digest"].removeprefix("sha256:")
    basename = re.sub(r"[^A-Za-z0-9._-]+", "-", Path(checkpoint["path"]).name)
    target = f"/opt/pokebot/checkpoint/sha256-{digest_hex}-{basename}"
    temp = f"/tmp/pokebot-checkpoint-{digest_hex}.partial"
    port = config.ssh_tunnel_ports[index]
    remote_args = _ssh_target_args(state, port)
    verify = (
        f"sudo -n test -f {shlex.quote(target)} && "
        f"printf '%s  %s\\n' {shlex.quote(digest_hex)} {shlex.quote(target)} | "
        "sudo -n sha256sum -c - >/dev/null"
    )
    if subprocess.run(["ssh", *remote_args, verify], check=False).returncode == 0:
        return target
    remote_receive = " && ".join(
        [
            f"cat > {shlex.quote(temp)}",
            f"printf '%s  %s\\n' {shlex.quote(digest_hex)} {shlex.quote(temp)} | sha256sum -c -",
            f"sudo -n test ! -e {shlex.quote(target)}",
            f"sudo -n install -m 0444 -o root -g root {shlex.quote(temp)} {shlex.quote(target)}",
            f"rm -f -- {shlex.quote(temp)}",
        ]
    )
    local_checkpoint = Path(str(checkpoint["path"]))
    if local_checkpoint.is_file():
        source = ["cat", "--", str(local_checkpoint)]
    else:
        source = [
            "ssh",
            "-o",
            "BatchMode=yes",
            config.source_checkpoint_host,
            "cat",
            "--",
            checkpoint["path"],
        ]
    _pipe_commands(source, ["ssh", *remote_args, remote_receive])
    if subprocess.run(["ssh", *remote_args, verify], check=False).returncode != 0:
        raise AwsFleetError(f"worker {index + 1} checkpoint verification failed")
    return target


def _start_worker(
    config: AwsFleetConfig,
    state: Path,
    index: int,
    checkpoint_path: str,
) -> None:
    port = config.ssh_tunnel_ports[index]
    env = {
        "POKEBOT_CHECKPOINT": checkpoint_path,
        "POKEBOT_REMOTE_CHECKPOINT_ROOT": "/opt/pokebot/checkpoint",
        "POKEBOT_REMOTE_ACTIVE_CHECKPOINT_FILE": "/opt/pokebot/state/active-checkpoint.json",
        "POKEBOT_REMOTE_WORKER_ARM_FILE": "/opt/pokebot/state/REMOTE_WORKER_ARMED",
        "SIM_WORKERS": str(config.worker_capacity),
        "SIM_DEFAULT_WORKERS": str(config.worker_capacity),
        "ELMO_SIM_WORKER_CEILING": "60",
        "LEAF_SERVERS": "2",
        "LEAF_GPU": "cuda:0",
        "LEAF_MAX_BATCH": "256",
        "POKEBOT_REMOTE_MAX_CONNECTIONS": "128",
        "POKEBOT_REMOTE_TREE_RSS_LIMIT_GB": "190",
        "POKEBOT_REMOTE_MIN_FREE_RAM_GB": "24",
    }
    env_args = " ".join(f"-e {shlex.quote(k + '=' + v)}" for k, v in env.items())
    command = "\n".join(
        [
            "set -eu",
            "printf 'armed\\n' | sudo -n tee /opt/pokebot/state/REMOTE_WORKER_ARMED >/dev/null",
            "sudo -n docker rm -f pokebot-remote-worker >/dev/null 2>&1 || true",
            "sudo -n docker run -d --name pokebot-remote-worker --restart unless-stopped "
            "--network host --gpus all --cpus 60 --memory 224g --memory-swap 224g "
            "--pids-limit 4096 --shm-size 4g "
            "-v /opt/pokebot/checkpoint:/opt/pokebot/checkpoint:ro "
            "-v /opt/pokebot/state:/opt/pokebot/state "
            "-v /opt/pokebot/logs:/workspace/runtime-logs "
            + env_args
            + " "
            + shlex.quote(config.source_image_ref),
        ]
    )
    _run(["ssh", *_ssh_target_args(state, port), command], timeout=120)


def _register(
    config: AwsFleetConfig,
    state: Path,
    count: int,
) -> dict[str, Any]:
    discovered: list[DiscoveredBackend] = []
    for index, worker_port in enumerate(config.local_tunnel_ports[:count]):
        _wait_port(worker_port, timeout=600)
        row = discover_backend(
            f"127.0.0.1:{worker_port}",
            capacity=config.worker_capacity,
            backend_id=f"aws-g6-16xl-{index + 1:02d}",
            trainer_root="/home/inzi/poke-bot-agent/",
            worker_root="/workspace/",
            timeout_s=30,
        )
        entry = dict(row.entry)
        entry["checkpoint_path_template"] = (
            "/opt/pokebot/checkpoint/sha256-{digest}-{basename}"
        )
        entry["checkpoint_stage"] = {
            "mode": "ssm_ssh_v1",
            "ssh_host": "127.0.0.1",
            "ssh_port": config.ssh_tunnel_ports[index],
            "ssh_user": "ec2-user",
            "identity_file": str(state / "id_ed25519"),
            "known_hosts_file": str(state / "known_hosts"),
        }
        discovered.append(DiscoveredBackend(entry=entry, health=row.health))
    return add_backends(config.gateway_manifest, discovered, replace=True)


def command_check(
    config: AwsFleetConfig,
    *,
    count: Optional[int] = None,
    cost_ceiling: Optional[float] = None,
) -> dict[str, Any]:
    selected = int(count or config.instance_count)
    selected_ceiling = float(
        cost_ceiling if cost_ceiling is not None else _cost_ceiling(config, selected)
    )
    prereqs = prerequisite_report()
    missing = [
        name
        for name in (
            "aws",
            "ssh",
            "session-manager-plugin",
            "ssh-keygen",
            "ssh-keyscan",
        )
        if shutil.which(name) is None
    ]
    gateway_ready = False
    gateway: dict[str, Any]
    if not config.gateway_manifest.is_file():
        gateway = {
            "ready": False,
            "error": f"missing active gateway manifest: {config.gateway_manifest}",
        }
    else:
        try:
            manifest = FleetManifest.load(config.gateway_manifest)
            required_capacity = selected * config.worker_capacity
            gateway_ready = bool(
                manifest.activation_allowed
                and manifest.fleet_worker_ceiling >= required_capacity
            )
            gateway = {
                "ready": gateway_ready,
                "activation_allowed": manifest.activation_allowed,
                "bind": f"{manifest.bind_host}:{manifest.bind_port}",
                "fleet_worker_ceiling": manifest.fleet_worker_ceiling,
                "required_aws_capacity": required_capacity,
                "instruction": (
                    None
                    if gateway_ready
                    else "complete the separate one-time trainer gateway adoption first"
                ),
            }
        except (OSError, ValueError) as exc:
            gateway = {"ready": False, "error": f"invalid gateway manifest: {exc}"}
    output: dict[str, Any] = {
        "ok": not missing and gateway_ready,
        "config": config.redacted(),
        "gateway": gateway,
        "missing_tools": missing,
        "prerequisites": prereqs,
        "instance_count": selected,
        "cost_ceiling_usd": selected_ceiling,
        "within_configured_limit": selected_ceiling <= config.spend_limit_usd,
    }
    if not missing:
        output["aws"] = _cloud_preflight(config, selected)
        output["checkpoint"] = _checkpoint_info(config)
        output["image"] = _image_info(config)
    print(json.dumps(output, indent=2, sort_keys=True))
    if missing or not gateway_ready or not output["within_configured_limit"]:
        raise SystemExit(2)
    return output


def command_launch(
    config: AwsFleetConfig,
    confirmation: Optional[float],
    *,
    count: Optional[int] = None,
) -> None:
    selected = int(count or config.instance_count)
    if not config.activation_allowed:
        raise AwsFleetError("config.activation_allowed=false; refusing billable launch")
    state = _state_root(config)
    cost_ceiling = _expansion_cost_ceiling(config, state, selected)
    _assert_budget(config, confirmation, selected, ceiling=cost_ceiling)
    preflight = command_check(config, count=selected, cost_ceiling=cost_ceiling)
    private, public = _ensure_key(state)
    del private
    checkpoint = _checkpoint_info(config)
    image = _image_info(config)
    existing_receipt = state / "launch-receipt.json"
    prior_count = 0
    if existing_receipt.is_file():
        try:
            prior_receipt = json.loads(existing_receipt.read_text(encoding="utf-8"))
            expire_at_epoch = int(prior_receipt["expire_at_epoch"])
            prior_count = len(prior_receipt["instance_ids"])
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise AwsFleetError("existing launch receipt has no valid expiry") from exc
    else:
        expire_at_epoch = int(time.time()) + config.max_runtime_hours * 3600
    if prior_count > selected:
        raise AwsFleetError(
            f"refusing implicit downscale from {prior_count} to {selected}; use stop"
        )
    if expire_at_epoch <= int(time.time()):
        raise AwsFleetError("fleet expiry has passed; stop it before starting a new burst")
    _deploy(
        config,
        public,
        selected,
        str(preflight["aws"]["selected_availability_zone"]),
        expire_at_epoch,
    )
    instance_ids = _stack_outputs(config)
    if len(instance_ids) != selected:
        raise AwsFleetError("stack worker count differs from requested count")
    _install_tunnel_units(config, instance_ids, state)
    _capture_host_keys(config, state, selected)
    target_paths: list[Optional[str]] = [None] * selected

    def initialize(index: int) -> None:
        _send_image(config, state, index, image)
        target_paths[index] = _send_checkpoint(config, state, index, checkpoint)
        _start_worker(config, state, index, str(target_paths[index]))

    new_indexes = list(range(prior_count, selected))
    if new_indexes:
        with ThreadPoolExecutor(max_workers=min(2, len(new_indexes))) as pool:
            list(pool.map(initialize, new_indexes))
    result = _register(config, state, selected)
    receipt = {
        "schema": "poke_bot.aws_remote_fleet_launch_receipt/v1",
        "stack_name": config.stack_name,
        "instance_ids": instance_ids,
        "checkpoint": checkpoint,
        "image": image,
        "worker_ports": list(config.local_tunnel_ports[:selected]),
        "ssh_ports": list(config.ssh_tunnel_ports[:selected]),
        "registry": result,
        "cost_ceiling_usd": cost_ceiling,
        "planned_runtime_hours": config.max_runtime_hours,
        "expire_at_epoch": expire_at_epoch,
        "auto_terminate_hours": config.max_runtime_hours,
        "receipt_written_at_epoch": int(time.time()),
    }
    write_json_atomic(state / "launch-receipt.json", receipt, mode=0o600)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def command_status(config: AwsFleetConfig) -> None:
    state = _state_root(config)
    value: dict[str, Any] = {"stack_name": config.stack_name}
    try:
        value["instance_ids"] = _stack_outputs(config)
    except AwsFleetError as exc:
        value["stack_error"] = str(exc)
    if config.gateway_manifest.exists():
        manifest = FleetManifest.load(config.gateway_manifest)
        value["gateway"] = {
            "activation_allowed": manifest.activation_allowed,
            "backends": [backend.id for backend in manifest.backends],
            "capacity": sum(b.capacity for b in manifest.backends if b.enabled),
        }
    value["receipt"] = str(state / "launch-receipt.json")
    print(json.dumps(value, indent=2, sort_keys=True))


def command_stop(config: AwsFleetConfig) -> None:
    # CloudFormation deletion terminates only instances in this exact stack.
    _run(
        [
            "aws",
            "--profile",
            config.aws_profile,
            "--region",
            config.region,
            "cloudformation",
            "delete-stack",
            "--stack-name",
            config.stack_name,
        ]
    )
    stopped_units: list[str] = []
    for index in range(4):
        for kind in ("worker-tunnel", "ssh-tunnel"):
            unit = _unit_name(kind, index)
            result = subprocess.run(
                ["systemctl", "--user", "disable", "--now", unit],
                text=True,
                capture_output=True,
                check=False,
            )
            if result.returncode == 0:
                stopped_units.append(unit)
    print(
        json.dumps(
            {
                "ok": True,
                "delete_requested": config.stack_name,
                "managed_tunnel_units_stopped": stopped_units,
            }
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="copy the one-file AWS input template")
    init.add_argument("output", type=Path)
    init.add_argument("--profile", default="pokebot-burst")
    init.add_argument("--region", default="us-east-1")
    for name in ("check", "status", "stop"):
        cmd = sub.add_parser(name)
        cmd.add_argument("config", type=Path)
    launch = sub.add_parser("launch")
    launch.add_argument("config", type=Path)
    launch.add_argument("--confirm-cost-limit", type=float, required=True)
    expand = sub.add_parser("add-two")
    expand.add_argument("config", type=Path)
    expand.add_argument("--confirm-cost-limit", type=float, required=True)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "init":
            output = args.output.expanduser().resolve()
            if output.exists():
                raise AwsFleetError(f"refusing to overwrite existing config: {output}")
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.profile):
                raise AwsFleetError("--profile is invalid")
            if not re.fullmatch(r"[a-z]{2}(?:-gov)?-[a-z]+-\d+", args.region):
                raise AwsFleetError("--region is invalid")
            value = json.loads(EXAMPLE.read_text(encoding="utf-8"))
            value["aws_profile"] = args.profile
            value["region"] = args.region
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json_atomic(output, value, mode=0o600)
            print(output)
            return 0
        config = AwsFleetConfig.load(args.config)
        if args.command == "check":
            command_check(config)
        elif args.command == "launch":
            command_launch(config, args.confirm_cost_limit)
        elif args.command == "add-two":
            command_launch(config, args.confirm_cost_limit, count=4)
        elif args.command == "status":
            command_status(config)
        elif args.command == "stop":
            command_stop(config)
        return 0
    except AwsFleetError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
