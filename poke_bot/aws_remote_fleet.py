"""AWS helpers for the staged elastic remote-fleet gateway.

The module has two deliberately separate responsibilities:

* strict parsing of the one-file AWS fleet configuration used by the operator
  wrapper; and
* create-only direct SSM/SSH checkpoint staging used by the gateway immediately
  before its existing verify-before-reload transaction.

It never stores AWS credentials.  The AWS CLI profile named in the immutable
configuration supplies credentials through the normal AWS credential chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


AWS_FLEET_CONFIG_SCHEMA = "poke_bot.aws_remote_fleet/v1"
DEFAULT_AMI_PARAMETER = (
    "/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id"
)
_CONFIG_FIELDS = {
    "schema",
    "activation_allowed",
    "aws_profile",
    "region",
    "vpc_id",
    "subnet_id",
    "instance_type",
    "instance_count",
    "market",
    "root_volume_gib",
    "worker_capacity",
    "max_runtime_hours",
    "ami_ssm_parameter",
    "stack_name",
    "local_tunnel_ports",
    "ssh_tunnel_ports",
    "gateway_manifest",
    "source_image_host",
    "source_image_ref",
    "source_checkpoint_host",
    "source_checkpoint_path",
    "spend_limit_usd",
}


class AwsFleetError(RuntimeError):
    """A local or AWS fleet preflight failed safely."""


def _strict_fields(value: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise AwsFleetError(f"{where} has unknown fields: {', '.join(unknown)}")


def _safe_token(value: Any, pattern: str, where: str) -> str:
    text = str(value or "").strip()
    if not re.fullmatch(pattern, text):
        raise AwsFleetError(f"{where} is invalid")
    return text


@dataclass(frozen=True)
class AwsFleetConfig:
    path: Path
    activation_allowed: bool
    aws_profile: str
    region: str
    vpc_id: str
    subnet_id: str
    instance_type: str
    instance_count: int
    market: str
    root_volume_gib: int
    worker_capacity: int
    max_runtime_hours: int
    ami_ssm_parameter: str
    stack_name: str
    local_tunnel_ports: tuple[int, ...]
    ssh_tunnel_ports: tuple[int, ...]
    gateway_manifest: Path
    source_image_host: str
    source_image_ref: str
    source_checkpoint_host: str
    source_checkpoint_path: str
    spend_limit_usd: float

    @classmethod
    def load(cls, path: Path) -> "AwsFleetConfig":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AwsFleetError(f"cannot read AWS fleet config {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise AwsFleetError("AWS fleet config must be an object")
        _strict_fields(value, _CONFIG_FIELDS, "config")
        if value.get("schema") != AWS_FLEET_CONFIG_SCHEMA:
            raise AwsFleetError(
                f"config.schema must be {AWS_FLEET_CONFIG_SCHEMA!r}"
            )
        if not isinstance(value.get("activation_allowed", False), bool):
            raise AwsFleetError("config.activation_allowed must be a boolean")
        profile = _safe_token(
            value.get("aws_profile"), r"[A-Za-z0-9_.-]+", "config.aws_profile"
        )
        region = _safe_token(
            value.get("region"), r"[a-z]{2}(?:-gov)?-[a-z]+-\d+", "config.region"
        )
        raw_vpc_id = str(value.get("vpc_id") or "auto").strip()
        raw_subnet_id = str(value.get("subnet_id") or "auto").strip()
        if (raw_vpc_id == "auto") != (raw_subnet_id == "auto"):
            raise AwsFleetError(
                "config.vpc_id and config.subnet_id must both be 'auto' or both be IDs"
            )
        if raw_vpc_id == "auto":
            vpc_id = "auto"
            subnet_id = "auto"
        else:
            vpc_id = _safe_token(raw_vpc_id, r"vpc-[0-9a-f]+", "config.vpc_id")
            subnet_id = _safe_token(
                raw_subnet_id, r"subnet-[0-9a-f]+", "config.subnet_id"
            )
        instance_type = str(value.get("instance_type") or "g6.16xlarge").strip()
        if instance_type != "g6.16xlarge":
            raise AwsFleetError(
                "this staged burst is pinned to instance_type='g6.16xlarge'"
            )
        count = int(value.get("instance_count", 2))
        if count not in {2, 4}:
            raise AwsFleetError("config.instance_count must be 2 or 4")
        market = str(value.get("market") or "on-demand").strip()
        if market not in {"on-demand", "spot"}:
            raise AwsFleetError("config.market must be 'on-demand' or 'spot'")
        volume = int(value.get("root_volume_gib", 300))
        if not 100 <= volume <= 2048:
            raise AwsFleetError("config.root_volume_gib must be 100..2048")
        capacity = int(value.get("worker_capacity", 48))
        if not 4 <= capacity <= 60:
            raise AwsFleetError("config.worker_capacity must be 4..60")
        hours = int(value.get("max_runtime_hours", 48))
        if not 1 <= hours <= 168:
            raise AwsFleetError("config.max_runtime_hours must be 1..168")
        ami = str(value.get("ami_ssm_parameter") or DEFAULT_AMI_PARAMETER).strip()
        if not ami.startswith("/aws/service/ecs/optimized-ami/"):
            raise AwsFleetError(
                "config.ami_ssm_parameter must be an ECS public AMI parameter"
            )
        stack = _safe_token(
            value.get("stack_name") or "pokebot-aws-remote-fleet",
            r"[A-Za-z][A-Za-z0-9-]{0,127}",
            "config.stack_name",
        )
        raw_ports = value.get("local_tunnel_ports") or [18765, 18766, 18767, 18768]
        if not isinstance(raw_ports, list) or len(raw_ports) < count:
            raise AwsFleetError("config.local_tunnel_ports must cover every instance")
        # Retain the optional expansion ports even when the initial fleet has
        # two members.  ``add-two`` must not require rewriting networking
        # coordinates while a two-node stack is live.
        ports = tuple(int(port) for port in raw_ports)
        if len(set(ports)) != len(ports) or any(
            not 1024 <= p <= 65535 for p in ports
        ):
            raise AwsFleetError(
                "config.local_tunnel_ports must be unique ports 1024..65535"
            )
        raw_ssh_ports = value.get("ssh_tunnel_ports") or [19022, 19023, 19024, 19025]
        if not isinstance(raw_ssh_ports, list) or len(raw_ssh_ports) < count:
            raise AwsFleetError("config.ssh_tunnel_ports must cover every instance")
        ssh_ports = tuple(int(port) for port in raw_ssh_ports)
        if (
            len(set(ssh_ports)) != len(ssh_ports)
            or any(not 1024 <= p <= 65535 for p in ssh_ports)
            or set(ports).intersection(ssh_ports)
        ):
            raise AwsFleetError(
                "config.ssh_tunnel_ports must be unique and not overlap worker ports"
            )
        manifest = Path(
            str(
                value.get("gateway_manifest")
                or "/etc/pokebot/remote-fleet-gateway.active.json"
            )
        )
        if not manifest.is_absolute():
            raise AwsFleetError("config.gateway_manifest must be absolute")
        source_image_host = _safe_token(
            value.get("source_image_host") or "elmo",
            r"[A-Za-z0-9_.@-]+",
            "config.source_image_host",
        )
        source_image_ref = str(
            value.get("source_image_ref")
            or "poke-bot-truenas-worker:r125-checkpoint-digest-verify-v2"
        ).strip()
        if not re.fullmatch(r"[A-Za-z0-9._/-]+:[A-Za-z0-9._-]+", source_image_ref):
            raise AwsFleetError("config.source_image_ref is invalid")
        source_checkpoint_host = _safe_token(
            value.get("source_checkpoint_host") or "inzi",
            r"[A-Za-z0-9_.@-]+",
            "config.source_checkpoint_host",
        )
        source_checkpoint_path = str(
            value.get("source_checkpoint_path") or "auto"
        ).strip()
        if source_checkpoint_path != "auto" and not source_checkpoint_path.startswith(
            "/"
        ):
            raise AwsFleetError(
                "config.source_checkpoint_path must be 'auto' or an absolute remote path"
            )
        spend_limit = float(value.get("spend_limit_usd", 500.0))
        if not 50.0 <= spend_limit <= 10000.0:
            raise AwsFleetError("config.spend_limit_usd must be 50..10000")
        return cls(
            path=path.resolve(),
            activation_allowed=bool(value.get("activation_allowed", False)),
            aws_profile=profile,
            region=region,
            vpc_id=vpc_id,
            subnet_id=subnet_id,
            instance_type=instance_type,
            instance_count=count,
            market=market,
            root_volume_gib=volume,
            worker_capacity=capacity,
            max_runtime_hours=hours,
            ami_ssm_parameter=ami,
            stack_name=stack,
            local_tunnel_ports=ports,
            ssh_tunnel_ports=ssh_ports,
            gateway_manifest=manifest,
            source_image_host=source_image_host,
            source_image_ref=source_image_ref,
            source_checkpoint_host=source_checkpoint_host,
            source_checkpoint_path=source_checkpoint_path,
            spend_limit_usd=spend_limit,
        )

    def redacted(self) -> dict[str, Any]:
        return {
            "schema": AWS_FLEET_CONFIG_SCHEMA,
            "activation_allowed": self.activation_allowed,
            "aws_profile": self.aws_profile,
            "region": self.region,
            "vpc_id": self.vpc_id,
            "subnet_id": self.subnet_id,
            "instance_type": self.instance_type,
            "instance_count": self.instance_count,
            "market": self.market,
            "root_volume_gib": self.root_volume_gib,
            "worker_capacity": self.worker_capacity,
            "max_runtime_hours": self.max_runtime_hours,
            "ami_ssm_parameter": self.ami_ssm_parameter,
            "stack_name": self.stack_name,
            "local_tunnel_ports": list(self.local_tunnel_ports),
            "ssh_tunnel_ports": list(self.ssh_tunnel_ports),
            "gateway_manifest": str(self.gateway_manifest),
            "source_image_host": self.source_image_host,
            "source_image_ref": self.source_image_ref,
            "source_checkpoint_host": self.source_checkpoint_host,
            "source_checkpoint_path": self.source_checkpoint_path,
            "spend_limit_usd": self.spend_limit_usd,
        }


class AwsCli:
    """Small JSON AWS CLI wrapper with no credential material in arguments."""

    def __init__(
        self,
        profile: str,
        region: str,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.profile = profile
        self.region = region
        self._runner = runner

    def command(self, *args: str) -> list[str]:
        return [
            "aws",
            "--profile",
            self.profile,
            "--region",
            self.region,
            *args,
        ]

    def run_json(self, *args: str) -> dict[str, Any]:
        command = self.command(*args, "--output", "json")
        result = self._runner(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise AwsFleetError(
                f"AWS command failed ({' '.join(command[:7])} ...): "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )
        try:
            value = json.loads(result.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise AwsFleetError("AWS CLI returned non-JSON output") from exc
        if not isinstance(value, dict):
            raise AwsFleetError("AWS CLI returned a non-object response")
        return value

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        command = self.command(*args)
        result = self._runner(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise AwsFleetError(result.stderr.strip() or result.stdout.strip())
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def prerequisite_report() -> dict[str, Any]:
    return {
        "aws_cli": shutil.which("aws"),
        "session_manager_plugin": shutil.which("session-manager-plugin"),
        "ssh": shutil.which("ssh"),
        "ssh_keygen": shutil.which("ssh-keygen"),
        "ssh_keyscan": shutil.which("ssh-keyscan"),
        "required_for_check": ["aws", "ssh", "ssh-keygen", "ssh-keyscan"],
        "required_for_tunnels": ["aws", "session-manager-plugin", "ssh"],
    }


def _run_checked(
    command: Sequence[str],
    *,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        list(command),
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise AwsFleetError(
            f"command failed ({' '.join(command[:8])} ...): "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result


def stage_checkpoint_for_backend(
    source_path: Path,
    requested_digest: str,
    backend: Any,
    target_path: str,
) -> None:
    """Create-only stage one local checkpoint through an SSM SSH forward.

    This function is called before the gateway opens its verify/reload fanout.
    A failure may leave a private partial or verified storage-local file, but
    it never mutates a resident model or sends reload/pin.
    """

    stage = backend.checkpoint_stage
    if stage is None:
        return
    if stage.mode != "ssm_ssh_v1":
        raise AwsFleetError(f"unsupported checkpoint stage mode {stage.mode!r}")
    source = source_path.expanduser().resolve()
    if not source.is_file():
        raise AwsFleetError(f"checkpoint source is not a file: {source}")
    actual = sha256_file(source)
    if actual != requested_digest:
        raise AwsFleetError(
            f"local checkpoint digest mismatch: {actual} != {requested_digest}"
        )
    destination = Path(target_path)
    if not destination.is_absolute():
        raise AwsFleetError("AWS checkpoint destination must be absolute")
    digest_hex = requested_digest.removeprefix("sha256:")
    temp_path = f"/tmp/pokebot-checkpoint-{digest_hex}.partial"
    identity = Path(stage.identity_file)
    known_hosts = Path(stage.known_hosts_file)
    if not identity.is_file():
        raise AwsFleetError(f"SSH identity file is missing: {identity}")
    if not known_hosts.is_file():
        raise AwsFleetError(f"SSH known-hosts file is missing: {known_hosts}")
    ssh_common = [
        "-i",
        str(identity),
        "-p",
        str(stage.ssh_port),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=15",
    ]
    remote = f"{stage.ssh_user}@{stage.ssh_host}"
    scp_common = list(ssh_common)
    scp_common[scp_common.index("-p")] = "-P"
    _run_checked(["scp", *scp_common, str(source), f"{remote}:{temp_path}"])
    shell = "\n".join(
        [
            "set -eu",
            f"dest={shlex.quote(target_path)}",
            f"tmp={shlex.quote(temp_path)}",
            'sudo -n mkdir -p -- "$(dirname -- "$dest")"',
            'if test -f "$dest" && printf "%s  %s\\n" '
            + shlex.quote(digest_hex)
            + ' "$dest" | sudo -n sha256sum -c - >/dev/null 2>&1; then rm -f -- "$tmp"; exit 0; fi',
            'sudo -n test ! -e "$dest" || { echo "conflicting destination exists" >&2; exit 73; }',
            'printf "%s  %s\\n" '
            + shlex.quote(digest_hex)
            + ' "$tmp" | sha256sum -c -',
            'chmod 0444 "$tmp"',
            'sudo -n install -m 0444 -o root -g root "$tmp" "$dest"',
            'rm -f -- "$tmp"',
            'sudo -n test -f "$dest"',
        ]
    )
    _run_checked(["ssh", *ssh_common, remote, shell])


def write_json_atomic(path: Path, value: Any, mode: int = 0o600) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, sort_keys=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
