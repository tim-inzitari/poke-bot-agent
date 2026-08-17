from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from poke_bot.aws_remote_fleet import (
    AWS_FLEET_CONFIG_SCHEMA,
    AwsFleetConfig,
    AwsFleetError,
    stage_checkpoint_for_backend,
)
from scripts.aws_remote_fleet import (
    _assert_budget,
    _cost_ceiling,
    _expansion_cost_ceiling,
)


def _config(tmp_path: Path, **updates) -> AwsFleetConfig:
    value = {
        "schema": AWS_FLEET_CONFIG_SCHEMA,
        "activation_allowed": False,
        "aws_profile": "pokebot-burst",
        "region": "us-east-1",
        "vpc_id": "vpc-0123456789abcdef0",
        "subnet_id": "subnet-0123456789abcdef0",
        "instance_type": "g6.16xlarge",
        "instance_count": 2,
        "market": "on-demand",
        "root_volume_gib": 300,
        "worker_capacity": 48,
        "max_runtime_hours": 48,
        "ami_ssm_parameter": "/aws/service/ecs/optimized-ami/amazon-linux-2023/gpu/recommended/image_id",
        "stack_name": "pokebot-aws-remote-fleet",
        "local_tunnel_ports": [18765, 18766, 18767, 18768],
        "ssh_tunnel_ports": [19022, 19023, 19024, 19025],
        "gateway_manifest": "/etc/pokebot/remote-fleet-gateway.active.json",
        "source_image_host": "elmo",
        "source_image_ref": "poke-bot-truenas-worker:r125-checkpoint-digest-verify-v2",
        "source_checkpoint_host": "inzi",
        "source_checkpoint_path": "auto",
        "spend_limit_usd": 500.0,
    }
    value.update(updates)
    path = tmp_path / "aws.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return AwsFleetConfig.load(path)


def test_two_node_config_retains_four_expansion_ports(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert config.instance_count == 2
    assert config.local_tunnel_ports == (18765, 18766, 18767, 18768)
    assert config.ssh_tunnel_ports == (19022, 19023, 19024, 19025)


def test_managed_network_needs_no_vpc_or_subnet_ids(tmp_path: Path) -> None:
    config = _config(tmp_path, vpc_id="auto", subnet_id="auto")
    assert config.vpc_id == "auto"
    assert config.subnet_id == "auto"

    with pytest.raises(AwsFleetError, match="both be 'auto'"):
        _config(tmp_path, vpc_id="auto")


def test_budget_guard_allows_two_and_refuses_four_under_500(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assert _cost_ceiling(config, 2) == 404.0
    _assert_budget(config, 500.0, 2)
    with pytest.raises(AwsFleetError, match="exceeds configured"):
        _assert_budget(config, 500.0, 4)
    with pytest.raises(AwsFleetError, match="exactly matching"):
        _assert_budget(config, 499.0, 2)


def test_late_add_two_reuses_deadline_and_counts_prior_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(tmp_path)
    state = tmp_path / "state"
    state.mkdir()
    now = 2_000_000_000
    (state / "launch-receipt.json").write_text(
        json.dumps(
            {
                "instance_ids": ["i-00000001", "i-00000002"],
                "expire_at_epoch": now + 12 * 3600,
                "planned_runtime_hours": 48,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr("scripts.aws_remote_fleet.time.time", lambda: now)
    assert _expansion_cost_ceiling(config, state, 4) == 500.0
    _assert_budget(config, 500.0, 4, ceiling=500.0)


def test_config_rejects_credentials_and_unknown_fields(tmp_path: Path) -> None:
    with pytest.raises(AwsFleetError, match="unknown fields"):
        _config(tmp_path, aws_access_key_id="must-not-live-here")


def test_direct_checkpoint_stage_verifies_before_remote_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "candidate.pt"
    source.write_bytes(b"immutable checkpoint")
    identity = tmp_path / "id_ed25519"
    known = tmp_path / "known_hosts"
    identity.write_text("key", encoding="utf-8")
    known.write_text("host", encoding="utf-8")
    import hashlib

    digest = "sha256:" + hashlib.sha256(source.read_bytes()).hexdigest()
    backend = SimpleNamespace(
        checkpoint_stage=SimpleNamespace(
            mode="ssm_ssh_v1",
            ssh_host="127.0.0.1",
            ssh_port=19022,
            ssh_user="ec2-user",
            identity_file=str(identity),
            known_hosts_file=str(known),
        )
    )
    commands: list[list[str]] = []

    def fake_run(command, **_kwargs):
        commands.append(list(command))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("poke_bot.aws_remote_fleet.subprocess.run", fake_run)
    target = f"/opt/pokebot/checkpoint/sha256-{digest[7:]}-candidate.pt"
    stage_checkpoint_for_backend(source, digest, backend, target)
    assert commands[0][0] == "scp"
    assert commands[1][0] == "ssh"
    assert target in commands[1][-1]


def test_direct_checkpoint_stage_rejects_wrong_local_digest(tmp_path: Path) -> None:
    source = tmp_path / "candidate.pt"
    source.write_bytes(b"wrong")
    backend = SimpleNamespace(
        checkpoint_stage=SimpleNamespace(
            mode="ssm_ssh_v1",
            ssh_host="127.0.0.1",
            ssh_port=19022,
            ssh_user="ec2-user",
            identity_file=str(tmp_path / "missing"),
            known_hosts_file=str(tmp_path / "missing-known"),
        )
    )
    with pytest.raises(AwsFleetError, match="local checkpoint digest mismatch"):
        stage_checkpoint_for_backend(
            source,
            "sha256:" + "0" * 64,
            backend,
            "/opt/pokebot/checkpoint/candidate.pt",
        )
