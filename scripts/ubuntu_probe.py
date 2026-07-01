#!/usr/bin/env python3
"""Print an Ubuntu/CUDA readiness report for unattended training."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_agent.config import build_config
from poke_agent.deck import read_deck
from poke_agent.device import cuda_device_summaries, preferred_cuda_index, torch_device
from poke_agent.simulator import load_simulator


def nvidia_smi() -> list[dict[str, str]]:
    if shutil.which("nvidia-smi") is None:
        return []
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,temperature.gpu,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, check=True, capture_output=True)
    except Exception:
        return []
    rows: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append({
            "index": parts[0],
            "name": parts[1],
            "memory_total_mib": parts[2],
            "memory_used_mib": parts[3],
            "temperature_c": parts[4],
            "utilization_percent": parts[5],
        })
    return rows


def latest_checkpoint(checkpoint_dir: Path, fallback: Path) -> Path | None:
    manifest = checkpoint_dir / "manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            for entry in reversed(data.get("iterations", [])):
                candidate = Path(entry.get("saved_checkpoint", ""))
                if candidate.exists():
                    return candidate
        except Exception:
            pass
    candidates = sorted(checkpoint_dir.glob("iter_*.pt"))
    if candidates:
        return candidates[-1]
    return fallback if fallback.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Ubuntu training readiness.")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    args = parser.parse_args()

    config = build_config(ROOT)
    simulator = load_simulator(ROOT)
    deck, deck_source = read_deck(config, ROOT)
    checkpoint_dir = ROOT / config["self_play"]["checkpoint_dir"]
    latest = latest_checkpoint(checkpoint_dir, Path(config["output_path"]))
    disk = shutil.disk_usage(ROOT)

    report = {
        "root": str(ROOT),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "selected_device": str(torch_device()),
        "preferred_cuda_index": preferred_cuda_index() if torch.cuda.is_available() else None,
        "cuda_devices": cuda_device_summaries(),
        "nvidia_smi": nvidia_smi(),
        "simulator": {
            "available": simulator.available,
            "lib_path": simulator.lib_path,
            "error": simulator.error,
        },
        "deck": {
            "source": str(deck_source),
            "cards": len(deck),
        },
        "model": config["model"],
        "training": config["training"],
        "self_play": config["self_play"],
        "checkpoint": {
            "base": str(config["output_path"]),
            "latest": str(latest) if latest is not None else None,
            "checkpoint_dir": str(checkpoint_dir),
        },
        "disk": {
            "total_gib": round(disk.total / (1024 ** 3), 1),
            "free_gib": round(disk.free / (1024 ** 3), 1),
        },
    }

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return

    print("Ubuntu runner probe")
    print("-------------------")
    print(f"root: {report['root']}")
    print(f"platform: {report['platform']}")
    print(f"python: {report['python']}  torch: {report['torch']}  torch_cuda: {report['torch_cuda_version']}")
    print(f"selected device: {report['selected_device']}")
    for gpu in report["nvidia_smi"]:
        print(
            f"gpu {gpu['index']}: {gpu['name']} "
            f"{gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB "
            f"temp={gpu['temperature_c']}C util={gpu['utilization_percent']}%"
        )
    print(f"simulator: available={simulator.available} path={simulator.lib_path} error={simulator.error}")
    print(f"deck: {deck_source} ({len(deck)} cards)")
    print(f"latest checkpoint: {report['checkpoint']['latest']}")
    print(
        "model: "
        f"d_model={config['model']['d_model']} heads={config['model']['heads']} "
        f"layers={config['model']['layers']} kan={config['model'].get('use_kan')}"
    )
    print(
        "training: "
        f"batch_games={config['training']['batch_games']} "
        f"epochs={config['training']['epochs']} "
        f"patience={config['training']['patience']} "
        f"min_delta={config['training']['min_delta']}"
    )
    print(
        "self-play: "
        f"games={config['self_play']['games_per_iteration']} "
        f"eval={config['self_play']['eval_games']} "
        f"baselines={config['self_play']['baseline_names']}"
    )
    print(f"disk free: {report['disk']['free_gib']} GiB / {report['disk']['total_gib']} GiB")


if __name__ == "__main__":
    main()
