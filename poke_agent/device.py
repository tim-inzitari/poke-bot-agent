from __future__ import annotations

import os

import torch


def _truthy(value: str | None, *, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def cuda_device_summaries() -> list[dict[str, int | str]]:
    if not torch.cuda.is_available():
        return []
    summaries: list[dict[str, int | str]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        summaries.append({
            "index": index,
            "name": props.name,
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
        })
    return summaries


def preferred_cuda_index() -> int:
    explicit = os.environ.get("POKE_AGENT_CUDA_DEVICE")
    if explicit is not None and explicit.strip() != "":
        return int(explicit)

    summaries = cuda_device_summaries()
    if not summaries:
        return 0
    if not _truthy(os.environ.get("POKE_AGENT_SELECT_LARGEST_CUDA"), default=True):
        return 0
    return int(max(summaries, key=lambda item: int(item["total_memory_bytes"]))["index"])


def torch_device() -> torch.device:
    forced = os.environ.get("POKE_AGENT_DEVICE") or os.environ.get("TORCH_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{preferred_cuda_index()}")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_summary() -> None:
    print("torch device", torch_device())
    if torch.cuda.is_available():
        print("cuda available true")
        for summary in cuda_device_summaries():
            gib = int(summary["total_memory_bytes"]) / (1024 ** 3)
            marker = "*" if int(summary["index"]) == preferred_cuda_index() else " "
            print(
                f"{marker} cuda:{summary['index']} {summary['name']} "
                f"{gib:.1f} GiB sm={summary['multi_processor_count']}"
            )
    else:
        print("cuda available false")
