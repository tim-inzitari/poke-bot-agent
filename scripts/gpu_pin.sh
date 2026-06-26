#!/usr/bin/env bash
# Pin training to a GPU by name substring (robust across nvidia-smi vs PyTorch index drift).
# Usage: source scripts/gpu_pin.sh && pin_training_gpu "3080"
pin_training_gpu() {
  local pattern="$1"
  if [[ -z "${pattern}" ]]; then
    echo "pin_training_gpu: missing name pattern" >&2
    return 1
  fi

  unset CUDA_VISIBLE_DEVICES
  local idx
  idx="$(
    python3 - "${pattern}" <<'PY'
import sys
import torch

pattern = sys.argv[1].lower()
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")
for i in range(torch.cuda.device_count()):
    if pattern in torch.cuda.get_device_name(i).lower():
        print(i)
        break
else:
    names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    raise SystemExit(f"no GPU matching {pattern!r}; visible: {names}")
PY
  )" || return 1

  export CUDA_VISIBLE_DEVICES="${idx}"
  export TORCH_DEVICE=cuda:0

  python3 - <<'PY'
import os
import torch
from poke_agent.device import describe_torch_device, torch_device

name = torch.cuda.get_device_name(0)
print(
    f"pinned training GPU: {describe_torch_device(torch_device())} "
    f"(CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']!r})"
)
PY
}
