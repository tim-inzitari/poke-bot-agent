"""Host/GPU sampling, OOM batch guard, and advisory knob ratchets."""

from .oom import OomGuard, is_cuda_oom
from .plan import Knob, ResourcePlan, ratchet_step
from .sample import sample_cpu, sample_gpus, sample_ram

__version__ = "0.2.0"

__all__ = [
    "Knob",
    "OomGuard",
    "ResourcePlan",
    "is_cuda_oom",
    "ratchet_step",
    "sample_cpu",
    "sample_gpus",
    "sample_ram",
    "__version__",
]
