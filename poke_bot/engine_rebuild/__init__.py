"""Engine rebuild spike: multi-env / batch-step interfaces for CABT throughput.

M0 finding: official ``libcg`` C ABI is already multi-handle (``ApiData*``).
The singleton is Python ``cg.game.Battle.battle_ptr``. Use
:class:`LibcgMultiEnv` for many battles per process without a C++ fork; keep
fork work for SoA / ``step_batch`` in C++ / GPU kernels.

Competition source and binaries stay out of git (``kaggle/input/`` ignored).
"""

from .interfaces import (
    Action,
    BatchObs,
    EnvObs,
    MultiEnv,
    ResetSpec,
)
from .fake_env import FakeMultiEnv
from .libcg_batch import BatchedLibcgMultiEnv, load_batch_library
from .libcg_multi_env import LibcgMultiEnv
from .parity import (
    TransitionRecord,
    assert_parity,
    fingerprint_select,
    record_episode,
    transition_hash,
)

__all__ = [
    "Action",
    "BatchObs",
    "BatchedLibcgMultiEnv",
    "EnvObs",
    "FakeMultiEnv",
    "LibcgMultiEnv",
    "MultiEnv",
    "ResetSpec",
    "TransitionRecord",
    "assert_parity",
    "fingerprint_select",
    "record_episode",
    "load_batch_library",
    "transition_hash",
]
