"""Engine rebuild spike: multi-env / batch-step interfaces for CABT throughput.

M0 finding: official ``libcg`` C ABI is already multi-handle (``ApiData*``).
The singleton is Python ``cg.game.Battle.battle_ptr``. Use
:class:`LibcgMultiEnv` for many battles per process without a C++ fork.

M3: additive ``libcg_step_batch.so`` (see ``native/``) provides C++ ``StepBatch``
via dlopen of stock ``libcg.so`` when the full ``ptcg_engine`` tree is unavailable.
``LibcgMultiEnv`` prefers the native path automatically.

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
from .libcg_multi_env import LibcgMultiEnv
from .libcg_step_batch import has_step_batch, load_step_batch_lib
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
    "EnvObs",
    "FakeMultiEnv",
    "LibcgMultiEnv",
    "MultiEnv",
    "ResetSpec",
    "TransitionRecord",
    "assert_parity",
    "fingerprint_select",
    "has_step_batch",
    "load_step_batch_lib",
    "record_episode",
    "transition_hash",
]
