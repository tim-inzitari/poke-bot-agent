"""Engine rebuild spike: multi-env / batch-step interfaces for a ptcg_engine fork.

Official ``libcg`` exposes a process-global Game API (no battle handle). This
package defines the Python contracts a forked engine should satisfy so pure-RL
collect can run many battles in one process and call ``step_batch``.

No competition C++ source or ``libcg.so`` is vendored here. ``FakeMultiEnv`` is
a deterministic toy for unit tests; ``LibcgProcessAdapter`` documents how a
future binding would wrap one official battle per OS process until the fork
lands.
"""

from .interfaces import (
    Action,
    BatchObs,
    EnvObs,
    MultiEnv,
    ResetSpec,
)
from .fake_env import FakeMultiEnv
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
    "MultiEnv",
    "ResetSpec",
    "TransitionRecord",
    "assert_parity",
    "fingerprint_select",
    "record_episode",
    "transition_hash",
]
