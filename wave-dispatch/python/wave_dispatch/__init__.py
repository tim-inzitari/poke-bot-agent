"""wave_dispatch — multi-machine RL collect dispatcher (C++ core).

Install::

    pip install -e /path/to/wave-dispatch

Example::

    from wave_dispatch import JobClient, MidWaveScheduler, run_scheduled_wave
"""

from __future__ import annotations

from wave_dispatch._native import (  # noqa: F401
    PROTO_VERSION,
    DEFAULT_PORT,
    MAX_FRAME_BYTES,
    WaveDispatchError,
    ProtocolError,
    TransportError,
    TimeoutError,
    encode_frame,
    encode_message,
    decode_frame,
    decode_message,
    WorkerInfo,
    JobClient,
    parse_endpoint,
    WorkerFarm,
    ServerConfig,
    serve_forever,
    HardwareSignals,
    sample_hardware_signals,
    SchedulerDecision,
    SchedulerConfig,
    WaveGpsTracker,
    MidWaveScheduler,
    CollectConfig,
    run_scheduled_wave,
)

try:
    from wave_dispatch._native import __version__ as __version__
except ImportError:  # pragma: no cover
    __version__ = "0.1.0"

__all__ = [
    "__version__",
    "PROTO_VERSION",
    "DEFAULT_PORT",
    "MAX_FRAME_BYTES",
    "WaveDispatchError",
    "ProtocolError",
    "TransportError",
    "TimeoutError",
    "encode_frame",
    "encode_message",
    "decode_frame",
    "decode_message",
    "WorkerInfo",
    "JobClient",
    "parse_endpoint",
    "WorkerFarm",
    "ServerConfig",
    "serve_forever",
    "HardwareSignals",
    "sample_hardware_signals",
    "SchedulerDecision",
    "SchedulerConfig",
    "WaveGpsTracker",
    "MidWaveScheduler",
    "CollectConfig",
    "run_scheduled_wave",
]
