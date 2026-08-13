#!/usr/bin/env python3
"""Benchmark serial vs deterministic parallel replay preparation in isolation."""

from __future__ import annotations

import argparse
import json
import resource
import shutil
import time
from pathlib import Path

import torch

from poke_bot.device_corpus import DeviceResidentBootstrapCorpus
from poke_bot.pure_rl.dataset_bridge import _dataset_from_shard_serial
from poke_bot.pure_rl.replay_parallel_prepare import (
    _sha256_file,
    build_parallel_replay_pack,
    validate_corpus_parity,
    validate_one_step_result,
    write_validation_receipt,
)


def _rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if __import__("sys").platform == "darwin" else value * 1024


def _one_step(corpus: DeviceResidentBootstrapCorpus) -> dict[str, torch.Tensor]:
    """Small deterministic optimizer probe over an actual packed loss input."""

    torch.manual_seed(773)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    values = corpus.tensor_state()["value_target"][:4096].float().reshape(-1, 1)
    optimizer.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(model(values), values).backward()
    optimizer.step()
    result = {
        f"model.{name}": value.detach().clone()
        for name, value in model.state_dict().items()
    }
    for parameter_index, parameter in enumerate(model.parameters()):
        for name, value in optimizer.state[parameter].items():
            if isinstance(value, torch.Tensor):
                result[f"optimizer.{parameter_index}.{name}"] = value.detach().clone()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", default="1,8,16,32")
    parser.add_argument("--max-context", type=int, default=320)
    parser.add_argument("--exact-card-vocab", type=int)
    parser.add_argument("--memory-reserve-gib", type=float, default=4.0)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)

    serial_started = time.perf_counter()
    serial_dataset = _dataset_from_shard_serial(
        args.source, verify_info_set=False, max_context=args.max_context
    )
    serial = DeviceResidentBootstrapCorpus.from_splits(
        serial_dataset.sequences,
        (),
        device=torch.device("cpu"),
        exact_card_vocab=args.exact_card_vocab,
    )
    serial_seconds = time.perf_counter() - serial_started
    serial_rss = _rss_bytes()
    serial_one_step = _one_step(serial)
    rows = []
    for workers in [int(value) for value in args.workers.split(",") if value]:
        target = output / f"workers-{workers:02d}"
        started = time.perf_counter()
        corpus, manifest = build_parallel_replay_pack(
            args.source,
            target,
            workers=workers,
            max_context=args.max_context,
            exact_card_vocab=args.exact_card_vocab,
            memory_reserve_gib=args.memory_reserve_gib,
            semantic_contract={"benchmark": True},
        )
        elapsed = time.perf_counter() - started
        parity = validate_corpus_parity(serial, corpus)
        optimizer_parity = validate_one_step_result(
            serial_one_step, _one_step(corpus), floating_atol=0.0
        )
        reuse_started = time.perf_counter()
        reused, reused_manifest = build_parallel_replay_pack(
            args.source,
            target,
            workers=workers,
            max_context=args.max_context,
            exact_card_vocab=args.exact_card_vocab,
            memory_reserve_gib=args.memory_reserve_gib,
            semantic_contract={"benchmark": True},
        )
        reuse_seconds = time.perf_counter() - reuse_started
        if not reused_manifest.get("cache_reused"):
            raise RuntimeError("parallel replay cache reuse did not activate")
        validate_corpus_parity(corpus, reused)
        rows.append(
            {
                "workers": workers,
                "seconds": elapsed,
                "speedup_vs_serial": serial_seconds / max(elapsed, 1e-9),
                "peak_rss_bytes": _rss_bytes(),
                "reuse_seconds": reuse_seconds,
                "output_digest": manifest["output_digest"],
                "pack_sha256": manifest["pack_sha256"],
                "rows": manifest["rows"],
                "games": manifest["games"],
                "decisions": manifest["decisions"],
                "parity": parity,
                "one_step_optimizer_parity": optimizer_parity,
            }
        )
    digests = {row["output_digest"] for row in rows}
    if len(digests) != 1:
        raise RuntimeError("parallel output is nondeterministic across worker counts")
    report = {
        "schema": "poke_bot.parallel_replay_benchmark/v1",
        "source": {
            "path": str(args.source.resolve()),
            "sha256": _sha256_file(args.source),
            "size_bytes": args.source.stat().st_size,
        },
        "serial": {
            "seconds": serial_seconds,
            "peak_rss_bytes": serial_rss,
            "games": serial.train_games,
            "decisions": serial.decisions,
        },
        "parallel": rows,
    }
    (output / "benchmark.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    recommended = min(rows, key=lambda row: row["seconds"])
    write_validation_receipt(
        output / "validation-receipt.json",
        source=report["source"],
        code_sha256=_sha256_file(
            Path(__file__).resolve().parents[1]
            / "poke_bot/pure_rl/replay_parallel_prepare.py"
        ),
        worker_count=int(recommended["workers"]),
        output_digest=str(recommended["output_digest"]),
        counts={
            "rows": int(recommended["rows"]),
            "games": int(recommended["games"]),
            "decisions": int(recommended["decisions"]),
        },
        timing={
            "serial_seconds": serial_seconds,
            "parallel_seconds": float(recommended["seconds"]),
            "cache_reuse_seconds": float(recommended["reuse_seconds"]),
        },
        memory={
            "serial_peak_rss_bytes": serial_rss,
            "parallel_peak_rss_bytes": int(recommended["peak_rss_bytes"]),
        },
        validation={
            "worker_counts": [row["workers"] for row in rows],
            "cross_worker_output_digest_equal": True,
            "serial_parallel_tensor_parity": True,
            "identical_seeded_one_step_optimizer_result": True,
            "recommended_workers": int(recommended["workers"]),
        },
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
