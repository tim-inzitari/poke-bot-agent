#!/usr/bin/env python3
"""Collect a protected Alakazam rule-teacher corpus for neural distillation."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poke_bot.baselines_runtime import (
    baseline_spec_payload,
    ensure_baselines_installed,
    load_manifest,
)
from poke_bot.ladder_deck_mix import (
    load_ladder_deck_mix,
    load_ladder_deck_representatives,
)
from poke_bot.replay_writer import OrderedReplayWriter
from poke_bot.rule_teacher import (
    RULE_TEACHER_SCHEMA,
    _worker_rule_teacher_game,
    atomic_json,
    build_rule_teacher_jobs,
    deck_digest,
    file_digest,
    result_metadata,
    summarize_journal,
    validate_rule_teacher_corpus,
)
from poke_bot.worker_pool import WorkerPool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--games", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=910_000)
    parser.add_argument("--job-offset", type=int, default=0)
    parser.add_argument("--teacher", default="ryota-alakazam-best5")
    parser.add_argument("--opponent", default="iono")
    parser.add_argument("--archetype", default="alakazam")
    parser.add_argument("--outcome-filter", choices=("wins", "all"), default="wins")
    parser.add_argument("--timeout-s", type=int, default=180)
    parser.add_argument("--min-valid-frac", type=float, default=0.99)
    parser.add_argument("--queue-depth", type=int, default=64)
    parser.add_argument("--fsync-batch", type=int, default=8)
    return parser.parse_args(argv)


def resolve_deck(archetype: str) -> tuple[list[int], dict]:
    mix_path = ROOT / "data/training_mixes/top_ladder.v1.json"
    reps_path = ROOT / "data/training_mixes/top_ladder_representatives.v1.json"
    mix = load_ladder_deck_mix(mix_path)
    representatives = load_ladder_deck_representatives(reps_path)
    matches = [
        item
        for item in representatives.bind(mix)
        if item.bucket.deck_id == str(archetype)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one pinned representative for {archetype!r}")
    item = matches[0]
    cards = list(item.card_ids)
    return cards, {
        "archetype": str(archetype),
        "source": "pinned_top_ladder_modal_representative",
        "mix": str(mix_path.resolve()),
        "representatives": str(reps_path.resolve()),
        "digest": deck_digest(cards),
        "canonical_multiset_sha256": item.canonical_multiset_sha256,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("workers must be positive")
    if not 0.0 < float(args.min_valid_frac) <= 1.0:
        raise ValueError("min-valid-frac must be in (0, 1]")

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus = out_dir / "teacher_wins.jsonl"
    partial = out_dir / "teacher_wins.jsonl.partial"
    report_path = out_dir / "PROTECTED_RULE_TEACHER_CORPUS.json"
    if corpus.is_file() and report_path.is_file():
        report = json.loads(report_path.read_text())
        if file_digest(corpus) != report.get("corpus", {}).get("digest"):
            raise RuntimeError("protected teacher corpus digest mismatch")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    if corpus.exists() or report_path.exists():
        raise RuntimeError("incomplete finalized teacher artifact set")

    specs = {spec.id: spec for spec in ensure_baselines_installed(load_manifest())}
    teacher = specs.get(str(args.teacher))
    opponent = specs.get(str(args.opponent))
    if teacher is None or opponent is None:
        raise ValueError(
            f"missing teacher/opponent from baseline manifest: "
            f"{args.teacher!r}/{args.opponent!r}"
        )
    teacher_payload = baseline_spec_payload(teacher)
    opponent_payload = baseline_spec_payload(opponent)
    cards, deck_identity = resolve_deck(str(args.archetype))
    jobs = build_rule_teacher_jobs(
        games=int(args.games),
        seed=int(args.seed),
        job_offset=int(args.job_offset),
        teacher_spec=teacher_payload,
        opponent_spec=opponent_payload,
        teacher_deck=cards,
        archetype=str(args.archetype),
        outcome_filter=str(args.outcome_filter),
        timeout_s=int(args.timeout_s),
    )

    os.environ["POKEBOT_WORKER_CPU_ONLY"] = "1"
    started = time.perf_counter()
    writer = OrderedReplayWriter(
        partial,
        expected_jobs=len(jobs),
        queue_depth=int(args.queue_depth),
        fsync_batch=int(args.fsync_batch),
    )
    remaining = jobs[writer.resume_index :]
    from tqdm.auto import tqdm

    bar = tqdm(
        total=len(jobs),
        initial=writer.resume_index,
        desc="rule teacher",
        unit="game",
        dynamic_ncols=False,
        mininterval=0.5,
    )
    try:
        with WorkerPool(num_workers=int(args.workers)) as pool:
            for result in pool.imap_unordered(
                _worker_rule_teacher_game, remaining, chunksize=1
            ):
                writer.submit(
                    int(result["job_index"]),
                    result.get("record_json"),
                    result_metadata(result),
                )
                bar.update(1)
                telemetry = writer.telemetry()
                bar.set_postfix(
                    saved=telemetry["written_records"],
                    queue=telemetry["queue_depth"],
                )
        writer.close()
    except BaseException as exc:
        writer.abort(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        bar.close()

    journal = partial.with_suffix(partial.suffix + ".journal")
    accounting = summarize_journal(journal)
    valid_fraction = int(accounting.get("valid_games", 0)) / max(1, len(jobs))
    if valid_fraction < float(args.min_valid_frac):
        raise RuntimeError(
            f"teacher collection valid fraction {valid_fraction:.4f} < "
            f"{float(args.min_valid_frac):.4f}: {accounting.get('errors')}"
        )
    writer.finalize(corpus)
    validation = validate_rule_teacher_corpus(
        corpus,
        expected_deck_digest=str(deck_identity["digest"]),
        expected_teacher=str(args.teacher),
        expected_opponent=str(args.opponent),
        require_wins=str(args.outcome_filter) == "wins",
    )
    if validation["records"] != int(accounting.get("records_written", 0)):
        raise RuntimeError("writer journal/corpus record count mismatch")

    report = {
        "schema": RULE_TEACHER_SCHEMA,
        "protected": True,
        "prune_policy": "never",
        "created_at_unix": time.time(),
        "elapsed_seconds": time.perf_counter() - started,
        "configuration": {
            "games": int(args.games),
            "workers": int(args.workers),
            "seed": int(args.seed),
            "job_offset": int(args.job_offset),
            "outcome_filter": str(args.outcome_filter),
            "teacher": teacher_payload,
            "opponent": opponent_payload,
            "deck": deck_identity,
            "simulator": "competition_libcg",
            "engine_seedable": False,
            "final_agent_runtime": "neural_only",
        },
        "accounting": accounting,
        "validation": validation,
        "corpus": {
            "path": str(corpus),
            "bytes": corpus.stat().st_size,
            "digest": file_digest(corpus),
        },
        "journal": {
            "path": str(journal),
            "bytes": journal.stat().st_size,
            "digest": file_digest(journal),
        },
    }
    atomic_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
