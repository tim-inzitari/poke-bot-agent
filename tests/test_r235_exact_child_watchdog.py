"""Focused safety tests for the isolated R235 exact-child watchdog."""

from __future__ import annotations

import sys

from poke_bot.r235_exact_child_watchdog import R235ExactChildWatchdog


def _watchdog(*, timeout: float = 0.4) -> R235ExactChildWatchdog:
    # Linux production uses /proc start ticks.  This explicit opt-in keeps the
    # test portable to macOS without weakening the production CLI path.
    return R235ExactChildWatchdog(
        timeout_seconds=timeout,
        term_grace_seconds=0.08,
        kill_grace_seconds=0.15,
        allow_non_linux_test_identity=True,
    )


def test_new_noninteractive_child_owns_its_own_session_and_group() -> None:
    outcome = _watchdog().run(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(0.08); print('probe-ok')",
        ]
    )

    assert outcome.completed
    assert outcome.identity.process_group_id == outcome.identity.pid
    assert outcome.identity.session_id == outcome.identity.pid
    assert outcome.stdout == b"probe-ok\n"
    assert outcome.containment is None
    assert outcome.fallback_permitted is False


def test_fast_owned_child_still_captures_identity_before_reap() -> None:
    outcome = _watchdog().run([sys.executable, "-c", "print('fast-probe')"])

    assert outcome.completed
    assert outcome.identity.process_group_id == outcome.identity.pid
    assert outcome.identity.session_id == outcome.identity.pid
    assert outcome.stdout == b"fast-probe\n"


def test_timeout_terms_only_the_owned_child_group_then_reaps_it() -> None:
    outcome = _watchdog(timeout=0.12).run(
        [sys.executable, "-c", "import time; time.sleep(10)"]
    )

    assert outcome.status == "deadline_contained"
    assert outcome.containment is not None
    assert outcome.containment.term_sent is True
    assert outcome.containment.reaped is True
    assert outcome.containment.kill_sent is False
    assert outcome.fallback_permitted is True


def test_timeout_escalates_to_kill_only_after_bounded_term_grace() -> None:
    outcome = _watchdog(timeout=0.2).run(
        [
            sys.executable,
            "-c",
            "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(10)",
        ]
    )

    assert outcome.status == "deadline_contained"
    assert outcome.containment is not None
    assert outcome.containment.term_sent is True
    assert outcome.containment.kill_sent is True
    assert outcome.containment.reaped is True
    assert outcome.fallback_permitted is True


def test_precomputed_fallback_is_exposed_only_after_exact_child_reap() -> None:
    outcome, fallback = _watchdog().run_with_precomputed_fallback(
        [sys.executable, "-c", "import sys, time; time.sleep(0.08); sys.exit(9)"],
        precomputed_fallback=[1],
    )

    assert outcome.status == "child_nonzero_reaped"
    assert outcome.fallback_permitted is True
    assert fallback == [1]
