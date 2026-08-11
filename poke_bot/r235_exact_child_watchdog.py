"""R235 exact-child containment for offline Kaggle package preflights.

This module deliberately owns only a process it creates with ``Popen``.  The
child starts in a new noninteractive session whose process-group leader is the
child PID.  On a bounded timeout the watchdog validates that exact ownership
again, sends TERM to that one process group, then escalates to KILL only if it
has not reaped.  It does not enumerate, inspect, or signal any unrelated
process, service, terminal, SSH, Codex, or editor session.

It is intentionally a small primitive rather than an R225/R229 runtime.  A
future R235 parent may use :meth:`run_with_precomputed_fallback`, but must pass
an already validated fallback value; this module never computes an action.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Generic, Sequence, TypeVar


T = TypeVar("T")

DEFAULT_MAX_CAPTURE_BYTES = 1 << 20


class ExactChildWatchdogError(RuntimeError):
    """The exact owned-child containment contract could not be established."""


class ExactChildIdentityError(ExactChildWatchdogError):
    """The child no longer proves the PID/session/group identity we created."""


class ExactChildUnreapedError(ExactChildWatchdogError):
    """The exact owned child survived both bounded containment windows."""


@dataclass(frozen=True)
class ExactChildIdentity:
    """Identity recorded before the child is allowed to perform probe work."""

    pid: int
    process_group_id: int
    session_id: int
    start_identity_kind: str
    start_identity: str

    def as_dict(self) -> dict[str, object]:
        return {
            "pid": self.pid,
            "process_group_id": self.process_group_id,
            "session_id": self.session_id,
            "start_identity_kind": self.start_identity_kind,
            "start_identity": self.start_identity,
        }


@dataclass(frozen=True)
class ContainmentReport:
    """Bounded TERM/KILL and reaping telemetry for the one owned child group."""

    reason: str
    term_sent: bool
    kill_sent: bool
    reaped: bool
    returncode: int | None
    term_wait_seconds: float
    kill_wait_seconds: float
    identity_verified_before_term: bool
    identity_verified_before_kill: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "reason": self.reason,
            "term_sent": self.term_sent,
            "kill_sent": self.kill_sent,
            "reaped": self.reaped,
            "returncode": self.returncode,
            "term_wait_seconds": self.term_wait_seconds,
            "kill_wait_seconds": self.kill_wait_seconds,
            "identity_verified_before_term": self.identity_verified_before_term,
            "identity_verified_before_kill": self.identity_verified_before_kill,
        }


@dataclass(frozen=True)
class ExactChildOutcome:
    """Result of one exact owned-child launch.

    ``fallback_permitted`` is true only after the child has been reaped.  It is
    deliberately separate from whether the child completed successfully.
    """

    status: str
    command: tuple[str, ...]
    identity: ExactChildIdentity
    elapsed_seconds: float
    returncode: int | None
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    exact_child_peak_rss_bytes: int | None
    exact_child_peak_rss_source: str | None
    containment: ContainmentReport | None
    fallback_permitted: bool

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "command": list(self.command),
            "identity": self.identity.as_dict(),
            "elapsed_seconds": self.elapsed_seconds,
            "returncode": self.returncode,
            "stdout_captured_bytes": len(self.stdout),
            "stderr_captured_bytes": len(self.stderr),
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "exact_child_peak_rss_bytes": self.exact_child_peak_rss_bytes,
            "exact_child_peak_rss_source": self.exact_child_peak_rss_source,
            "containment": None
            if self.containment is None
            else self.containment.as_dict(),
            "fallback_permitted": self.fallback_permitted,
            "new_noninteractive_session": True,
            "new_exact_child_process_group": True,
        }


class _BoundedPipeDrainer:
    """Drain a pipe without allowing a child to consume unbounded parent RAM."""

    def __init__(self, stream: Any, *, cap_bytes: int) -> None:
        self._stream = stream
        self._cap_bytes = cap_bytes
        self._body = bytearray()
        self.truncated = False
        self._thread = threading.Thread(target=self._drain, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _drain(self) -> None:
        try:
            while True:
                block = self._stream.read(64 * 1024)
                if not block:
                    return
                remaining = self._cap_bytes - len(self._body)
                if remaining > 0:
                    self._body.extend(block[:remaining])
                if len(block) > remaining:
                    self.truncated = True
        except (OSError, ValueError):
            # Parent-side closing after a bounded containment result is normal.
            return

    def finish(self) -> bytes:
        self._thread.join(timeout=0.5)
        if self._thread.is_alive():
            try:
                self._stream.close()
            except (OSError, ValueError):
                pass
            self._thread.join(timeout=0.5)
        return bytes(self._body)


def _positive_seconds(value: float, *, label: str) -> float:
    parsed = float(value)
    if not parsed > 0.0 or parsed != parsed or parsed == float("inf"):
        raise ValueError(f"{label} must be a positive finite number")
    return parsed


def _linux_proc_start_identity(pid: int) -> str:
    """Read only the owned child's Linux start time, never a process listing."""

    stat_path = Path("/proc") / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExactChildIdentityError(
            "cannot read the exact owned child's Linux start identity"
        ) from exc
    try:
        # The comm field is parenthesized and may itself contain spaces, so
        # split only after its final closing parenthesis.  Field 22 is the
        # twentieth field after the process-state field.
        fields = raw.rsplit(")", 1)[1].strip().split()
        start_ticks = fields[19]
    except (IndexError, ValueError) as exc:
        raise ExactChildIdentityError(
            "owned child /proc stat does not contain a start identity"
        ) from exc
    if not start_ticks.isdigit():
        raise ExactChildIdentityError("owned child start identity is malformed")
    return start_ticks


def _linux_proc_peak_rss_bytes(pid: int) -> tuple[int, str]:
    """Read only the owned child's Linux high-water RSS from ``/proc``."""

    status_path = Path("/proc") / str(pid) / "status"
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExactChildIdentityError(
            "cannot read the exact owned child's Linux RSS evidence"
        ) from exc
    values: dict[str, int] = {}
    for line in lines:
        key, separator, remainder = line.partition(":")
        if not separator or key not in {"VmHWM", "VmRSS"}:
            continue
        pieces = remainder.split()
        if len(pieces) < 2 or pieces[1] != "kB" or not pieces[0].isdigit():
            raise ExactChildIdentityError("owned child Linux RSS evidence is malformed")
        values[key] = int(pieces[0]) * 1024
    if "VmHWM" in values:
        return values["VmHWM"], "linux_proc_VmHWM"
    if "VmRSS" in values:
        return values["VmRSS"], "linux_proc_VmRSS_fallback"
    raise ExactChildIdentityError("owned child Linux RSS evidence is absent")


class R235ExactChildWatchdog:
    """Launch and, if necessary, contain exactly one noninteractive child group.

    The default requires Linux's exact ``/proc/<pid>/stat`` start identity,
    which is the production Kaggle path.  Offline unit tests on macOS may opt
    into the weaker Popen/waitpid ownership token explicitly; such a result is
    never represented as a production execution receipt.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float,
        term_grace_seconds: float,
        kill_grace_seconds: float,
        max_capture_bytes: int = DEFAULT_MAX_CAPTURE_BYTES,
        allow_non_linux_test_identity: bool = False,
    ) -> None:
        self.timeout_seconds = _positive_seconds(timeout_seconds, label="timeout_seconds")
        self.term_grace_seconds = _positive_seconds(
            term_grace_seconds, label="term_grace_seconds"
        )
        self.kill_grace_seconds = _positive_seconds(
            kill_grace_seconds, label="kill_grace_seconds"
        )
        if not isinstance(max_capture_bytes, int) or max_capture_bytes <= 0:
            raise ValueError("max_capture_bytes must be a positive integer")
        self.max_capture_bytes = max_capture_bytes
        self.allow_non_linux_test_identity = bool(allow_non_linux_test_identity)

    def _record_identity(self, child: subprocess.Popen[bytes]) -> ExactChildIdentity:
        # Do not call ``poll`` before reading the identity.  A fast child is
        # still a zombie owned by this Popen object until *we* waitpid it, so
        # its PID/session and Linux start tick remain available for capture.
        # Polling first would turn harmless short probes into false failures.
        pid = int(child.pid)
        if pid <= 1:
            raise ExactChildIdentityError("owned child PID is not signal-safe")
        try:
            process_group_id = int(os.getpgid(pid))
            session_id = int(os.getsid(pid))
        except OSError as exc:
            raise ExactChildIdentityError(
                "cannot read the exact owned child session/process group"
            ) from exc
        if process_group_id != pid or session_id != pid:
            raise ExactChildIdentityError(
                "owned child did not enter its own session and process group"
            )
        if sys.platform.startswith("linux"):
            return ExactChildIdentity(
                pid=pid,
                process_group_id=process_group_id,
                session_id=session_id,
                start_identity_kind="linux_proc_start_ticks",
                start_identity=_linux_proc_start_identity(pid),
            )
        if not self.allow_non_linux_test_identity:
            raise ExactChildIdentityError(
                "production exact-child start identity requires Linux /proc; "
                "use dry/offline mode on this host"
            )
        return ExactChildIdentity(
            pid=pid,
            process_group_id=process_group_id,
            session_id=session_id,
            start_identity_kind="popen_waitpid_owner_test_only",
            start_identity="non_linux_offline_test",
        )

    def _cleanup_unverified_direct_child(self, child: subprocess.Popen[bytes]) -> None:
        """Boundedly reap only this Popen-owned direct child on setup failure.

        There is intentionally no group signal here: identity setup did not
        establish that the recorded group is still safe.  ``Popen`` still owns
        this direct PID, so this cannot inspect or signal an interactive or
        managed process.  The caller receives the identity failure regardless
        of whether this best-effort exact-child cleanup succeeds.
        """

        if child.poll() is not None:
            return
        try:
            child.terminate()
        except ProcessLookupError:
            return
        if self._wait(child, self.term_grace_seconds) is not None:
            return
        try:
            child.kill()
        except ProcessLookupError:
            return
        if self._wait(child, self.kill_grace_seconds) is None:
            raise ExactChildUnreapedError(
                "unverified direct child survived bounded setup-failure cleanup"
            )

    def _verify_identity(
        self, child: subprocess.Popen[bytes], identity: ExactChildIdentity
    ) -> bool:
        # poll() uses this Popen object's waitpid ownership.  If it returns a
        # status, the exact child is already reaped and no signal is needed.
        if child.poll() is not None:
            return False
        try:
            process_group_id = int(os.getpgid(identity.pid))
            session_id = int(os.getsid(identity.pid))
        except OSError as exc:
            raise ExactChildIdentityError(
                "exact owned child identity disappeared before containment"
            ) from exc
        if (
            process_group_id != identity.process_group_id
            or session_id != identity.session_id
            or process_group_id != identity.pid
            or session_id != identity.pid
        ):
            raise ExactChildIdentityError(
                "exact owned child session/process-group identity changed"
            )
        if identity.start_identity_kind == "linux_proc_start_ticks":
            if _linux_proc_start_identity(identity.pid) != identity.start_identity:
                raise ExactChildIdentityError("exact owned child start identity changed")
        elif identity.start_identity_kind != "popen_waitpid_owner_test_only":
            raise ExactChildIdentityError("exact owned child identity kind is unknown")
        return True

    @staticmethod
    def _sample_exact_child_peak_rss(
        identity: ExactChildIdentity,
    ) -> tuple[int, str] | None:
        """Sample the one owned child only; a post-reap absence is expected."""

        if identity.start_identity_kind != "linux_proc_start_ticks":
            return None
        try:
            return _linux_proc_peak_rss_bytes(identity.pid)
        except ExactChildIdentityError:
            # This may be a normal race with the Popen owner reaping a short
            # child.  Production preflight separately rejects missing sampled
            # evidence; the generic watchdog remains useful for short probes.
            return None

    @staticmethod
    def _wait(child: subprocess.Popen[bytes], seconds: float) -> int | None:
        try:
            return int(child.wait(timeout=seconds))
        except subprocess.TimeoutExpired:
            return None

    def _contain(
        self,
        child: subprocess.Popen[bytes],
        identity: ExactChildIdentity,
        *,
        reason: str,
    ) -> ContainmentReport:
        """Contain only the recorded exact child group, then boundedly reap it."""

        if child.poll() is not None:
            return ContainmentReport(
                reason=reason,
                term_sent=False,
                kill_sent=False,
                reaped=True,
                returncode=child.returncode,
                term_wait_seconds=0.0,
                kill_wait_seconds=0.0,
                identity_verified_before_term=False,
                identity_verified_before_kill=False,
            )

        verified_before_term = self._verify_identity(child, identity)
        if not verified_before_term:
            return ContainmentReport(
                reason=reason,
                term_sent=False,
                kill_sent=False,
                reaped=True,
                returncode=child.returncode,
                term_wait_seconds=0.0,
                kill_wait_seconds=0.0,
                identity_verified_before_term=False,
                identity_verified_before_kill=False,
            )
        try:
            os.killpg(identity.process_group_id, signal.SIGTERM)
        except ProcessLookupError:
            # A post-validation exit is harmless only if the Popen owner can
            # immediately reap the exact child.
            raced_returncode = self._wait(child, 0.05)
            if raced_returncode is not None:
                return ContainmentReport(
                    reason=reason,
                    term_sent=False,
                    kill_sent=False,
                    reaped=True,
                    returncode=raced_returncode,
                    term_wait_seconds=0.0,
                    kill_wait_seconds=0.0,
                    identity_verified_before_term=True,
                    identity_verified_before_kill=False,
                )
            raise ExactChildIdentityError(
                "exact child process group disappeared without an owned-child reap"
            )
        except PermissionError as exc:
            raise ExactChildIdentityError(
                "cannot signal the exact owned child process group"
            ) from exc

        term_started = time.monotonic()
        returncode = self._wait(child, self.term_grace_seconds)
        term_wait_seconds = time.monotonic() - term_started
        if returncode is not None:
            return ContainmentReport(
                reason=reason,
                term_sent=True,
                kill_sent=False,
                reaped=True,
                returncode=returncode,
                term_wait_seconds=term_wait_seconds,
                kill_wait_seconds=0.0,
                identity_verified_before_term=True,
                identity_verified_before_kill=False,
            )

        verified_before_kill = self._verify_identity(child, identity)
        if not verified_before_kill:
            return ContainmentReport(
                reason=reason,
                term_sent=True,
                kill_sent=False,
                reaped=True,
                returncode=child.returncode,
                term_wait_seconds=term_wait_seconds,
                kill_wait_seconds=0.0,
                identity_verified_before_term=True,
                identity_verified_before_kill=False,
            )
        try:
            os.killpg(identity.process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            raced_returncode = self._wait(child, 0.05)
            if raced_returncode is not None:
                return ContainmentReport(
                    reason=reason,
                    term_sent=True,
                    kill_sent=False,
                    reaped=True,
                    returncode=raced_returncode,
                    term_wait_seconds=term_wait_seconds,
                    kill_wait_seconds=0.0,
                    identity_verified_before_term=True,
                    identity_verified_before_kill=True,
                )
            raise ExactChildIdentityError(
                "exact child group disappeared before KILL without an owned-child reap"
            )
        except PermissionError as exc:
            raise ExactChildIdentityError(
                "cannot KILL the exact owned child process group"
            ) from exc

        kill_started = time.monotonic()
        returncode = self._wait(child, self.kill_grace_seconds)
        kill_wait_seconds = time.monotonic() - kill_started
        if returncode is None:
            report = ContainmentReport(
                reason=reason,
                term_sent=True,
                kill_sent=True,
                reaped=False,
                returncode=None,
                term_wait_seconds=term_wait_seconds,
                kill_wait_seconds=kill_wait_seconds,
                identity_verified_before_term=True,
                identity_verified_before_kill=True,
            )
            raise ExactChildUnreapedError(
                "exact child survived bounded TERM and KILL containment: "
                f"{report.as_dict()}"
            )
        return ContainmentReport(
            reason=reason,
            term_sent=True,
            kill_sent=True,
            reaped=True,
            returncode=returncode,
            term_wait_seconds=term_wait_seconds,
            kill_wait_seconds=kill_wait_seconds,
            identity_verified_before_term=True,
            identity_verified_before_kill=True,
        )

    def run(
        self,
        command: Sequence[str | os.PathLike[str]],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> ExactChildOutcome:
        """Run a new exact child session/process group under one hard deadline.

        ``shell`` is never used.  ``stdin`` is always ``DEVNULL`` and the
        child cannot inherit an interactive terminal from this watchdog.
        """

        argv = tuple(os.fspath(part) for part in command)
        if not argv or not all(isinstance(part, str) and part for part in argv):
            raise ValueError("command must be a non-empty sequence of non-empty strings")
        child_cwd: str | None = None
        if cwd is not None:
            resolved_cwd = Path(cwd).expanduser().resolve()
            if not resolved_cwd.is_dir() or resolved_cwd.is_symlink():
                raise ValueError("child cwd must be a physical directory")
            child_cwd = str(resolved_cwd)

        started = time.monotonic()
        child = subprocess.Popen(
            list(argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=child_cwd,
            env=env,
            close_fds=True,
            start_new_session=True,
            text=False,
            bufsize=0,
        )
        if child.stdout is None or child.stderr is None:  # pragma: no cover - Popen invariant
            raise ExactChildWatchdogError("owned child has no capture pipes")
        try:
            identity = self._record_identity(child)
        except Exception:
            try:
                self._cleanup_unverified_direct_child(child)
            finally:
                # Closing captures avoids retaining a parent-side FD if the
                # child already exited or setup could not establish ownership.
                child.stdout.close()
                child.stderr.close()
            raise
        stdout = _BoundedPipeDrainer(child.stdout, cap_bytes=self.max_capture_bytes)
        stderr = _BoundedPipeDrainer(child.stderr, cap_bytes=self.max_capture_bytes)
        stdout.start()
        stderr.start()
        child_peak_rss_bytes: int | None = None
        child_peak_rss_source: str | None = None

        def sample_peak_rss() -> None:
            nonlocal child_peak_rss_bytes, child_peak_rss_source
            sample = self._sample_exact_child_peak_rss(identity)
            if sample is None:
                return
            sampled_bytes, sampled_source = sample
            if child_peak_rss_bytes is None or sampled_bytes > child_peak_rss_bytes:
                child_peak_rss_bytes = sampled_bytes
                child_peak_rss_source = sampled_source

        sample_peak_rss()
        containment: ContainmentReport | None = None
        status = "completed"
        try:
            deadline = started + self.timeout_seconds
            while child.poll() is None:
                sample_peak_rss()
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    sample_peak_rss()
                    containment = self._contain(child, identity, reason="deadline_exceeded")
                    status = "deadline_contained"
                    break
                time.sleep(min(0.01, remaining))
            if status == "completed":
                if child.returncode != 0:
                    status = "child_nonzero_reaped"
                elif stdout.truncated or stderr.truncated:
                    status = "child_output_limit_reaped"
        except BaseException:
            # A caller interruption or an unexpected parent-side exception
            # must not leave the fresh detached child behind.  This invokes
            # only the already recorded exact child group.
            if child.poll() is None:
                self._contain(child, identity, reason="watchdog_parent_interrupted")
            raise
        finally:
            captured_stdout = stdout.finish()
            captured_stderr = stderr.finish()
        elapsed = time.monotonic() - started
        reaped = child.poll() is not None
        if not reaped:
            # This path is reachable only if an identity/containment exception
            # was raised.  Never mark a fallback safe without a reap.
            raise ExactChildUnreapedError("exact child was not reaped before watchdog return")
        return ExactChildOutcome(
            status=status,
            command=argv,
            identity=identity,
            elapsed_seconds=elapsed,
            returncode=child.returncode,
            stdout=captured_stdout,
            stderr=captured_stderr,
            stdout_truncated=stdout.truncated,
            stderr_truncated=stderr.truncated,
            exact_child_peak_rss_bytes=child_peak_rss_bytes,
            exact_child_peak_rss_source=child_peak_rss_source,
            containment=containment,
            fallback_permitted=status != "completed" and reaped,
        )

    def run_with_precomputed_fallback(
        self,
        command: Sequence[str | os.PathLike[str]],
        *,
        precomputed_fallback: T,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[ExactChildOutcome, T | None]:
        """Return an already prepared fallback only after exact child reaping.

        The caller owns legality/fingerprint validation.  This function merely
        prevents a fallback from becoming available while a child may still
        own native work.
        """

        outcome = self.run(command, cwd=cwd, env=env)
        if outcome.completed:
            return outcome, None
        if not outcome.fallback_permitted:
            raise ExactChildUnreapedError(
                "precomputed fallback is forbidden until the exact child is reaped"
            )
        return outcome, precomputed_fallback
