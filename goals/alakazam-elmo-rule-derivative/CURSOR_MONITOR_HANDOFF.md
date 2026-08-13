# Cursor read-only monitoring handoff

Recorded: `2026-08-13T00:48:43Z`

This handoff permits observation only. Cursor must not edit files, signal or
renice processes, delete or move spools, restart jobs, transfer artifacts,
control services, or alter r274/production. Do not run commands that create
bytecode, logs, receipts, temporary files, or output directories.

Before monitoring, read these files completely:

- `/Users/tsinzitari/Documents/poke-agent-codex/GOAL.md`
- `/Users/tsinzitari/Documents/poke-agent-codex/goals/alakazam-elmo-rule-derivative/GOAL.md`
- `/Users/tsinzitari/Documents/poke-agent-codex/goals/alakazam-elmo-rule-derivative/contract.json`
- `/Users/tsinzitari/Documents/poke-agent-codex/goals/alakazam-elmo-rule-derivative/STATUS.json`

## Live workloads

### Elmo second half

- Host alias: `truenas`
- Parent PID: `1832369`
- Expected children: `15`
- Partition: odd manifest indices, 15 UTC days
- RAM spool: `/dev/shm/alakazam-refeature-1832369-1`
- Final output root: `/mnt/Main/main/poke-bot-agent/outputs/experiments/alakazam-elmo-rule-derivative-g1/fast-refeature-elmo-second-half-r309`
- Completion boundary: `COMPLETE.json` directly under that output root
- Observed spool at handoff: `23,208,740,808` bytes

### Inzi first-half retry

- Host alias: `inzi`
- Python PID: `223632` (wrapper/supervisor PID `223629`)
- Expected children: `32`
- Partition: even manifest indices with inner-day episode sharding, 15 UTC days
- RAM spool: `/dev/shm/alakazam-refeature-223632-0`
- Final output root: `/home/inzi/alakazam-r309-work/first-half-retry-r309`
- Completion boundary: `COMPLETE.json` directly under that output root
- Observed spool at handoff: `5,919,772,476` bytes
- This retry contains public-rule adapter source SHA-256
  `7b223a40e9ba86e7cd21d7cc0edbe1d5ed58e2ae47b8412f98b9cf5e1622f688`.

The retired Elmo first-half duplicate PID `1973900` is intentionally stopped
and its tmpfs spool was reclaimed. Do not restart it. The earlier Inzi
first-half PID `156013` failed on stale source and is not eligible evidence.

## Safe polling

Poll no more often than once every ten minutes unless a parent exits. These
commands are read-only:

```sh
ssh truenas '
  if [ -f /mnt/Main/main/poke-bot-agent/outputs/experiments/alakazam-elmo-rule-derivative-g1/fast-refeature-elmo-second-half-r309/COMPLETE.json ]; then
    echo COMPLETE
  elif [ -d /proc/1832369 ]; then
    echo RUNNING
  else
    echo EXITED_WITHOUT_COMPLETION
  fi
  ps -o pid=,ppid=,stat=,etime=,%cpu=,rss= -p 1832369
  ps --ppid 1832369 -o pid= | wc -l
  du -sb /dev/shm/alakazam-refeature-1832369-1 2>/dev/null || true
  df -B1 /dev/shm | tail -1
'
```

```sh
ssh inzi '
  if sudo test -f /home/inzi/alakazam-r309-work/first-half-retry-r309/COMPLETE.json; then
    echo COMPLETE
  elif [ -d /proc/223632 ]; then
    echo RUNNING
  else
    echo EXITED_WITHOUT_COMPLETION
  fi
  ps -o pid=,ppid=,stat=,etime=,%cpu=,rss= -p 223632
  pgrep -P 223632 | wc -l
  sudo du -sb /dev/shm/alakazam-refeature-223632-0 2>/dev/null || true
  df -B1 /dev/shm | tail -1
'
```

Dashboard presentation can be checked read-only at
`http://127.0.0.1:8780/api/status`, field
`alakazam_derivative_progress`. Dashboard labels are not receipt authority.

## What to report

Report only a state transition:

- a parent exited without `COMPLETE.json`;
- `COMPLETE.json` appeared;
- `/dev/shm` free space fell below 8 GiB;
- expected child count changed persistently for two polls; or
- SSH/read-only telemetry became unavailable.

On completion, report the exact `COMPLETE.json` path and its physical SHA-256,
but do not modify, copy, transfer, validate by rewriting, or consume it. The
main Codex task owns receipt validation, final shard transfer, combined corpus
sealing, STATUS updates, and every subsequent action.

Never touch either r274 managed service, any Kaggle service, or any interactive
SSH/Codex/terminal/editor session.
