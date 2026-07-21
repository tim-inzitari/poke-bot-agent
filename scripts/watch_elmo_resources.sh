#!/usr/bin/env bash
set -u

host=${1:-elmo}

while :; do
  printf '\033[H\033[2J'
  if ! ssh -o BatchMode=yes -o ConnectTimeout=4 "$host" '
    echo "ELMO RESOURCE VIEW"
    date
    uptime
    echo
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader
    echo
    mpstat 1 1 | tail -n 4
    echo
    free -h | sed -n "1,3p"
    pid=$(pgrep -f "^python /workspace/scripts/run_remote_worker.py " | head -n 1 || true)
    if test -n "$pid"; then
      cgroup=$(sed -n "s/^0:://p" "/proc/$pid/cgroup")
      echo
      echo "remote-controller pid=$pid cgroup=$cgroup"
      for metric in memory.current memory.max memory.swap.current memory.swap.max pids.current pids.max cpu.max; do
        value=$(cat "/sys/fs/cgroup${cgroup}/$metric" 2>/dev/null || echo unavailable)
        echo "$metric=$value"
      done
      ps --ppid "$pid" -o pcpu=,rss=,comm= | awk '\''
        {cpu += $1; rss += $2; processes += 1}
        END {printf "direct_children=%d cpu_sum=%.1f%% rss=%.2fGiB\n", processes, cpu, rss/1048576}
      '\''
    else
      echo "remote controller is not ready"
    fi
  '; then
    echo
    echo "Elmo SSH unavailable; retrying"
    sleep 2
  fi
done
