#!/usr/bin/env bash
# Safely pull cold pages back from swap when the host has ample RAM headroom.
#
# This deliberately operates on all configured swap devices: Linux does not
# expose a safe way to page in only one cgroup.  The headroom gate prevents a
# swapoff from competing with a memory-heavy training phase.

set -euo pipefail

reserve_gib="${POKEBOT_SWAPIN_RESERVE_GIB:-24}"
min_swap_mib="${POKEBOT_SWAPIN_MIN_MIB:-128}"
lock_file="${POKEBOT_SWAPIN_LOCK_FILE:-/run/pokebot-safe-swap-repatriate.lock}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "swap-repatriate: must run as root" >&2
  exit 2
fi

exec 9>"${lock_file}"
if ! flock -n 9; then
  echo "swap-repatriate: another pass is active; skipping"
  exit 0
fi

read_meminfo_kib() {
  awk -v key="$1" '$1 == key ":" { print $2; found=1; exit } END { if (!found) print 0 }' /proc/meminfo
}

mem_available_kib="$(read_meminfo_kib MemAvailable)"
swap_total_kib="$(read_meminfo_kib SwapTotal)"
swap_free_kib="$(read_meminfo_kib SwapFree)"
swap_used_kib="$((swap_total_kib - swap_free_kib))"
reserve_kib="$((reserve_gib * 1024 * 1024))"
min_swap_kib="$((min_swap_mib * 1024))"
required_kib="$((swap_used_kib + reserve_kib))"

if (( swap_total_kib == 0 || swap_used_kib < min_swap_kib )); then
  echo "swap-repatriate: nothing material to move (used_kib=${swap_used_kib})"
  exit 0
fi

if (( mem_available_kib < required_kib )); then
  echo "swap-repatriate: deferred (available_kib=${mem_available_kib} used_swap_kib=${swap_used_kib} reserve_kib=${reserve_kib})"
  exit 0
fi

# Recheck immediately before the state change so a concurrent allocator cannot
# invalidate the first observation while this script was being scheduled.
mem_available_kib="$(read_meminfo_kib MemAvailable)"
swap_free_kib="$(read_meminfo_kib SwapFree)"
swap_used_kib="$((swap_total_kib - swap_free_kib))"
required_kib="$((swap_used_kib + reserve_kib))"
if (( mem_available_kib < required_kib )); then
  echo "swap-repatriate: deferred after recheck (available_kib=${mem_available_kib} required_kib=${required_kib})"
  exit 0
fi

echo "swap-repatriate: moving ${swap_used_kib} KiB to RAM (available_kib=${mem_available_kib}, reserve_kib=${reserve_kib})"

# Always attempt to restore configured swap if swapoff is interrupted or fails.
restore_swap=1
trap 'if (( restore_swap )); then swapon -a || true; fi' EXIT
swapoff -a
swapon -a
restore_swap=0
trap - EXIT

remaining_kib="$(( $(read_meminfo_kib SwapTotal) - $(read_meminfo_kib SwapFree) ))"
echo "swap-repatriate: complete (remaining_kib=${remaining_kib})"
