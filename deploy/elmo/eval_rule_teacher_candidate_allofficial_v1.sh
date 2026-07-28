#!/bin/sh
set -eu

container="poke-bot-truenas-worker"
name="alakazam_iono_teacher_iter1_v1.official1000"
container_out="/workspace/outputs/eval/$name.json"
host_out_dir="/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/checkpoint/rule-teacher-candidates"

set +e
/usr/bin/docker exec \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  "$container" \
  sh -lc "cd /workspace && exec nice -n 10 python -u scripts/eval_vs_baselines.py \
    --checkpoint /workspace/checkpoint/rule-teacher-candidates/alakazam_iono_teacher_iter1_v1.best.pt \
    --our-archetype alakazam \
    --games-per-opp 250 \
    --min-games-per-opp 250 \
    --workers 24 \
    --agent-mode policy \
    --gate 0.50 \
    --only iono dragapult-ex mega-abomasnow-ex mega-lucario-ex \
    --seed 962000 \
    --leaf-eval gpu-server \
    --leaf-gpu cuda:0 \
    --leaf-max-batch 4096 \
    --leaf-coalesce-ms 0.2 \
    --out $container_out"
eval_rc=$?
set -e

mkdir -p "$host_out_dir"
/usr/bin/docker cp \
  "$container:$container_out" \
  "$host_out_dir/$name.json"
chmod 0644 "$host_out_dir/$name.json"

# Exit 1 is a complete valid evaluation that missed the gate. Larger values
# indicate infrastructure failure and remain visible to systemd.
case "$eval_rc" in
  0|1) exit 0 ;;
  *) exit "$eval_rc" ;;
esac
