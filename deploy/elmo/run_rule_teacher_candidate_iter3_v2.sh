#!/bin/sh
set -eu

container="poke-bot-truenas-worker"
run_name="alakazam_iono_teacher_iter3_v2"
artifact_dir="/workspace/checkpoint/rule-teacher-candidates"
host_artifact_dir="/mnt/Main/Elmo/poke-bot-agent/containers/truenas-worker/checkpoint/rule-teacher-candidates"

/usr/bin/docker exec \
  -e OMP_NUM_THREADS=4 \
  -e MKL_NUM_THREADS=4 \
  "$container" \
  sh -lc "cd /workspace && exec nice -n 15 python -u scripts/train_rule_teacher_candidate.py \
    --report /workspace/runtime-logs/rule-teacher-iono-v1-elmo/PROTECTED_RULE_TEACHER_CORPUS.json \
    --corpus /workspace/runtime-logs/rule-teacher-iono-v1-elmo/teacher_wins.jsonl \
    --init-checkpoint $artifact_dir/iter_00003.pt \
    --run-name $run_name \
    --device cuda:0 \
    --epochs 5 \
    --lr 5e-5 \
    --games-per-batch 32 \
    --max-decisions-per-batch 2048 \
    --patience 2 \
    --value-loss-weight 0.05"

mkdir -p "$host_artifact_dir"
/usr/bin/docker cp \
  "$container:/workspace/outputs/checkpoints/$run_name.best.pt" \
  "$host_artifact_dir/$run_name.best.pt"
/usr/bin/docker cp \
  "$container:/workspace/outputs/train/$run_name.teacher_candidate.json" \
  "$host_artifact_dir/$run_name.teacher_candidate.json"
chmod 0644 \
  "$host_artifact_dir/$run_name.best.pt" \
  "$host_artifact_dir/$run_name.teacher_candidate.json"
