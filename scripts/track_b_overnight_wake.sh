#!/bin/bash
cd /home/inzi/poke-bot-agent
while true; do
  sleep 180
  echo 'AGENT_LOOP_TICK_track_b_overnight {"prompt":"Track B overnight: Core=Elmo+Bert (192.168.1.143:8765 + bert.local:8766), BW=local-only. Read core_kernel.log blackwell.log SOLE_TRAIN_OWNER.lock TRACK_A_TOPOLOGY.txt. Soft-drop Bert only on bert slot-fail storm. Never attach Bert to BW. Obey RESTART_POLICY.md. Update SOLE_TRAIN_OWNER.lock. Continue indefinite watch."}'
done
