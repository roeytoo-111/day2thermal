#!/usr/bin/env bash
# VM queue after v5 (notebook 2026-10-15 section 8). All on rgb_v5a's real valid/test; recipe = fw4's (train_rgb_like_fw4.py).
#   v5s_s0     with_sim: rgb_v5a minus the sim tail          (baseline for the next two)
#   v5s_p2_s0  same data, yolo26s-p2.yaml (extra stride-4 head)       -> P2 effect
#   v5n_s0     no_sim: rgb_v5a minus ALL simulator renders            -> simulator effect
#   v5s_s1     v5s seed 1                                              -> seed spread
# Each run is scored (background, cumulative) while the next one trains: scripts/score_rgb_ladder.sh
cd ~/day2thermal/detection && source ../.venv/bin/activate
until [ -f ~/v5.done ] || [ -f ~/v5.failed ]; do sleep 60; done
D=~/day2thermal/data; B=~/models/yolo26s_small_07072026.pt; F=~/models/yolo26s_fw4_20261011.pt
RUNS="v5a_s0 v5_s0"
run() {  # name data [extra args]
  n=$1; d=$2; shift 2
  python3 scripts/train_rgb_like_fw4.py --fw4 $F --base $B --data $d --name $n "$@" > ~/train_$n.log 2>&1
  W=$(find ~/day2thermal -path "*/rgb/$n/weights/best.pt" | head -1)
  [ -s "$W" ] || { touch ~/$n.failed; return; }
  RUNS="$RUNS $n"
  setsid nohup bash scripts/score_rgb_ladder.sh $RUNS > /dev/null 2>&1 < /dev/null &
  touch ~/$n.done
}
run v5s_s0    $D/rgb_sim/with_sim/data.yaml
run v5s_p2_s0 $D/rgb_sim/with_sim/data.yaml --arch yolo26s-p2.yaml
run v5n_s0    $D/rgb_sim/no_sim/data.yaml
run v5s_s1    $D/rgb_sim/with_sim/data.yaml --seed 1
sleep 1800   # let the last scoring finish, then cost table
python3 src/eval/model_cost.py fw4=$F $(for r in $RUNS; do echo $r=$(find ~/day2thermal -path "*/rgb/$r/weights/best.pt" | head -1); done) > results/model_cost.md 2>&1
touch ~/ablate.done
