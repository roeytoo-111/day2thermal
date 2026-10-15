#!/usr/bin/env bash
# VM: after v5 is scored, train v5a seed 1 (same data and recipe as v5a_s0, seed=1) to measure the RGB seed spread,
# then rescore the ladder with all runs.
cd ~/day2thermal/detection && source ../.venv/bin/activate
until [ -f ~/v5.done ] || [ -f ~/v5.failed ]; do sleep 60; done
python3 scripts/train_rgb_like_fw4.py --fw4 ~/models/yolo26s_fw4_20261011.pt --base ~/models/yolo26s_small_07072026.pt \
    --data ~/day2thermal/data/rgb_v5a/data.yaml --name v5a_s1 --seed 1 > ~/train_v5a_s1.log 2>&1
W=$(find ~/day2thermal -path "*/rgb/v5a_s1/weights/best.pt" | head -1)
[ -s "$W" ] || { touch ~/v5a_s1.failed; exit 1; }
bash scripts/arsuf_dets.sh day v5a_s1 $W
bash scripts/score_rgb_ladder.sh v5a_s0 v5_s0 v5a_s1
touch ~/v5a_s1.done
