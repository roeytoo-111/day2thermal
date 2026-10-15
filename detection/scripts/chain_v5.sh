#!/usr/bin/env bash
# VM queue, RGB ladder fw4 -> v5a -> v5 (notebook 2026-10-15): train v5 (v5a's split frozen + flight tiles + Open Images
# tiles, fw4 recipe), then score fw4 / v5a / v5 on every set + Arsuf day detections. v5a is scored separately
# (scripts/score_rgb_ladder.sh v5a_s0) so the GPU never waits for a scoring run.
cd ~/day2thermal/detection && source ../.venv/bin/activate
for y in ../data/rgb_v5/extra_tests/*/data.yaml; do sed -i "s#^path: .*#path: $(cd $(dirname $y) && pwd)#" $y; done
sed -i "s#^path: .*#path: /home/daniel/day2thermal/data/rgb_v5#" ../data/rgb_v5/data.yaml
python3 scripts/train_rgb_like_fw4.py --fw4 ~/models/yolo26s_fw4_20261011.pt --base ~/models/yolo26s_small_07072026.pt \
    --data ~/day2thermal/data/rgb_v5/data.yaml --name v5_s0 > ~/train_v5.log 2>&1
V5=$(find ~/day2thermal -path "*/rgb/v5_s0/weights/best.pt" | head -1)
[ -s "$V5" ] || { touch ~/v5.failed; exit 1; }
bash scripts/arsuf_dets.sh day v5_s0 $V5
bash scripts/score_rgb_ladder.sh v5_s0
touch ~/v5.done
