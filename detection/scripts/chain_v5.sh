#!/usr/bin/env bash
# VM queue, RGB ladder fw4 -> v5a -> v5 (notebook 2026-10-15). Runs after scripts/after_rgb_train.sh v5a_s0 has finished:
#   1. score fw4 + v5a on the two new test sets (flight1014, oi_sky)
#   2. train v5 (v5a's split frozen + flight tiles + Open Images tiles), fw4 recipe
#   3. score v5 on all sets (v5 test, unseen, boson, flight1014, oi_sky) + Arsuf day detections
cd ~/day2thermal/detection && source ../.venv/bin/activate
until [ -f ~/after_v5a_s0.done ] || [ -f ~/after_v5a_s0.failed ]; do sleep 60; done
for y in ../data/rgb_v5/extra_tests/*/data.yaml; do sed -i "s#^path: .*#path: $(cd $(dirname $y) && pwd)#" $y; done
sed -i "s#^path: .*#path: /home/daniel/day2thermal/data/rgb_v5#" ../data/rgb_v5/data.yaml
EXTRA="--set flight1014 ../data/rgb_v5/extra_tests/flight1014/data.yaml --set oi_sky ../data/rgb_v5/extra_tests/oi_sky/data.yaml"
best() { find ~/day2thermal -path "*/rgb/$1/weights/best.pt" | head -1; }
V5A=$(best v5a_s0)
python3 src/eval/score_rgb_models.py --model fw4 ~/models/yolo26s_fw4_20261011.pt --model v5a_s0 $V5A $EXTRA \
    --out results/scores_rgb_v5a_s0_extra.md > ~/score_v5a_extra.log 2>&1
touch ~/v5a_extra.done
python3 scripts/train_rgb_like_fw4.py --fw4 ~/models/yolo26s_fw4_20261011.pt --base ~/models/yolo26s_small_07072026.pt \
    --data ~/day2thermal/data/rgb_v5/data.yaml --name v5_s0 > ~/train_v5.log 2>&1
V5=$(best v5_s0)
[ -s "$V5" ] || { touch ~/v5.failed; exit 1; }
bash scripts/arsuf_dets.sh day v5_s0 $V5
python3 src/eval/score_rgb_models.py --model fw4 ~/models/yolo26s_fw4_20261011.pt --model v5a_s0 $V5A --model v5_s0 $V5 \
    --set v5_test ../data/rgb_eval/v5_test.yaml --set unseen ../data/rgb_eval/unseen_test/data.yaml \
    --set boson ../data/rgb_eval/boson_heldout/data.yaml $EXTRA --out results/scores_rgb_ladder.md > ~/score_ladder.log 2>&1
touch ~/v5.done
