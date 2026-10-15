#!/usr/bin/env bash
# VM: score fw4 + every named run on all six comparison sets -> results/scores_rgb_ladder_<last run>.md
# usage: bash scripts/score_rgb_ladder.sh v5a_s0 [v5_s0 ...]   (Arsuf day detections must exist: scripts/arsuf_dets.sh day RUN WEIGHTS)
cd ~/day2thermal/detection && source ../.venv/bin/activate
for y in ../data/rgb_eval/*/data.yaml ../data/rgb_v5/extra_tests/*/data.yaml; do sed -i "s#^path: .*#path: $(cd $(dirname $y) && pwd)#" $y; done
sed -i "s#^path: .*#path: /home/daniel/day2thermal/data/rgb_v5a#" ../data/rgb_eval/v5_test.yaml
M=""; for r in "$@"; do M="$M --model $r $(find ~/day2thermal -path "*/rgb/$r/weights/best.pt" | head -1)"; last=$r; done
python3 src/eval/score_rgb_models.py --model fw4 ~/models/yolo26s_fw4_20261011.pt $M \
    --set v5_test ../data/rgb_eval/v5_test.yaml --set unseen ../data/rgb_eval/unseen_test/data.yaml \
    --set boson ../data/rgb_eval/boson_heldout/data.yaml --set flight1014 ../data/rgb_v5/extra_tests/flight1014/data.yaml \
    --set oi_sky ../data/rgb_v5/extra_tests/oi_sky/data.yaml --out results/scores_rgb_ladder_$last.md > ~/score_ladder_$last.log 2>&1
echo done > ~/score_ladder_$last.done
