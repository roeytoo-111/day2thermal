#!/usr/bin/env bash
# VM queue: when an RGB training run ends, score it against fw4 at once (no idle GPU):
#   Arsuf day detections (tiled, every 6th frame, as fw4's) + score_rgb_models on v5 test / unseen / Boson held-out.
# usage: bash scripts/after_rgb_train.sh RUN_NAME        (run dir ../runs/rgb/RUN_NAME)
cd ~/day2thermal/detection && source ../.venv/bin/activate
RUN=$1; W=../runs/rgb/$RUN/weights/best.pt
while pgrep -f "[t]rain_rgb_like_fw4.py.*--name $RUN" > /dev/null; do sleep 60; done
[ -s $W ] || { echo "no $W"; touch ~/after_$RUN.failed; exit 1; }
for y in ../data/rgb_eval/*/data.yaml; do sed -i "s#^path: .*#path: $(cd $(dirname $y) && pwd)#" $y; done
sed -i "s#^path: .*#path: /home/daniel/day2thermal/data/rgb_v5a#" ../data/rgb_eval/v5_test.yaml
bash scripts/arsuf_dets.sh day $RUN $W
python3 src/eval/score_rgb_models.py --model fw4 ~/models/yolo26s_fw4_20261011.pt --model $RUN $W \
    --set v5_test ../data/rgb_eval/v5_test.yaml --set unseen ../data/rgb_eval/unseen_test/data.yaml \
    --set boson ../data/rgb_eval/boson_heldout/data.yaml --out results/scores_rgb_$RUN.md > ~/score_$RUN.log 2>&1
touch ~/after_$RUN.done
