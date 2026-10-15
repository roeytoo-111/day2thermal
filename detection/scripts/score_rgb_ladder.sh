#!/usr/bin/env bash
# VM: score fw4 + every named run on every comparison set:
#   Ultralytics sets (v5 test, unseen, Boson held-out, flight1014, oi_sky, sim_tail if present) -> results/scores_rgb_ladder_<last>.md
#   labelled flight GT, day camera, tiled 4K (Daniel's labels): Arsuf 10-05 (2,073 drone / 667 empty frames) and 08_58_15 cut
#   (945 frames) -> results/scores_dayGT_<last>.md
# usage: bash scripts/score_rgb_ladder.sh v5a_s0 [v5_s0 ...]
cd ~/day2thermal/detection && source ../.venv/bin/activate
for y in ../data/rgb_eval/*/data.yaml ../data/rgb_v5/extra_tests/*/data.yaml ../data/rgb_sim/sim_tail/data.yaml; do
  [ -f $y ] && sed -i "s#^path: .*#path: $(cd $(dirname $y) && pwd)#" $y; done
sed -i "s#^path: .*#path: /home/daniel/day2thermal/data/rgb_v5a#" ../data/rgb_eval/v5_test.yaml
FW4=~/models/yolo26s_fw4_20261011.pt
M=""; last=""
SETS="--set v5_test ../data/rgb_eval/v5_test.yaml --set unseen ../data/rgb_eval/unseen_test/data.yaml --set boson ../data/rgb_eval/boson_heldout/data.yaml"
[ -f ../data/rgb_v5/extra_tests/flight1014/data.yaml ] && SETS="$SETS --set flight1014 ../data/rgb_v5/extra_tests/flight1014/data.yaml --set oi_sky ../data/rgb_v5/extra_tests/oi_sky/data.yaml"
[ -f ../data/rgb_sim/sim_tail/data.yaml ] && SETS="$SETS --set sim_tail ../data/rgb_sim/sim_tail/data.yaml"
declare -A W; W[fw4]=$FW4
for r in "$@"; do W[$r]=$(find ~/day2thermal -path "*/rgb/$r/weights/best.pt" | head -1); M="$M --model $r ${W[$r]}"; last=$r; done
python3 src/eval/score_rgb_models.py --model fw4 $FW4 $M $SETS --out results/scores_rgb_ladder_$last.md > ~/score_ladder_$last.log 2>&1
AM=""; CM=""
for r in fw4 "$@"; do
  bash scripts/arsuf_dets.sh day $r ${W[$r]}; bash scripts/cut0858_dets.sh $r ${W[$r]}
  AM="$AM --model $r results/arsuf/day_$r.json"; CM="$CM --model $r results/cut0858/day_$r.json"
done
{ echo "# Day-camera flight GT"; echo; echo "## Arsuf 10-05"; python3 src/eval/score_arsuf_gt.py --gt ../data/arsuf_1005/pipeline_gt/arsuf_1005 --camera day $AM
  echo; echo "## 08_58_15 cut"; python3 src/eval/score_arsuf_gt.py --gt ../data/cut0858/pipeline_gt/boson_0858 --camera day $CM; } > results/scores_dayGT_$last.md 2>&1
echo done > ~/score_ladder_$last.done
