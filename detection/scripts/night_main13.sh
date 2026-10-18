#!/usr/bin/env bash
# Queue v13 = v12 reorganised so the GPU never waits on the CPU-bound inference stage (run_yolo_inference on the test
# videos is decode-limited: ~15 % GPU for ~30 min per run). Seed 1 trains NOW, alongside seed 0 (started by v12);
# each seed's inference runs as soon as its own training ends, overlapping the other's training.
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
post() { bash scripts/infer_heldout.sh $1; bash scripts/arsuf_dets.sh thermal $1; touch ~/${1#boson}.done; }
( bash scripts/train_and_infer.sh bosonN2D_s1 data/thermal-9-N2D-u0 yolo26s-p2.yaml pretrained=$Y26 seed=1; post bosonN2D_s1 ) &
S1=$!
# seed 0: its yolo training was started by v12; wait until it exits, then run its inference (train_and_infer skips training)
while pgrep -f "[y]olo detect train.*name=bosonN2D_s0" > /dev/null; do sleep 60; done
bash scripts/train_and_infer.sh bosonN2D_s0 data/thermal-9-N2D-u0 yolo26s-p2.yaml pretrained=$Y26 seed=0
post bosonN2D_s0
wait $S1
touch ~/N2D.done
