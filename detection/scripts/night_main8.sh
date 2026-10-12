#!/usr/bin/env bash
# Queue v8 (2026-10-12): thermal over-firing. Criteria: notebook/2026-10-12.md section 2.
#   mine hard examples (M0 s0 + M0 s1) -> N1 / N2 / N3 datasets -> seed 0 of each -> seed 1 of each
#   every run: test videos (train_and_infer), Boson held-out, Arsuf bench thermal frames -> shutdown.
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
[ -f data/boson_work/hardneg/summary.json ] || python3 src/data/mine_thermal_hard.py --labels-dir data/boson_work/labels \
    --videos ../data/boson_work_thermal --train data/thermal-5-bosonP-u0 \
    --models ../runs/thermal_uav/bosonM/weights/best.pt ../runs/thermal_uav/bosonM0u_s1/weights/best.pt \
    --out data/boson_work/hardneg > ~/mine.log 2>&1
[ -d data/boson_work/clean_neg ] || python3 src/data/make_clean_negatives.py --work data/bgpaste data/boson_work/bg_boson \
    --out data/boson_work/clean_neg > ~/clean_neg.log 2>&1
[ -d data/thermal-7-N1-u0 ] || python3 src/data/build_hardneg_sets.py --base data/thermal-5-bosonP-u0 \
    --hard data/boson_work/hardneg --neg data/boson_work/clean_neg --out-prefix data/thermal-7 > ~/build_n.log 2>&1
touch ~/n_data.done
for s in 0 1; do
  for n in N1 N2 N3; do
    bash scripts/train_and_infer.sh boson${n}_s$s data/thermal-7-$n-u0 yolo26s-p2.yaml pretrained=$Y26 seed=$s
    bash scripts/infer_heldout.sh boson${n}_s$s
    bash scripts/arsuf_dets.sh thermal boson${n}_s$s
    touch ~/${n}_s$s.done
  done
done
sleep 60; sudo shutdown -h now
