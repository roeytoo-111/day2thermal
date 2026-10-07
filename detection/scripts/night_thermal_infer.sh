#!/usr/bin/env bash
# GPU job 2 (VM): current best thermal model (YOLO26s-P2, seed 1) over every Boson thermal video, conf 0.05 JSON.
until [ -f ~/rgb_label.done ]; do sleep 20; done
cd ~/day2thermal/detection && source ../.venv/bin/activate
W=~/day2thermal/runs/thermal_uav/pasteG_y26sp2_seed1/weights/best.pt
OUT=~/day2thermal/data/boson_work/thermdets; mkdir -p $OUT
for f in ~/day2thermal/data/boson_work_thermal/*_thermal.mp4; do
  s=$(basename $f _thermal.mp4)
  [ -s $OUT/$s.json ] || python3 src/deploy/run_yolo_inference.py --model $W --video $f --out_json $OUT/$s.json --imgsz 512 --progress_every 100000 2>&1 | tail -1
done
touch ~/thermal_infer.done
