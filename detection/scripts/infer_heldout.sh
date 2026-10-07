#!/usr/bin/env bash
# VM: run models on the two held-out Boson thermal videos -> results/detections/<model>/{0858,1015}.json (conf 0.05, imgsz 512)
# usage: bash scripts/infer_heldout.sh model1 model2 ...      (model = run dir name under runs/thermal_uav)
cd ~/day2thermal/detection && source ../.venv/bin/activate
for m in "$@"; do
  W=../runs/thermal_uav/$m/weights/best.pt; mkdir -p results/detections/$m
  [ -s results/detections/$m/0858.json ] || python3 src/deploy/run_yolo_inference.py --model $W --video ../data/boson_work_thermal/08_58_15_thermal.mp4 --out_json results/detections/$m/0858.json --progress_every 100000 2>&1 | tail -1
  [ -s results/detections/$m/1015.json ] || python3 src/deploy/run_yolo_inference.py --model $W --video ../data/boson_work_thermal/10_15_59_thermal.mp4 --out_json results/detections/$m/1015.json --progress_every 100000 2>&1 | tail -1
done
