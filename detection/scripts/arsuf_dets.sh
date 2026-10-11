#!/usr/bin/env bash
# VM: detections on exactly the Arsuf review frames (thermal every 5th decoded frame, day every 6th), keyed by pts_time.
# usage: bash scripts/arsuf_dets.sh thermal NAME...   (NAME = run dir under runs/thermal_uav)
#        bash scripts/arsuf_dets.sh day LABEL WEIGHTS
cd ~/day2thermal/detection && source ../.venv/bin/activate
A=../data/arsuf_1005; R=results/arsuf; mkdir -p $R
if [ "$1" = thermal ]; then
  shift
  for m in "$@"; do
    [ -s $R/th_$m.json ] || python3 src/deploy/detect_ts.py --video $A/th_output.ts --model ../runs/thermal_uav/$m/weights/best.pt \
        --out $R/th_$m.json --tile 0 --every 5 2>&1 | tail -1
  done
else
  [ -s $R/day_$2.json ] || python3 src/deploy/detect_ts.py --video $A/day_output.ts --model $3 --out $R/day_$2.json --every 6 2>&1 | tail -1
fi
