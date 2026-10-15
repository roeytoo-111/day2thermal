#!/usr/bin/env bash
# VM: day-model detections on the 08_58_15 from-281 s cut (Daniel's day GT, data/boson_work/cut0858/pipeline_gt/boson_0858),
# tiled, every 3rd frame (the GT frames). usage: bash scripts/cut0858_dets.sh LABEL WEIGHTS
cd ~/day2thermal/detection && source ../.venv/bin/activate
R=results/cut0858; mkdir -p $R
[ -s $R/day_$1.json ] || python3 src/deploy/detect_ts.py --video ../data/cut0858/2026-10-06T08_58_15_from_281.ts --model $2 --out $R/day_$1.json --every 3 2>&1 | tail -1
