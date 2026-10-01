#!/usr/bin/env bash
# Train one YOLO run with the pinned settings, then log detections (conf 0.05) on the three test videos.
# Run from detection/:   bash scripts/train_and_infer.sh NAME DATA_DIR INIT_WEIGHTS [extra yolo args...]
#   e.g.  bash scripts/train_and_infer.sh pasteG data/thermal-3-pasteG yolo11np2-rgbdetection.pt
# Writes ../runs/thermal_uav/NAME/ and results/detections/NAME/{0708,0730,0715}.json, then
# results/detections/NAME/DONE (or FAILED). Re-running skips whatever already exists.
set -u
NAME=$1; DATA=$2; INIT=$3; shift 3
RUN=$PWD/../runs/thermal_uav/$NAME
OUT=results/detections/$NAME
mkdir -p "$OUT"
IMGSZ=512
for x in "$@"; do case $x in imgsz=*) IMGSZ=${x#imgsz=};; esac; done
if [ ! -f "$RUN/weights/best.pt" ]; then
  rm -rf "$RUN"
  yolo detect train model="$INIT" data="$PWD/$DATA/data.yaml" epochs=150 patience=25 batch=-1 imgsz=512 \
    optimizer=AdamW lr0=0.001667 seed=0 close_mosaic=15 project="$PWD/../runs/thermal_uav" name="$NAME" \
    exist_ok=True "$@" || { touch "$OUT/FAILED"; exit 1; }
fi
[ -f "$RUN/weights/best.pt" ] || { echo "no best.pt"; touch "$OUT/FAILED"; exit 1; }
declare -A V=([0708]=data/videos/thermal.mp4
              [0730]=../data/new_videos/clipped/2026-07-30T09_27_04_thermal_clipped.mp4
              [0715]=../data/new_videos/clipped/2026-07-15_thermal_clipped.mp4)
for k in 0708 0730 0715; do
  [ -s "$OUT/$k.json" ] && continue
  python3 src/deploy/run_yolo_inference.py --model "$RUN/weights/best.pt" --video "${V[$k]}" \
    --out_json "$OUT/$k.json" --imgsz "$IMGSZ" --progress_every 5000 || { touch "$OUT/FAILED"; exit 1; }
done
touch "$OUT/DONE"
