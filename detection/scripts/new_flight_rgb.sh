#!/usr/bin/env bash
# New flight day (4K day + Boson thermal) -> RGB training tiles + human review queues. Same chain as the 2026-10-06 addon
# (notebook 2026-10-07/08), plus mine_day_misses.py (drones the thermal model sees and the day model misses).
#
#   scripts/new_flight_rgb.sh prep   RAW W      remux .ts -> mp4, sync, registration, overlay flags      (local, CPU)
#   scripts/new_flight_rgb.sh detect W [dev]    day model (tiled 4K, every 3rd frame) + thermal model    (GPU: VM)
#   scripts/new_flight_rgb.sh labels W          pos/hard/neg/tneg per thermal frame, tiles, review queues (local)
#   scripts/new_flight_rgb.sh review W          the human step (review_rgb_tiles.py), then --finalize
#
# RAW = folder of sessions as recorded: RAW/<date>T<HH_MM_SS>/<date>T<HH_MM_SS>.ts + ..._thermal.ts
# W   = work folder, e.g. ../data/flight_20261014. Session id = MMDD_HH_MM_SS (does not collide with Oct-06 ids).
# Usable time windows (plane on the ground, ...): optional W/windows.txt, lines "<session> --t-min 120 --t-max 900".
# Class order of everything written into W/rgb_addon is Fixed_wing_v3's (bird, airplane, uav) as declared in its
# data.yaml; resplit_by_source.py remaps it to uav = 0 when it is merged.
set -eo pipefail
D=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-$D/../.venv/bin/python}
RGB_MODEL=${RGB_MODEL:-$D/../../ANN-Detection/models/yolo26s_rgb_20261012/yolo26s_fw4_20261011.pt}
TH_MODEL=${TH_MODEL:-$D/../../ANN-Detection/models/thermal_yolo26sp2_v1.0_r02_261007/thermal_yolo26sp2_v1.0_r02_261007.pt}
V3=${V3:-$D/../../ANN-Detection/data/datasets/Fixed_wing_v3}
cmd=$1; shift
sessions() { ls "$1"/mp4/*_thermal.mp4 2>/dev/null | xargs -n1 basename | sed 's/_thermal.mp4//'; }

case $cmd in
prep)
  RAW=$1; W=$2; mkdir -p "$W"/mp4 "$W"/sync "$W"/reg
  for d in "$RAW"/*T*; do
    b=$(basename "$d"); s=$(echo "$b" | sed -E 's/^[0-9]{4}-([0-9]{2})-([0-9]{2})T/\1\2_/')
    [ -s "$d/$b.ts" ] && [ -s "$d/${b}_thermal.ts" ] || { echo "skip $b (day or thermal missing/empty)"; continue; }
    [ -s "$W/mp4/${s}_day.mp4" ] || ffmpeg -nostdin -loglevel error -i "$d/$b.ts" -map 0:v:0 -c copy "$W/mp4/${s}_day.mp4"
    [ -s "$W/mp4/${s}_thermal.mp4" ] || ffmpeg -nostdin -loglevel error -i "$d/${b}_thermal.ts" -map 0:v:0 -c copy "$W/mp4/${s}_thermal.mp4"
    echo "remuxed $s"
  done
  for s in $(sessions "$W"); do
    [ -s "$W/sync/$s/sync_report.json" ] || $PY $D/src/data/verify_stream_sync.py --day "$W/mp4/${s}_day.mp4" \
        --thermal "$W/mp4/${s}_thermal.mp4" --out "$W/sync/$s" --max_lag_s 6 --n_windows 4 > "$W/sync/$s.log" 2>&1 || echo "sync FAILED $s"
    [ -s "$W/reg/$s/registration.json" ] || (cd $D/.. && $PY -m day2thermal.video_pairs calibrate --day "$W/mp4/${s}_day.mp4" \
        --thermal "$W/mp4/${s}_thermal.mp4" --out "$W/reg/$s" --sync-report "$W/sync/$s/sync_report.json" --n-frames 60 \
        > "$W/reg/$s.log" 2>&1) || echo "registration FAILED $s"
    echo "$s: sync + registration done"
  done
  $PY $D/src/data/thermal_overlay_flags.py --mp4-dir "$W/mp4" --out "$W/overlay_flags.json" --sheet-dir "$W/overlay_sheets"
  ;;
detect)
  W=$1; DEV=${2:-0}; mkdir -p "$W"/rgbdets "$W"/thermdets
  for s in $(sessions "$W"); do
    [ -s "$W/rgbdets/$s.json" ] || $PY $D/src/data/rgb_label_videos.py --model "$RGB_MODEL" --every 3 --device "$DEV" \
        --video "$s:$W/mp4/${s}_day.mp4" --out "$W/rgbdets" 2>&1 | grep -v "WARN\|Warn" || true
    [ -s "$W/thermdets/$s.json" ] || $PY $D/src/deploy/run_yolo_inference.py --model "$TH_MODEL" --video "$W/mp4/${s}_thermal.mp4" \
        --out_json "$W/thermdets/$s.json" --imgsz 512 --device "$DEV" --progress_every 100000 2>&1 | tail -1
  done
  ;;
labels)
  W=$1; mkdir -p "$W"/labels
  for s in $(sessions "$W"); do
    [ -s "$W/reg/$s/registration.json" ] && [ -s "$W/rgbdets/$s.json" ] && [ -s "$W/thermdets/$s.json" ] || { echo "skip $s (missing reg/dets)"; continue; }
    WIN=$(grep -s "^$s " "$W/windows.txt" | cut -d' ' -f2- || true)
    $PY $D/src/data/build_boson_labels.py --session $s --registration "$W/reg/$s/registration.json" --rgb-json "$W/rgbdets/$s.json" \
        --therm-json "$W/thermdets/$s.json" --overlay "$W/overlay_flags.json" --out "$W/labels/$s.csv" $WIN
  done
  $PY $D/src/data/build_rgb_addon.py --labels-dir "$W/labels" --day-dir "$W/mp4" --v3 "$V3" --out "$W/rgb_addon"
  # FP mining assumes ONE drone in the air (any day detection far from the confirmed drone = false alarm).
  # Flights with two drones (interceptor + target): MULTI_DRONE=1 skips it.
  [ "${MULTI_DRONE:-0}" = 1 ] || $PY $D/src/data/mine_rgb_fp.py --labels-dir "$W/labels" --rgb-dir "$W/rgbdets" --day-dir "$W/mp4" --addon "$W/rgb_addon" --holdout ""
  $PY $D/src/data/mine_day_misses.py --labels-dir "$W/labels" --reg-dir "$W/reg" --day-dir "$W/mp4" --addon "$W/rgb_addon"
  echo "auto-labelled tiles: $(ls "$W"/rgb_addon/train/images 2>/dev/null | wc -l); review queue: $W/rgb_addon/review/tiles.csv"
  ;;
review)
  W=$1
  $PY $D/src/eval/review_rgb_tiles.py --addon "$W/rgb_addon"
  ;;
*) echo "usage: $0 prep RAW W | detect W [device] | labels W | review W"; exit 1;;
esac
