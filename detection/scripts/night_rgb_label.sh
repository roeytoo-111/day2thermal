#!/usr/bin/env bash
# GPU job 1 (VM): tiled RGB detection on every uploaded day video, waiting for sessions that are still uploading.
# rsync only renames a file into place when it is complete, so "exists" == "fully uploaded".
cd ~/day2thermal/detection && source ../.venv/bin/activate
B=~/day2thermal/data/boson_oct06; OUT=~/day2thermal/data/boson_work/rgbdets; mkdir -p $OUT
SESS="08_58_15 09_06_16 09_07_28 09_09_29 09_58_37 10_00_40 10_04_01 10_05_11 10_08_31 10_11_46 10_12_48 10_13_12 10_14_01 10_15_28 10_15_59"
for s in $SESS; do
  d=$B/2026-10-06T$s
  until [ -f $d/2026-10-06T$s.ts ] && [ -f $d/2026-10-06T${s}_thermal.ts ]; do sleep 20; done
  python3 src/data/rgb_label_videos.py --model yolo26s_20260921_145030_rgbcurrent.pt --every 3 \
      --video $s:$d/2026-10-06T$s.ts --out $OUT 2>&1 | grep -v "WARN\|Warn" 
done
touch ~/rgb_label.done
