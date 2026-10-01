#!/usr/bin/env bash
# Pull detection JSONs for the given runs from the VM, then score them with the reference models.
# Run from detection/:   bash scripts/pull_and_score.sh OUT.md run1 run2 ...
set -eu
OUT=$1; shift
ARGS=""
for n in sessions_pinned pasteA pasteB "$@"; do
  if [ ! -s results/detections/$n/0730.json ]; then
    mkdir -p results/detections/$n
    scp -q "aerosentry-training:~/day2thermal/detection/results/detections/$n/*.json" results/detections/$n/
  fi
  ARGS="$ARGS --model $n results/detections/$n"
done
../.venv/bin/python src/eval/score_models.py $ARGS --out "$OUT"
