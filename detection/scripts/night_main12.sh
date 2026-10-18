#!/usr/bin/env bash
# Queue v12 (2026-10-18), after v11 (no-P2): N2D = thermal-7-N2-u0 + 800 terrain pastes in black / grey / white shades
# (src/data/paste_terrain.py, data/thermal-9-N2D-u0). Same recipe, architecture (yolo26s-p2) and starting weights as N2,
# seeds 0 and 1, so the pastes are the only change against bosonN2_s0 / bosonN2_s1. Then held-out + Arsuf detections.
cd ~/day2thermal/detection && source ../.venv/bin/activate
until [ -f ~/noP2.done ]; do sleep 60; done
Y26=yolo26s_20260921_145030_rgbcurrent.pt
for s in 0 1; do
  bash scripts/train_and_infer.sh bosonN2D_s$s data/thermal-9-N2D-u0 yolo26s-p2.yaml pretrained=$Y26 seed=$s
  bash scripts/infer_heldout.sh bosonN2D_s$s
  bash scripts/arsuf_dets.sh thermal bosonN2D_s$s
  touch ~/N2D_s$s.done
done
touch ~/N2D.done
