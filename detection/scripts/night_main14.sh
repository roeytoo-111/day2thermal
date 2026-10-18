#!/usr/bin/env bash
# Queue v14 (2026-10-18): the terrain pastes (thermal-9-N2D-u0) with the plain head chosen in §11 (yolo26s.yaml), seeds 0
# and 1 in parallel, same init (rgbcurrent) and recipe. Compared with N2n (same data minus the pastes, plain head) and N2D (P2).
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
run() { bash scripts/train_and_infer.sh bosonN2Dn_s$1 data/thermal-9-N2D-u0 yolo26s.yaml pretrained=$Y26 seed=$1
        bash scripts/infer_heldout.sh bosonN2Dn_s$1; bash scripts/arsuf_dets.sh thermal bosonN2Dn_s$1; touch ~/N2Dn_s$1.done; }
run 0 & run 1 & wait
touch ~/N2Dn.done
