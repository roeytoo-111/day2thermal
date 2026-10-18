#!/usr/bin/env bash
# Queue v11 (2026-10-18): the latest thermal models WITHOUT the P2 head, on the same data and from the same starting weights.
#   M0n = thermal-5-bosonP-u0 (M0 recipe; M0 s1 is the recommended deployable v1.0 r02) , N2n = thermal-7-N2-u0 (v1.1 r01 data)
#   model yolo26s.yaml (detection at strides 8/16/32) instead of yolo26s-p2.yaml; pretrained=rgbcurrent (all layers transfer
#   except the class head); every other argument as in train_and_infer.sh (150 epochs max, patience 25, AdamW, imgsz 512).
#   Seeds 0 and 1 of each (thermal seeds differ by up to 15 points). Then Boson held-out + Arsuf bench thermal detections.
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
for d in "M0n data/thermal-5-bosonP-u0" "N2n data/thermal-7-N2-u0"; do
  set -- $d; n=$1; data=$2
  for s in 0 1; do
    bash scripts/train_and_infer.sh boson${n}_s$s $data yolo26s.yaml pretrained=$Y26 seed=$s
    bash scripts/infer_heldout.sh boson${n}_s$s
    bash scripts/arsuf_dets.sh thermal boson${n}_s$s
    touch ~/${n}_s$s.done
  done
done
touch ~/noP2.done
