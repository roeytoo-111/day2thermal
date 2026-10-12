#!/usr/bin/env bash
# Queue v9 (2026-10-12): thermal from the new RGB weights (yolo26s_fw4_20261011). Criteria: notebook/2026-10-12.md section 5.
# Started on a fresh boot after queue v8 has finished and shut the VM down.
cd ~/day2thermal/detection && source ../.venv/bin/activate
FW4=yolo26s_fw4_20261011.pt
for d in "R0 data/thermal-5-bosonP-u0" "R2 data/thermal-7-N2-u0"; do
  set -- $d; n=$1; data=$2
  for s in 0 1; do
    bash scripts/train_and_infer.sh boson${n}_s$s $data yolo26s-p2.yaml pretrained=$FW4 seed=$s
    bash scripts/infer_heldout.sh boson${n}_s$s
    bash scripts/arsuf_dets.sh thermal boson${n}_s$s
    touch ~/${n}_s$s.done
  done
done
sleep 60; sudo shutdown -h now
