#!/usr/bin/env bash
# Master GPU queue (VM), run in tmux after the datasets exist. Sequential so the GPU never idles and never OOMs:
#   M seed0 (detector) -> generator retrain -> M seed1 -> generator eval (new vs old model on the same test pairs) -> shutdown
# Flags: ~/det_ready (thermal-4-bosonP built), ~/gen_ready (diff_pairs_boson uploaded). Each step logs a DONE marker.
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
until [ -f ~/det_ready ]; do sleep 20; done
bash scripts/train_and_infer.sh bosonM data/thermal-4-bosonP yolo26s-p2.yaml pretrained=$Y26 seed=0
touch ~/bosonM_seed0.done
cd ~/day2thermal
python -m day2thermal.diffusion.train_controlnet --data data/diff_pairs_boson --out runs/diff_cn_boson \
    --init-controlnet runs/diff_cn_all/best --steps 12000 --batch 4 --grad-accum 2 --val-every 500 > ~/gen_train.log 2>&1
touch ~/gen_train.done
cd ~/day2thermal/detection
bash scripts/train_and_infer.sh bosonM_seed1 data/thermal-4-bosonP yolo26s-p2.yaml pretrained=$Y26 seed=1
touch ~/bosonM_seed1.done
cd ~/day2thermal
for m in boson all; do
  ck=runs/diff_cn_$m/best
  python -m day2thermal.diffusion.infer --controlnet $ck --input data/diff_pairs_boson/test/rgb --out runs/diff_cn_$m/preds_boson_test > ~/gen_infer_$m.log 2>&1
  python -m day2thermal.diffusion.diagnose --pairs data/diff_pairs_boson --split test --pred-dir runs/diff_cn_$m/preds_boson_test \
      --out runs/diff_cn_$m/diag_boson_test > ~/gen_diag_$m.log 2>&1
done
touch ~/gen_eval.done
sleep 60; sudo shutdown -h now
