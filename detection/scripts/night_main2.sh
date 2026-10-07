#!/usr/bin/env bash
# Queue v2 (2026-10-07 11:30): the NEW thermal camera is the deployed one (Daniel), and its held-out human GT exists.
# Replaces night_main.sh after bosonM seed 0:  M2 (new-camera data x3 + new-camera sprites) -> M3 (M2 + old data zoomed
# in x2.2 to the new camera's angular resolution) -> generator retrain (8k steps) -> generator eval -> shutdown.
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
until [ -f ~/bosonM_seed0.done ]; do sleep 20; done
tmux kill-session -t main 2>/dev/null; sleep 3
for p in $(ps -eo pid,args | grep "[t]rain_controlnet" | awk '{print $1}'); do kill $p; done
git -C .. pull -q
rm -rf data/thermal-4-boson3 data/thermal-4-boson3P
python3 src/data/build_dataset.py --base data/thermal-3-pasteG --add data/boson_work/yolo_boson --oversample 3 \
    --keep_base_valid --out data/thermal-4-boson3 2>&1 | grep -E "built"
python3 src/data/paste_on_backgrounds.py build --work data/boson_work/bg_boson --out data/thermal-4-boson3P \
    --base data/thermal-4-boson3 --sprites_from data/boson_work/yolo_boson --tag bgbosonN --n_images 600 2>&1 | tail -2
bash scripts/train_and_infer.sh bosonM2 data/thermal-4-boson3P yolo26s-p2.yaml pretrained=$Y26 seed=0
bash scripts/infer_heldout.sh bosonM2; touch ~/bosonM2.done
rm -rf data/boson_work/zoomin_old data/thermal-4-boson3PZ
python3 src/data/zoomin_aug.py --src data/thermal-3-pasteG --out data/boson_work/zoomin_old --n 2000 2>&1 | tail -1
python3 src/data/build_dataset.py --base data/thermal-4-boson3P --add data/boson_work/zoomin_old --oversample 1 \
    --keep_base_valid --out data/thermal-4-boson3PZ 2>&1 | grep -E "built"
bash scripts/train_and_infer.sh bosonM3 data/thermal-4-boson3PZ yolo26s-p2.yaml pretrained=$Y26 seed=0
bash scripts/infer_heldout.sh bosonM3; touch ~/bosonM3.done
cd ~/day2thermal
python -m day2thermal.diffusion.train_controlnet --data data/diff_pairs_boson --out runs/diff_cn_boson \
    --init-controlnet runs/diff_cn_all/best --steps 8000 --batch 4 --grad-accum 2 --val-every 500 > ~/gen_train.log 2>&1
touch ~/gen_train.done
for m in boson all; do
  python -m day2thermal.diffusion.infer --controlnet runs/diff_cn_$m/best --input data/diff_pairs_boson/test/rgb \
      --out runs/diff_cn_$m/preds_boson_test > ~/gen_infer_$m.log 2>&1
  python -m day2thermal.diffusion.diagnose --pairs data/diff_pairs_boson --split test --pred-dir runs/diff_cn_$m/preds_boson_test \
      --out runs/diff_cn_$m/diag_boson_test > ~/gen_diag_$m.log 2>&1
done
touch ~/gen_eval.done
sleep 60; sudo shutdown -h now
