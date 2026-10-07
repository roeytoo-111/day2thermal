#!/usr/bin/env bash
# Queue v4 (2026-10-07 12:10). bosonM2 is already training (detached from v3). Then:
#   M3 = M2 + old data zoomed in x2.2 -> M4 = M3 + Daniel-reviewed thermal-only detections (y = drone positives,
#   n = verified false alarms/clouds as hard negatives), only if ~/tneg_reviewed exists by then -> generator (5k) -> eval -> shutdown
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
until [ -f results/detections/bosonM2/DONE ] || [ -f results/detections/bosonM2/FAILED ]; do sleep 30; done
bash scripts/infer_heldout.sh bosonM2; touch ~/bosonM2.done
rm -rf data/boson_work/zoomin_old data/thermal-4-boson3PZ
python3 src/data/zoomin_aug.py --src data/thermal-3-pasteG --out data/boson_work/zoomin_old --n 2000 2>&1 | tail -1
python3 src/data/build_dataset.py --base data/thermal-4-boson3P --add data/boson_work/zoomin_old --oversample 1 \
    --keep_base_valid --out data/thermal-4-boson3PZ 2>&1 | grep -E "built"
bash scripts/train_and_infer.sh bosonM3 data/thermal-4-boson3PZ yolo26s-p2.yaml pretrained=$Y26 seed=0
bash scripts/infer_heldout.sh bosonM3; touch ~/bosonM3.done
if [ -f ~/tneg_reviewed ]; then
  rm -rf data/thermal-4-boson3PZR
  python3 src/data/build_dataset.py --base data/thermal-4-boson3PZ --add data/boson_work/yolo_boson_reviewed --oversample 2 \
      --keep_base_valid --out data/thermal-4-boson3PZR 2>&1 | grep -E "built"
  bash scripts/train_and_infer.sh bosonM4 data/thermal-4-boson3PZR yolo26s-p2.yaml pretrained=$Y26 seed=0
  bash scripts/infer_heldout.sh bosonM4; touch ~/bosonM4.done
fi
cd ~/day2thermal
python -m day2thermal.diffusion.train_controlnet --data data/diff_pairs_boson --out runs/diff_cn_boson \
    --init-controlnet runs/diff_cn_all/best --steps 5000 --batch 4 --grad-accum 2 --val-every 500 > ~/gen_train.log 2>&1
touch ~/gen_train.done
for m in boson all; do
  python -m day2thermal.diffusion.infer --controlnet runs/diff_cn_$m/best --input data/diff_pairs_boson/test/rgb \
      --out runs/diff_cn_$m/preds_boson_test > ~/gen_infer_$m.log 2>&1
  python -m day2thermal.diffusion.diagnose --pairs data/diff_pairs_boson --split test --pred-dir runs/diff_cn_$m/preds_boson_test \
      --out runs/diff_cn_$m/diag_boson_test > ~/gen_diag_$m.log 2>&1
done
touch ~/gen_eval.done
sleep 60; sudo shutdown -h now
