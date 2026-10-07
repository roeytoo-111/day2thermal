#!/usr/bin/env bash
# Queue v5 (2026-10-07 12:40): today's main goal is an improved RGB (day) model (Daniel's boss).
#   wait for bosonM2 (already training) -> RGB fine-tune of the production model on Fixed_wing_v3 + Boson addon
#   -> RGB evaluation (v3 test + held-out Boson tiles, baseline vs new) -> thermal M4 (= M2 + Daniel's reviewed boxes)
#   -> shutdown. Dropped today: M3 (zoom-in) and the generator retrain.
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
until [ -f results/detections/bosonM2/DONE ] || [ -f results/detections/bosonM2/FAILED ]; do sleep 30; done
bash scripts/infer_heldout.sh bosonM2; touch ~/bosonM2.done
until [ -f ~/rgb_data_ready ]; do sleep 20; done
D=~/rgbdata/fw_v3boson
yolo detect train model=$Y26 data=$D/data.yaml epochs=40 patience=10 imgsz=640 batch=-1 optimizer=AdamW lr0=0.001 \
    seed=0 close_mosaic=5 project=$HOME/day2thermal/runs/rgb name=rgb_v3boson exist_ok=True > ~/rgb_train.log 2>&1
touch ~/rgb_train.done
for m in base new; do
  W=$Y26; [ $m = new ] && W=~/day2thermal/runs/rgb/rgb_v3boson/weights/best.pt
  yolo detect val model=$W data=$D/data.yaml split=test imgsz=640 batch=16 project=$HOME/day2thermal/runs/rgb name=val_test_$m exist_ok=True > ~/rgb_val_test_$m.log 2>&1
  yolo detect val model=$W data=$D/data_boson_test.yaml split=val imgsz=640 batch=16 project=$HOME/day2thermal/runs/rgb name=val_boson_$m exist_ok=True > ~/rgb_val_boson_$m.log 2>&1
done
touch ~/rgb_eval.done
rm -rf data/thermal-4-boson3PR
python3 src/data/build_dataset.py --base data/thermal-4-boson3P --add data/boson_work/yolo_boson_reviewed --oversample 2 \
    --keep_base_valid --out data/thermal-4-boson3PR 2>&1 | grep -E "built"
bash scripts/train_and_infer.sh bosonM4 data/thermal-4-boson3PR yolo26s-p2.yaml pretrained=$Y26 seed=0
bash scripts/infer_heldout.sh bosonM4; touch ~/bosonM4.done
sleep 60; sudo shutdown -h now
