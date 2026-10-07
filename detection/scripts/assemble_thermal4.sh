#!/usr/bin/env bash
# VM: thermal-3-pasteG (the dataset behind the current best model L) + real Boson labels + pastes on NEW real backgrounds.
# Needs ~/day2thermal/detection/data/boson_work/{yolo_boson,bg_boson} (uploaded from the laptop).
set -e
cd ~/day2thermal/detection && source ../.venv/bin/activate
rm -rf data/thermal-4-boson data/thermal-4-bosonP
python3 src/data/build_dataset.py --base data/thermal-3-pasteG --add data/boson_work/yolo_boson --oversample 1 \
    --keep_base_valid --out data/thermal-4-boson 2>&1 | grep -E "built|TRAIN"
NBG=$(($(wc -l < data/boson_work/bg_boson/backgrounds.csv)-1))
python3 src/data/paste_on_backgrounds.py build --work data/boson_work/bg_boson --out data/thermal-4-bosonP \
    --base data/thermal-4-boson --tag bgboson --n_images ${1:-1200} 2>&1 | tail -2
echo "backgrounds: $NBG"
