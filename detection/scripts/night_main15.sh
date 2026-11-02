#!/usr/bin/env bash
# Queue v15 (2026-11-02): thermal = M0 data + 07_38 (thermal_add_f0738: Daniel's 330 reviewed frames + ~1,100 unflagged,
# overlay-free auto boxes, end of flight as validation). Plain head (yolo26s.yaml), rgbcurrent init, M0n recipe, seeds 0+1
# in parallel; compared with bosonM0n_s0 / s1 (same data without 07_38). Then held-out Boson + Arsuf + test videos (as
# always) + the three tuning flights' clean thermal (scenario1, 13_21, 13_18) for every model. Ends with the VM shutting
# ITSELF down (a local watcher dies with the session).
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
D=data/thermal-10-M0f0738-u0
if [ ! -d $D ]; then
  mkdir -p $D
  for s in train valid test; do for sub in images labels; do mkdir -p $D/$s/$sub
    cp -al data/thermal-5-bosonP-u0/$s/$sub/. $D/$s/$sub/ 2>/dev/null
    [ -d ../data/thermal_add_f0738/$s/$sub ] && cp -al ../data/thermal_add_f0738/$s/$sub/. $D/$s/$sub/
  done; done
  sed "s#^path:.*##" data/thermal-5-bosonP-u0/data.yaml > $D/data.yaml
fi
run() { bash scripts/train_and_infer.sh bosonM0f_s$1 $D yolo26s.yaml pretrained=$Y26 seed=$1
        bash scripts/infer_heldout.sh bosonM0f_s$1; bash scripts/arsuf_dets.sh thermal bosonM0f_s$1; }
run 0 & run 1 & wait
mkdir -p results/tuning
for m in bosonM0f_s0 bosonM0f_s1 bosonM0n_s0 bosonM0n_s1 bosonM0u_s1; do
  for v in scenario1:../data/tuning/2026-10-20Tscenario1_clean_thermal.ts f1321:../data/tuning/2026-10-26T13_21_47_clean_thermal.ts \
           f1318:../data/tuning/2026-10-26T13_18_41_clean_thermal.ts; do
    k=${v%%:*}; f=${v#*:}
    [ -s results/tuning/${k}_$m.json ] || python3 src/deploy/detect_ts.py --video $f --model ../runs/thermal_uav/$m/weights/best.pt \
        --out results/tuning/${k}_$m.json --tile 0 --every 1 --imgsz 512 2>&1 | tail -1
  done
done
touch ~/M0f.done
sudo shutdown -h +2
