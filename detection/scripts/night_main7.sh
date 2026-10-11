#!/usr/bin/env bash
# Queue v7 (2026-10-11). Criteria: notebook/2026-10-11.md section 3 (written before launch).
#   Arsuf 10-05 detections (thermal M0 every frame, day RGB every 3rd frame tiled) -> Svanström -> YOLO
#   -> zero-shot val of M0s1 on the Svanström test split -> S1 (M0 data + Svanström) seeds 0,1
#   -> capacity runs C_s / C_m / C_l (yolo26{s,m,l}-p2 from COCO on M0 data) -> shutdown.
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
A=../data/arsuf_1005; R=results/arsuf; mkdir -p $R
[ -s $R/th_M0s1.json ] || python3 src/deploy/detect_ts.py --video $A/th_output.ts --model ../runs/thermal_uav/bosonM0u_s1/weights/best.pt \
    --out $R/th_M0s1.json --tile 0 --every 1 > ~/arsuf_th.log 2>&1
[ -s $R/day_rgbcurrent.json ] || python3 src/deploy/detect_ts.py --video $A/day_output.ts --model $Y26 \
    --out $R/day_rgbcurrent.json --every 3 --batch 32 > ~/arsuf_day.log 2>&1
touch ~/arsuf_det.done
[ -d data/svanstrom_yolo ] || python3 src/data/svanstrom_to_yolo.py --src ../data/svanstrom/Video_IR --out data/svanstrom_yolo > ~/svan_build.log 2>&1
yolo detect val model=../runs/thermal_uav/bosonM0u_s1/weights/best.pt data=$PWD/data/svanstrom_yolo/data.yaml split=test \
    imgsz=512 conf=0.001 project=$PWD/../runs/thermal_uav name=val_svan_M0s1 exist_ok=True > ~/svan_val_M0s1.log 2>&1
# S1 data: M0's u0 set + Svanström train, hard links (a folder symlink is resolved back to the source labels)
python3 - <<'EOF'
import os, glob
base, add, out = "data/thermal-5-bosonP-u0", "data/svanstrom_yolo", "data/thermal-6-bosonP-svan-u0"
for split, srcs in (("train", (base, add)), ("valid", (base,)), ("test", (base,))):
    for sub in ("images", "labels"):
        os.makedirs(f"{out}/{split}/{sub}", exist_ok=True)
        for s in srcs:
            for f in glob.glob(f"{s}/{split}/{sub}/*"):
                d = f"{out}/{split}/{sub}/{os.path.basename(f)}"
                if not os.path.exists(d):
                    os.link(os.path.realpath(f), d)
open(f"{out}/data.yaml", "w").write(open(f"{base}/data.yaml").read())
print({s: len(os.listdir(f"{out}/{s}/images")) for s in ("train", "valid", "test")})
EOF
for s in 0 1; do
  bash scripts/train_and_infer.sh bosonS1_s$s data/thermal-6-bosonP-svan-u0 yolo26s-p2.yaml pretrained=$Y26 seed=$s
  bash scripts/infer_heldout.sh bosonS1_s$s
done
touch ~/s1.done
for m in s m l; do
  bash scripts/train_and_infer.sh bosonC_$m data/thermal-5-bosonP-u0 yolo26$m-p2.yaml pretrained=yolo26$m.pt seed=0
  bash scripts/infer_heldout.sh bosonC_$m; touch ~/c_$m.done
done
sleep 60; sudo shutdown -h now
