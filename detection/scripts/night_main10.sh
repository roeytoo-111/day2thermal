#!/usr/bin/env bash
# Queue v10 (2026-10-13): single-model ways to get the ensemble's gain. Criteria: notebook/2026-10-13.md section 4.
#   SWA of N2 s0 and M0 s1 (15 epochs at constant lr, average the last 10, BN re-estimated)
#   -> ensemble pseudo-labels (A = M0 s1 + N2 s0 + R0 s0) on unlabelled TRAINING-session frames
#   -> students KD1 s0/s1 = N2 data + pseudo-labels -> shutdown.
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt; T=../runs/thermal_uav
for src in "bosonN2_s0 data/thermal-7-N2-u0" "bosonM0u_s1 data/thermal-5-bosonP-u0"; do
  set -- $src; r=$1; data=$2
  if [ ! -f $T/${r}_swa/weights/swa.pt ]; then
    yolo detect train model=$T/$r/weights/best.pt data=$PWD/$data/data.yaml epochs=15 lr0=0.0005 lrf=1.0 warmup_epochs=0 \
      optimizer=AdamW imgsz=512 batch=-1 close_mosaic=0 save_period=1 patience=100 seed=0 project=$PWD/$T name=${r}_swa exist_ok=True
    python3 src/train/swa_average.py --run $T/${r}_swa --data $data --last-n 10 --out $T/${r}_swa/weights/swa.pt
  fi
  mkdir -p results/detections/${r}_swa
  for k in 0708:data/videos/thermal.mp4 0730:../data/new_videos/clipped/2026-07-30T09_27_04_thermal_clipped.mp4 \
           0715:../data/new_videos/clipped/2026-07-15_thermal_clipped.mp4 0858:../data/boson_work_thermal/08_58_15_thermal.mp4 \
           1015:../data/boson_work_thermal/10_15_59_thermal.mp4; do
    [ -s results/detections/${r}_swa/${k%%:*}.json ] || python3 src/deploy/run_yolo_inference.py --model $T/${r}_swa/weights/swa.pt \
        --video ${k#*:} --out_json results/detections/${r}_swa/${k%%:*}.json --progress_every 100000
  done
  python3 src/deploy/detect_ts.py --video ../data/arsuf_1005/th_output.ts --model $T/${r}_swa/weights/swa.pt \
      --out results/arsuf/th_${r}_swa.json --tile 0 --every 5 --imgsz 512
  touch ~/swa_$r.done
done
[ -f data/boson_work/pseudo_ensA/summary.json ] || python3 src/data/make_pseudo_labels.py \
    --models $T/bosonM0u_s1/weights/best.pt $T/bosonN2_s0/weights/best.pt $T/bosonR0_s0/weights/best.pt \
    --videos ../data/boson_work_thermal/*_thermal.mp4 ../data/new_videos/clipped/2026-06-23_scenario2_thermal_clipped.mp4 \
             ../data/new_videos/clipped/2026-07-15_thermal_clipped.mp4 \
    --max-t 09_09_29:30,2026-06-23_scenario2:380,2026-07-15:370 --out data/boson_work/pseudo_ensA > ~/pseudo.log 2>&1
python3 - <<'PY'
import os, glob
base, add, out = "data/thermal-7-N2-u0", "data/boson_work/pseudo_ensA", "data/thermal-8-KD1-u0"
for split, srcs in (("train", (f"{base}/train", add)), ("valid", (f"{base}/valid",)), ("test", (f"{base}/test",))):
    for sub in ("images", "labels"):
        os.makedirs(f"{out}/{split}/{sub}", exist_ok=True)
        for s in srcs:
            for f in glob.glob(f"{s}/{sub}/*"):
                d = f"{out}/{split}/{sub}/{os.path.basename(f)}"
                if not os.path.exists(d): os.link(os.path.realpath(f), d)
open(f"{out}/data.yaml", "w").write(open(f"{base}/data.yaml").read())
print({s: len(os.listdir(f"{out}/{s}/images")) for s in ("train", "valid", "test")})
PY
touch ~/pseudo.done
for s in 0 1; do
  bash scripts/train_and_infer.sh bosonKD1_s$s data/thermal-8-KD1-u0 yolo26s-p2.yaml pretrained=$Y26 seed=$s
  bash scripts/infer_heldout.sh bosonKD1_s$s
  bash scripts/arsuf_dets.sh thermal bosonKD1_s$s
  touch ~/KD1_s$s.done
done
sleep 60; sudo shutdown -h now
