#!/usr/bin/env bash
# Queue v6 (2026-10-07 night). Criteria: notebook/2026-10-07.md section 14 (written before launch).
# Thermal, all uav = 0 (swap_uav_class.py copies; images symlinked):
#   M4 seed 1 -> M0 seed 1 (two seeds each, so M4 vs M0 can be called) -> M6 = M4 data from the NEW RGB weights
# Generator: old model on the Boson test (control) -> G1 continue diff_cn_all on diff_pairs_boson (8k)
#   -> G2 = G1 + Min-SNR gamma 5 (one change) -> eval each -> shutdown.
cd ~/day2thermal/detection && source ../.venv/bin/activate
Y26=yolo26s_20260921_145030_rgbcurrent.pt
RGBNEW=$HOME/day2thermal/runs/rgb/rgb_v3boson/weights/best.pt
python3 src/data/swap_uav_class.py --src data/thermal-4-bosonP --out data/thermal-5-bosonP-u0
python3 src/data/swap_uav_class.py --src data/thermal-4-boson3PR --out data/thermal-5-boson3PR-u0
bash scripts/train_and_infer.sh bosonM4u_s1 data/thermal-5-boson3PR-u0 yolo26s-p2.yaml pretrained=$Y26 seed=1
bash scripts/infer_heldout.sh bosonM4u_s1; touch ~/m4s1.done
bash scripts/train_and_infer.sh bosonM0u_s1 data/thermal-5-bosonP-u0 yolo26s-p2.yaml pretrained=$Y26 seed=1
bash scripts/infer_heldout.sh bosonM0u_s1; touch ~/m0s1.done
bash scripts/train_and_infer.sh bosonM6 data/thermal-5-boson3PR-u0 yolo26s-p2.yaml pretrained=$RGBNEW seed=0
bash scripts/infer_heldout.sh bosonM6; touch ~/m6.done
cd ~/day2thermal
gen_eval() {   # $1 = run dir name under runs/
  python -m day2thermal.diffusion.infer --controlnet runs/$1/best --input data/diff_pairs_boson/test/rgb \
      --out runs/$1/preds_boson_test > ~/gen_infer_$1.log 2>&1
  python -m day2thermal.diffusion.diagnose --pairs data/diff_pairs_boson --split test --pred-dir runs/$1/preds_boson_test \
      --out runs/$1/diag_boson_test > ~/gen_diag_$1.log 2>&1
}
gen_eval diff_cn_all; touch ~/g0.done
python -m day2thermal.diffusion.train_controlnet --data data/diff_pairs_boson --out runs/diff_cn_boson \
    --init-controlnet runs/diff_cn_all/best --steps 8000 --batch 4 --grad-accum 2 --val-every 500 > ~/gen_train_g1.log 2>&1
gen_eval diff_cn_boson; touch ~/g1.done
rm -rf runs/diff_cn_boson/last                                   # disk: keep best only
python -m day2thermal.diffusion.train_controlnet --data data/diff_pairs_boson --out runs/diff_cn_boson_snr \
    --init-controlnet runs/diff_cn_all/best --steps 8000 --batch 4 --grad-accum 2 --val-every 500 --snr-gamma 5 \
    > ~/gen_train_g2.log 2>&1
gen_eval diff_cn_boson_snr; touch ~/g2.done
sleep 60; sudo shutdown -h now
