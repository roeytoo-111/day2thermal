# Thermal UAV Detector: P2, leak-free, + synthetic small drones (option A v1)

**Run name:** `rgb_transfer_noleak_p2_synthA`
**Purpose:** Test whether pasted small drones (3–24 px) close the small-target recall gap. The only change from `rgb_transfer_noleak_p2` is the training data.

## Base weights
`yolo11np2-rgbdetection.pt` (same as `rgb_transfer_noleak_p2`).

## Training data
`data/thermal-1-noleak-synthA/data.yaml`: the 2759 real train images plus 2749 synthetic ones (5544 pasted drones: ⅔ downscaled real crops, ⅓ parametric blobs, pasted anywhere with background level ≤ 200). Built by `src/data/synth_copy_paste.py` v1. Val/test are identical to `thermal-1-noleak`.

## Key config
Identical to `rgb_transfer_noleak_p2` (see `args.yaml`).

## Results
- **Val:** best epoch 134. P 0.992, R 0.992, mAP50 0.993, mAP50-95 0.653.
- **Eval video, 613 GT frames** (`--conf_floor 0.1`): recall 94.6%, false-fire **79.1%**. That recall is **not usable**:
  - The model averages ~3 detections per frame, and frame-level recall doesn't check location.
  - At a matched false-fire rate it's no better than the baseline (~62% vs 67.7% at ~23% false-fire).
- **What went wrong:** it learned "small bright point = drone". It fires on static terrain hot spots and clutter. It also fires on some real tiny moving objects that the GT marks as empty.
- **Full breakdown:** notebook 2026-09-29 §7. The fix is generator v2: sky-only pastes, real crops only, a ~20–25% synthetic share.
