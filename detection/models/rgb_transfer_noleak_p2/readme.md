# Thermal UAV Detector: RGB transfer, P2 head, leak-free

**Run name:** `rgb_transfer_noleak_p2`
**Purpose:** The first honest baseline. Same idea as `rgb_transfer`, but trained without the leaked eval-video session, and with a P2 (stride-4) head for small targets.

## Base weights
`yolo11np2-rgbdetection.pt`: the production RGB drone detector, trained with the P2 head, so the whole network including the P2 branch is pretrained.

## Training data
`data/thermal-1-noleak/data.yaml`: `thermal-1-filtered` minus the 2260-image session that is the eval video (`src/data/remove_session_leak.py`). 2759 train / 210 val / 95 test.

## Key config
imgsz 512, epochs 150, patience 25, batch -1 (AutoBatch), seed 0, default augmentation (scale 0.5, mosaic 1.0). See `args.yaml`.

## Results
- **Val (Roboflow split):** best epoch 134. P 0.970, R 0.973, mAP50 0.986, mAP50-95 0.622. The val split shares sessions with train, so this is optimistic.
- **Eval video, 613 GT frames** (`results/detections/thermal_dets_rgb_transfer_noleak_p2.json`, `--conf_floor 0.1`): **recall 67.7% [61.8–73.1], false-fire 23.2% [19.1–28.0]**.
- **Threshold sweep:** notebook 2026-09-29 §7.

## Caveats
Two things changed at once versus `rgb_transfer` (leak fix and P2 head), so P2's own effect isn't isolated. That isolation run was skipped by decision on 2026-09-29.
