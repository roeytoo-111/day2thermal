# Thermal UAV Detector — RGB Transfer

**Run name:** `rgb_transfer`
**Purpose:** Test whether transferring from the production RGB drone detector's weights beats COCO-pretrained init on this (small, domain-shifted) thermal dataset. Compare directly against `coco_baseline` — same data, same config, only the starting weights differ.

## Base weights
`yolo11n_rgbdetection.pt` — the production RGB detector's trained weights (yolo11n architecture, no P2 head in this specific checkpoint). Full fine-tune, not frozen — all layers adapt to the thermal domain during training.

## Training data
Identical to `coco_baseline` — see that README for full detail. Same filtered set, same `data/thermal_1_filtered/data.yaml`, same splits.

## Training command
```bash
python3 src/train_thermal_yolo.py --data_yaml data/thermal_1_filtered/data.yaml --model yolo11n_rgbdetection.pt --name rgb_transfer --batch 8
```

## Key config
Identical to `coco_baseline` (`imgsz=512`, `epochs=150` ceiling, `patience=25`, `batch=8`) — deliberately kept the same so any performance difference is attributable to the starting weights, not a config change.

## Known caveats
Same domain-gap caveat as `coco_baseline` applies equally here — see that README. Additionally:
- The RGB source model was trained on 4K tiled input with very different target-scale statistics than this 512x512 thermal set; the transfer is a bet that mid/high-level "small aerial object" features generalize across that gap, not that low-level filters do (those were expected to, and should, adapt away from RGB-specific color/texture statistics during full fine-tuning).

## Results
_Fill in from `runs/thermal_uav/rgb_transfer/results.csv` (final epoch) once training completes:_
- mAP50-95: 0.59602
- mAP50: 0.97652
- Precision: 0.9753
- Recall: 0.97531
- Stopped at epoch: 150

