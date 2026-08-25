# Thermal UAV Detector — COCO Baseline

**Run name:** `coco_baseline`
**Purpose:** Reference baseline (per project roadmap step 3) — the number all later data/model changes have to beat.

## Base weights
`yolo11n.pt` — stock Ultralytics COCO-pretrained checkpoint. No transfer from the RGB detector.

## Training data
- Source: Roboflow project `thermal-7eu6d` (workspace `uavs-somnd`), originally 3 classes: `0`, `bird`, `thermal-uav`.
- **Class `0` excluded.** Audited (`audit_thermal_data.py`) and confirmed contaminated: abnormal box-to-frame area ratio, high OCR text-hit-rate, internal duplicate boxes. Later confirmed by the dataset's creator — class `0` is GAN-generated output, not real sensor data, so this was correctly excluded, not a data-quality false alarm.
- `bird`/`thermal-uav` confirmed by the creator to come from real IR recordings (multiple sessions; only one is paired with RGB).
- Filtered with `filter_dataset.py` → `nc=2`, `['bird', 'thermal-uav']`.
- Path used: `data/thermal_1_filtered/data.yaml`
- Split sizes: 5,019 train images (333 background), 210 valid images (18 background), remainder test (~95).

## Training command
```bash
python3 src/train_thermal_yolo.py --data_yaml data/thermal_1_filtered/data.yaml --model yolo11n.pt --name coco_baseline --batch 8
```

## Key config
- `imgsz=512` — matches native dataset resolution exactly (do not increase; source is natively 512x512 from a 640x512 sensor, no extra detail to gain).
- `epochs=150` ceiling, `patience=25` (early stopping on val mAP50-95, not a fixed epoch count).
- `batch=8` manually set for debugging; AutoBatch had found 64 fits comfortably (13GB/22GB on the L4) — worth trying `--batch -1` on future runs for faster training.
- Standard YOLO11n architecture (no P2 head in this run — considered for future small-target-focused runs, not yet implemented here).

## Known caveats for this dataset/run
- **Domain gap vs. real deployment footage confirmed and unresolved as of this run.** Resolution-controlled sharpness comparison (`thermal-uav` training class vs. `thermal.mp4` ground-truth video): 420.93 (training) vs. 1163.79 (real, resized to match) — training data noticeably less sharp than actual deployment camera output. Treat this baseline's eval numbers as a floor, not a target.
- Test split (~95 images) is thin; a larger/independent eval set from the ground-truth video is planned (see `gt_track_out/`, `clean_detections.py` pipeline).

## Results
- **mAP50-95:** 0.5960 (59.6%)
- **mAP50:** 0.9765 (97.7%)
- **Precision:** 0.8659 (86.6%)
- **Recall:** 0.9753 (97.5%)
- **Stopped at epoch:** 150