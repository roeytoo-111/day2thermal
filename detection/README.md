# Thermal UAV detection

YOLO11 detection of drones in LWIR thermal video (640×512 sensor), trained on
a small real thermal dataset, optionally extended with synthetic data.
Day-by-day history, decisions and every number are in
[`../notebook/`](../notebook/README.md). This file is the stable reference.

## Layout

```
detection/
  src/data/        dataset building: audit, filter, leak checks, leak removal, synthetic data, stream sync
  src/eval/        evaluation: GT sampling + labelling, recall/false-fire scoring, review tools
  src/deploy/      run a checkpoint over a video -> detection JSON
  models/<run>/    args.yaml, results.csv, readme.md per training run (weights are local only)
  results/
    detections/        thermal_dets_<run>.json, one per model, over data/videos/thermal.mp4
    precision_review/  confidence-bucket precision review manifests
  analysis/
    leakage/       pHash/SSIM train-vs-eval-video matches, hash tables
    dataset_audit/ per-image / per-box audit tables of the Roboflow set
  reports/         written reports; stream_sync/ (RGB/thermal time-offset verification)
  logs/
  data/            (not tracked) datasets, videos, GT frames
```

## Data

| Dataset (under `data/`, not tracked) | Built by | Train / val / test | Notes |
|---|---|---|---|
| `thermal-1` | Roboflow export `thermal-7eu6d` | — | Classes `0`, `bird`, `thermal-uav`. Class `0` is GAN output (confirmed by its creator). |
| `thermal-1-filtered` | `src/data/filter_dataset.py` | 5019 / 210 / 95 | Class `0` dropped. **Leaks:** 2260 train images (45%) are frames of the eval video. |
| `thermal-1-noleak` | `src/data/remove_session_leak.py` | 2759 / 210 / 95 | The whole leaked session removed (`leak_report.json`). **Use this.** |
| `thermal-1-noleak-synthA` | `src/data/synth_copy_paste.py` v1 | 5508 / 210 / 95 | +2749 images with pasted small drones, pasted anywhere. Not a win (see Runs). |
| `thermal-1-noleak-synthA2` | `src/data/synth_copy_paste.py` v2 (defaults) | 3675 / 210 / 95 | +916 images (25%), 1387 real-crop drones, pasted only on smooth, speck-free sky; terrain hot spots stay unlabelled as negatives. |

Real UAV boxes in `thermal-1-noleak`: median 40 px, 9% < 16 px. Drones in the
deployment video are ~13 px or smaller. That size gap is what the synthetic
data targets.

## Evaluation protocol

- **Eval set:** `data/recall_ground_truth/manifest.csv`. Every 40th frame of
  `data/videos/thermal.mp4`, labelled for drone presence without looking at
  any model: 257 positive / 340 empty. It is clean for every model trained on
  `thermal-1-noleak*`, and **only** for those.
- **Inference:** `src/deploy/run_yolo_inference.py`, which logs at conf ≥ 0.05
  (the default).
- **Scoring:** `src/eval/compute_recall_from_gt.py --conf_floor 0.1`, plus a
  threshold sweep. Compare models at a matched false-fire rate, not at a
  single threshold.
- **Location-aware scoring** (use this once `boxes.csv` exists): `--boxes
  data/recall_ground_truth/boxes.csv --sweep 0.1,...,0.7`. A positive frame
  counts only if a detection lands on the drone box, and the audit's
  corrections are applied.
  - Built by `src/eval/gt_boxes.py`. Box proposals come from a **model-free**
    small-target detector (motion-compensated temporal median plus top-hat,
    normalised by local clutter), never from a model being scored.
  - Every "empty" frame is audited, not only those where some model fired.
- **Legacy frame-level metric** (no `--boxes`): any detection within ±2
  frames counts, so a model that fires everywhere gets recall for free. The
  "empty" labels also miss some 1–3 px airborne objects (notebook 2026-09-29 §7).

## Runs

| Run | Init | Data | Recall / false-fire @ conf 0.1 (613 GT frames) | Notes |
|---|---|---|---|---|
| `coco_baseline` | yolo11n COCO | `thermal-1-filtered` (leaked) | 77.8% / 42.9% | Inflated by the leak; retired from comparisons |
| `rgb_transfer` | RGB drone detector | `thermal-1-filtered` (leaked) | 77.4% / 16.5% | Inflated by the leak; retired |
| **`rgb_transfer_noleak_p2`** | RGB detector with P2 head | `thermal-1-noleak` | **67.7% / 23.2%** | **Current honest baseline** |
| `rgb_transfer_noleak_p2_synthA` | same | `…-synthA` | 94.6% / 79.1% | Fires on any small bright point; at matched false-fire it's no better than the baseline |
| `rgb_transfer_noleak_p2_synthA2` | same | `…-synthA2` | pending | Generator v2 |

All training runs use imgsz 512, 150 epochs, patience 25, seed 0 (see each `models/<run>/args.yaml`).

## Typical commands (run from `detection/`)

```bash
# leak-free dataset
python3 src/data/remove_session_leak.py --dataset_dir data/thermal-1-filtered \
    --output_dir data/thermal-1-noleak --leakage_csv analysis/leakage/leakage_verified.csv \
    --video data/videos/thermal.mp4
# synthetic small drones (option A)
python3 src/data/synth_copy_paste.py --dataset_dir data/thermal-1-noleak --output_dir data/thermal-1-noleak-synthA
# inference + scoring
python3 src/deploy/run_yolo_inference.py --model models/<run>/weights/best.pt \
    --video data/videos/thermal.mp4 --out_json results/detections/thermal_dets_<run>.json
python3 src/eval/compute_recall_from_gt.py --manifest data/recall_ground_truth/manifest.csv \
    --json results/detections/thermal_dets_rgb_transfer_noleak_p2.json:p2_noleak --conf_floor 0.1
```
