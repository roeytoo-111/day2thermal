# Thermal UAV Detector — Baseline Evaluation Report

**Models evaluated:** `coco_baseline` (YOLO11n, COCO-pretrained) vs. `rgb_transfer` (YOLO11n, transfer-learned from the production RGB detector). Same architecture, same training data, same config — only the starting weights differ. See individual READMEs for full training detail.

## Executive summary

Both baselines catch roughly the same fraction of real drone-present frames (**~78% recall, statistically tied**). The real difference is false-alarm rate: `rgb_transfer` fires on empty background far less often (**16.5% vs. 42.9%** of confirmed-empty frames). **RGB transfer's benefit is specificity, not sensitivity** — it doesn't see more drones, it hallucinates fewer. The ~22% recall gap is shared by both models and is the actual priority for improvement; neither training approach touched it.

## Methodology

**Data.** Original 17k-image dataset had ~11k images confirmed contaminated (GAN-generated, not real sensor data — confirmed independently through audit checks and directly by the dataset's creator). Filtered to ~5k real thermal images (`bird`, `thermal-uav`) for training.

**Domain gap.** Resolution-controlled sharpness comparison between the training data and real deployment footage (`thermal.mp4`) showed a ~2.8–3x gap (420.93 training vs. 1163.79 real, matched resolution). Kept in mind throughout as context for why detection is harder here than on the RGB side.

**Evaluation — two independent measurements, deliberately separated:**

1. **Precision / calibration** — detections proposed by each model on the real thermal video were sampled stratified by confidence bucket and manually labeled true/false. This measures: *when a model fires, how often is it right, and does its confidence score mean anything.*
2. **Recall** — a systematic, model-independent sample of 613 frames (every 40th frame across the full 24,524-frame video) was manually labeled for ground-truth drone presence, **with no reference to either model's output**. Each model was then checked separately against this same shared ground truth. This separation matters: sampling frames based on what a model did or didn't detect biases recall toward or against that same model — the two measurements had to be kept independent to be valid.

## Results

### Precision by confidence (from model-proposed candidates)

| Confidence | coco precision | rgb_transfer precision |
|---|---|---|
| 0.1–0.2 | 4.3% | 18.4% |
| 0.2–0.3 | 14.9% | 25.0% |
| 0.3–0.4 | 14.6% | 43.8% |
| 0.4–0.5 | 15.6% | 64.6% |
| 0.5–0.6 | 22.9% | 65.9% |
| 0.6–0.7 | 54.2% | 78.7% |
| 0.7–0.8 | 85.1% | 89.8% |
| 0.8–0.9 | 100.0% | 100.0% |
| 0.9–1.0 | — (no detections above 0.9) | 100.0% |
| **Overall** | **39.7%** | **66.4%** |

`rgb_transfer`'s confidence score is meaningfully more informative below 0.6 — `coco`'s is close to flat/uninformative in that range (14–23%, barely better than noise).

### Recall and false-fire rate (independent ground-truth sample, n=257 positive / 340 negative frames)

| Model | Recall | 95% CI | False-fire rate (on empty frames) | 95% CI |
|---|---|---|---|---|
| coco | 77.82% (200/257) | ±5.1pp | 42.94% (146/340) | ±5.3pp |
| rgb_transfer | 77.43% (199/257) | ±5.1pp | 16.47% (56/340) | ±3.9pp |

**Recall difference is within noise — not a real effect.** **False-fire rate difference is real** — confidence intervals don't overlap.

*Caveat: both numbers reflect "any detection ≥0.1 confidence counts," not a specific deployment operating threshold. A follow-up recall-vs-threshold sweep is needed to pick an actual operating point — see Next Steps.*

## Interpretation

The two measurements tell a consistent, coherent story: `rgb_transfer` fires less often overall, and when it does fire it's right more often — but it isn't catching drones the `coco` baseline misses. The transfer-learning benefit is entirely on the false-positive side. Given ~43% of `coco`'s firings on genuinely empty frames were false alarms, at a low confidence floor it would be close to unusable as a standalone alert source; `rgb_transfer` is meaningfully more deployable as-is, but still not clean (1 in 6 empty frames still trigger it).

**The shared ~22% recall gap is the headline problem, not a model-selection question.** Neither baseline addresses it. Plausible contributors, consistent with everything found this week: the confirmed sharpness/domain gap between training data and real footage, and target sizes sitting near or below the pixel floor where YOLO's stride-based detection struggles (per the earlier dataset audit, a meaningful fraction of real targets are in the <16px bucket).

## Tools built this week (for reproducibility)

- `audit_thermal_data.py` / `filter_dataset.py` — dataset contamination audit and filtering
- `train_thermal_yolo.py` — training script for both baselines
- `run_yolo_video_inference.py` — model inference over the ground-truth video
- `extract_detection_json_crops.py`, `label_review.py` — confidence-stratified manual review
- `sample_frames_for_recall.py`, `compute_recall_from_ground_truth.py` — model-independent recall measurement

## Next steps

- Finish labeling the remaining 16 unlabeled ground-truth frames (small, unlikely to shift conclusions).
- **Recall/false-fire vs. confidence threshold sweep** — current numbers use any detection ≥0.1; need this at candidate operating points (e.g. 0.3, 0.5, 0.7) to actually choose a deployment threshold.
- Break recall down by target size/distance if possible, to confirm the "small targets are the recall gap" hypothesis rather than assume it.
- `find_detection_gaps.py` — run separately to identify joint blind spots (frames both models miss), as an operational risk check distinct from the recall measurement above.
- AirSim synthetic data pipeline, to address the domain gap directly rather than just measuring it.
- DVWELCM (classical contrast-based method) evaluation for far/small targets, as a complement to YOLO rather than a replacement.