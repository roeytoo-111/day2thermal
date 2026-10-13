# 2026-10-06 08_58_15 (Boson, deployed camera): ground-truth review

Videos (cut by the interceptor session, original timestamps kept, so labels also apply to the `from_286` cut):
`2026-10-06T08_58_15_from_281.ts` (4K day, 190 s, 5,692 frames) and `2026-10-06T08_58_15_thermal_from_281.ts`
(640x512, 69 s, 2,067 frames; the thermal recording ends earlier). Review at 5 Hz: 345 thermal and 949 day frames.

This session is HELD OUT: no model was ever trained on it (fine for testing). Pre-fill: thermal = ensemble of
N2 s0 + M0 s1 + R0 s0; day = the new RGB model (fw4). Sync thermal_t = day_t - 0.117 s (7/8 windows), day->thermal
mapping = the calibrated Boson registration (checked: median 2.6 px on 103 co-detections).

**Green overlay**: 39 thermal review frames (about 33-60 s) have a green square burned into the image, and the
thermal models fire on it. 38 such pre-filled boxes were removed; decide those frames yourself (drone visible or not).

Run from `day2thermal/detection/`. Both tools save as you go; q quits, rerun to resume.

## Thermal (k or Enter keeps the pre-filled box)
    ../.venv/bin/python src/eval/gt_boxes.py review --video ../data/boson_work/cut0858/2026-10-06T08_58_15_thermal_from_281.ts --manifest ../data/boson_work/cut0858/gt_thermal/manifest.csv --proposals ../data/boson_work/cut0858/gt_thermal/proposals.json --out ../data/boson_work/cut0858/gt_thermal/boxes.csv --verify --only all --gate

## Day (Space or k keeps; drag on either panel to draw)
    ../.venv/bin/python src/eval/review_day_gt.py --video ../data/boson_work/cut0858/2026-10-06T08_58_15_from_281.ts --frames ../data/boson_work/cut0858/gt_day/frames.csv --dets ../data/boson_work/cut0858/gt_day/candidates.json --out ../data/boson_work/cut0858/gt_day/boxes.csv --cache-dir ../data/arsuf_1005/.cache

## Export (pipeline_eval format)
    ../.venv/bin/python src/eval/export_pipeline_gt.py --out ../data/boson_work/cut0858/pipeline_gt/boson_0858 --thermal-boxes ../data/boson_work/cut0858/gt_thermal/boxes.csv --thermal-manifest ../data/boson_work/cut0858/gt_thermal/manifest.csv --day-boxes ../data/boson_work/cut0858/gt_day/boxes.csv --day-video 2026-10-06T08_58_15_from_281.ts --thermal-video 2026-10-06T08_58_15_thermal_from_281.ts --name "boson 2026-10-06 08_58_15 (from 281 s)" --videos-dir ../data/boson_work/cut0858 --cache-dir ../data/arsuf_1005/.cache
