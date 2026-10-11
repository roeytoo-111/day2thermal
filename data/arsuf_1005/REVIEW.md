# Arsuf 2026-10-05 bench: ground-truth review

Two videos (hard links to the originals): `day_output.ts` (4K, first decodable frame t = 8.599589) and `th_output.ts`
(640x512, 25 fps). Labels are keyed by timestamp `t` (pts_time), as pipeline_eval's GROUND_TRUTH.md requires.
Review is at 5 Hz over the whole video: 3,396 thermal frames, 3,365 day frames.

Everything below is run from `day2thermal/detection/`. Both tools save as you go; quit with `q` and rerun the same
command to resume.

## 1. Thermal (existing gt_boxes tool, verify mode)

    ../.venv/bin/python src/eval/gt_boxes.py review --video ../data/arsuf_1005/th_output.ts \
        --manifest ../data/arsuf_1005/gt_thermal/manifest.csv --proposals ../data/arsuf_1005/gt_thermal/proposals.json \
        --out ../data/arsuf_1005/gt_thermal/boxes.csv --verify --only all --gate

- 1,856 frames are pre-filled with a drone box from M0's tracks: **Enter keeps it**, or pick/draw a better one.
- 1,540 frames are unlabelled: Enter = nothing in this frame; 1-6 / click / drag = drone; w = bird or other; s = unsure.
- Candidates: M0 boxes first, then the day RGB detections projected into thermal, then the model-free ones.
- `p` plays +-12 frames around the cursor (useful for tiny targets), `b` goes back.

## 2. Day (new tool, same keys idea)

    ../.venv/bin/python src/eval/review_day_gt.py --video ../data/arsuf_1005/day_output.ts \
        --frames ../data/arsuf_1005/gt_day/frames.csv --dets ../data/arsuf_1005/gt_day/candidates.json \
        --out ../data/arsuf_1005/gt_day/boxes.csv --cache-dir ../data/arsuf_1005/.cache

- 2,481 frames pre-filled from the production RGB model's tracks (all 12 longest tracks checked: the real aircraft).
- Enter keeps | 1-9 candidate | n nothing | u unsure | b bird | o airplane/other | x clear | a/d prev/next |
  +/- zoom | click the overview to move the zoom | drag in the zoom to draw a box (4K pixels).
- Candidates include the thermal detections projected into the day frame ("thermal-projected").
- The first frame takes a few seconds (decoding from the previous keyframe); the next ones are prefetched.

## 3. Export to the pipeline_eval format

    ../.venv/bin/python src/eval/export_pipeline_gt.py --out ../data/arsuf_1005/pipeline_gt/arsuf_1005 \
        --thermal-boxes ../data/arsuf_1005/gt_thermal/boxes.csv --thermal-manifest ../data/arsuf_1005/gt_thermal/manifest.csv \
        --day-boxes ../data/arsuf_1005/gt_day/boxes.csv --videos-dir ../data/arsuf_1005 --cache-dir ../data/arsuf_1005/.cache

Writes `gt.json` + `labels.csv` (only reviewed frames). Validated with the interceptor repo's own
`GroundTruth.load` and `VideoIndex`: every t is an exact frame timestamp of its video.
Then: `--gt-dir ../data/arsuf_1005/pipeline_gt --videos-dir <folder with the .ts files>` when starting the UI.

## Geometry (for reference)
- Sync: thermal_t = day_t - 0.094 s (7/8 independent windows agree within 4 ms; `reports/stream_sync_arsuf1005`).
- Day -> thermal: similarity, scale 0.1317 thermal px / day px, rotation 0.56 deg, from 958/1,470 co-detections,
  median residual 1.0 px (`day2th_affine.npy`). Same angular resolution as the 07-02 / 07-15 thermal camera (0.133-0.136),
  not the 10-06 Boson (0.301).
