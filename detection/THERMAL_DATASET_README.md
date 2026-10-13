# Thermal UAV detection datasets (M0 and N2), 2026-10-13

Two YOLO datasets for the thermal (LWIR) drone detector, the training sets of the two candidate models:

| folder | model trained on it | train images | difference |
|---|---|---|---|
| `thermal_uav_M0/` | M0 seed 1 (`thermal_y26sp2_bosonM0u_s1_20261013.pt`) | 6,926 | reference set |
| `thermal_uav_N2/` | N2 seed 0 (`thermal_y26sp2_bosonN2_s0_20261013.pt`) | 6,109 | M0's set with **half of the synthetic pastes removed** (817 images, listed in `n2_removed_pastes.txt`); valid / test identical |

Both use the same validation and test splits. Each folder has its own `data.yaml` (paths relative to the yaml).
The archive stores N2's files as hard links to M0's, so it is barely larger than one set.

## Classes

YOLO txt labels, `class cx cy w h` (normalised). **0 = uav, 1 = bird** (the project-wide uav = 0 convention; older thermal sets used the reverse order).

## Splits

| split | images | empty (no object) | uav boxes | bird boxes | content |
|---|---|---|---|---|---|
| train (M0) | 6,926 | 538 | 6,500 | 633 | all sources below |
| train (N2) | 6,109 | 538 | 5,321 | 633 | as M0, half the pastes |
| valid | 245 | 50 | 195 | 0 | real frames: 167 from the 10-06 Boson sessions, 39 from 06-23, 39 from 07-15 (the last part of each session, after a time gap) |
| test | 95 | 8 | 57 | 30 | the public set's own test split (old, not representative; we do not use it to pick models) |

Model selection and the scores we report use separate, never-trained-on videos (see "Evaluation"), not these splits.

## Target sizes (uav long side, image pixels)

| split | p10 | median | p90 | < 8 px | 8–16 px | 16–32 px | ≥ 32 px |
|---|---|---|---|---|---|---|---|
| train (M0) | 6.0 | 19.0 | 61.0 | 1,285 | 1,617 | 1,438 | 2,160 |
| valid | 7.0 | 44.4 | 99.2 | 33 | 33 | 9 | 120 |

Image sizes: 640×512 (real camera frames), 512×512 (public set), ~466×262 (crops used as paste backgrounds). Train at imgsz 512.

## Where the training images come from (M0 train, 6,926)

| source | images | how it was made |
|---|---|---|
| Public thermal drone set (Roboflow "thermal-1") | 2,759 | Public Roboflow export. Its GAN-generated class was removed, and every image from our 2026-07-02 recording (our test video) was removed after a leak check (perceptual hash + filename families): 45 % of the original set was frames of that video. |
| Real frames, 06-23 sessions (old thermal camera, 640×512) | 1,227 → **409 unique ×3** | Our own flights. Boxes from (a) human review with our labelling tool (`gt_boxes.py`, model-free candidates, drag/click boxes) and (b) the day-camera RGB detector's boxes mapped into the thermal frame (time sync + registration) and confirmed by a human, plus boxes where the RGB and thermal detectors agree. Only the first 80 % of each session; the rest is validation. Oversampled ×3 (filename suffix `__os`). |
| Real frames, 07-15 session (old camera) | 597 → **199 unique ×3** | Same procedure; dark drones over terrain. Frames before 9,475 only. |
| Real frames, 2026-10-06 sessions (**Boson, the deployed camera**) | 710 | 13 sessions recorded with the day camera. Labels where the day RGB detector and the thermal detector both detect the drone at the same place after sync/registration (`pos`), sampled every few frames; empty frames where both are silent; a human QA pass; plus 137 boxes / 19 empties reviewed by a human (thermal-only and disagreement cases). Sessions 08_58_15 and 10_15_59 are held out (never in any split), 10_05_11 excluded (misregistered). |
| Synthetic pastes on real old-camera backgrounds | 1,243 (N2: 612) | Real thermal crops from the 06-23 / 07-15 training sessions; the real drone is detected and inpainted (crops where that would leave a smear on texture are dropped); 1–2 real drone sprites (3–24 px, cut from public-set boxes) pasted in contrast space on smooth sky only, with optics blur and matched noise, then re-encoded. Box = the pasted sprite. |
| Synthetic pastes on real Boson backgrounds | 390 (N2: 204) | Same, on Boson frames where neither detector fired, with sprites cut from Boson drones. |

Empty images (538, ~8 %) come from the real-frame sources (frames confirmed drone-free).

## Why N2

The deployed model fired on far too many empty frames (tiny bright dots in the sky). Pasting small bright drones on clean sky teaches "small bright dot on sky = drone". N2 keeps half of those pastes: on the evaluation videos it has the best recall at a matched false-alarm rate of all single models (see the models' sheet).

## Evaluation (not part of these folders)

- Deployed camera: two held-out Boson sessions with human labels (08_58_15: 22 drone / 88 empty frames; 10_15_59: 122 / 44).
- Old camera, large: the Arsuf 2026-10-05 bench recording (= second half of the 07-02 test video), 2,459 drone / 727 empty human-labelled frames, in the interceptor pipeline_eval format.
- Old-camera test videos 07-02 (320 / 246), 07-30 (137 / 296), 07-15 tail (146 / 111).
Metric: recall where the box must land on the drone, at a matched false-alarm rate (share of empty frames with any box), mean over 1–25 %.

Built with the day2thermal repo (`detection/src/data/`: build_dataset.py, paste_on_backgrounds.py, export_boson_yolo.py, export_reviewed.py, swap_uav_class.py, build_hardneg_sets.py); details in `notebook/2026-09-28 … 2026-10-13`.
