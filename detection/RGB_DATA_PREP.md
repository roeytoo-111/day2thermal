# RGB data prep: label fixes, re-split, new flight data

Tools for the next RGB run (after `yolo26s_fw4_20261011`). Class order everywhere: **0 uav, 1 airplane, 2 bird**.
Fixed_wing_v3 on disk is (bird, airplane, uav); the tools read each folder's `data.yaml` and remap.
Run everything from `detection/` with `../.venv/bin/python`. No tool uses the Enter key.

## 1. Fix labels (human): `review_label_fixes.py`

The proposals come from fw4 vs the labels (`src/data/propose_rgb_label_fixes.py`), mostly birds in partly boxed flocks.

| queue | proposals | images |
|---|---|---|
| `../data/rgb_fixes/v3`: v3 valid + test | 699 (538 missing birds) | 210 |
| `../data/rgb_fixes/v3_train`: v3 train | see its `queue.csv` | |

```
../.venv/bin/python src/eval/review_label_fixes.py --fixes ../data/rgb_fixes/v3                 # do this one first
../.venv/bin/python src/eval/review_label_fixes.py --fixes ../data/rgb_fixes/v3_train --kind wrong_class
../.venv/bin/python src/eval/review_label_fixes.py --fixes ../data/rgb_fixes/v3_train --kind missing
```

Keys:

| key | action |
|---|---|
| `u` / `p` / `b` | it is a uav / airplane / bird: adds the box, or changes the class |
| `x` | nothing there: rejects the proposal, or deletes the label |
| `k` | keep as is |
| `A` | accept every remaining "missing" box in this image as the model proposed (flocks) |
| `o` | drop the whole image |
| drag | redraw the box, then `u`/`p`/`b` |
| space | skip |
| `z` | back |
| `q` | quit (resumable) |

Rules of thumb:
- **Dense flock where you can't box every bird: press `o`.** A half-labelled flock teaches the model to ignore birds.
- **`o` is also for out-of-scope uav images**: drone on a stand or runway, in a hand, or a product shot.
- `unsupported` rows are labels the model sees nothing at. Delete them only if nothing is there.

When done, write the corrected labels:
```
../.venv/bin/python src/eval/review_label_fixes.py --fixes ../data/rgb_fixes/v3 --finalize --labels-out ../data/rgb_fixes/labels
../.venv/bin/python src/eval/review_label_fixes.py --fixes ../data/rgb_fixes/v3_train --finalize --labels-out ../data/rgb_fixes/labels
```
`../data/rgb_fixes/labels/` and `../data/rgb_fixes/excluded.txt` are keyed by file name, so they apply to the same images inside Fixed_wing_v4.

## 2. New flight data (today): `scripts/new_flight_rgb.sh`

```
scripts/new_flight_rgb.sh prep   "<raw folder with the 2026-10-14T... session folders>" ../data/flight_20261014   # local
scripts/new_flight_rgb.sh detect ../data/flight_20261014     # GPU: Claude runs it on the VM
scripts/new_flight_rgb.sh labels ../data/flight_20261014     # local, minutes
scripts/new_flight_rgb.sh review ../data/flight_20261014     # YOU: d / f / b per tile
../.venv/bin/python src/eval/review_rgb_tiles.py --addon ../data/flight_20261014/rgb_addon --finalize
```

- **Time windows.** Optional `../data/flight_20261014/windows.txt` keeps only the airborne part of each session, one line per session: `1014_09_12_30 --t-min 120 --t-max 900`. The session id is `MMDD_HH_MM_SS`.
- **Two drones in the air** (interceptor + target): run `labels` with `MULTI_DRONE=1`. Otherwise every second drone becomes a "false alarm" negative.
- **What the review queue holds:**
  - Day detections the thermal model did not confirm.
  - Thermal tracks the day model missed (`miss*` tiles, from `mine_day_misses.py`). These are the weakness cases: dark drones on overcast and drones low over terrain.
- **How to decide a tile:**
  - `d` = drone visible in the day tile, even faintly.
  - `f` = nothing there. The tile becomes a negative.
  - `b` = bird.
- Auto-labelled tiles (day and thermal agree on the drone) need no review.

## 3. Build the dataset: `resplit_by_source.py`

```
../.venv/bin/python src/data/resplit_by_source.py \
    --data ../../ANN-Detection/data/datasets/Fixed_wing_v4_20261011_191146 ../data/flight_20261014/rgb_addon \
    --labels-override ../data/rgb_fixes/labels --exclude ../data/rgb_fixes/excluded.txt \
    --force "test:^session(08_58_15|10_15_59)$" --out ../../ANN-Detection/data/datasets/Fixed_wing_v5
```

What it does:
- Pools all splits of all inputs.
- Groups images by source video / photo series and by near-duplicate picture.
- Assigns whole groups to train / valid / test, balancing small drones, airplanes, birds and empty images.
- Keeps one picture per near-duplicate cluster in valid/test.
- Writes hard links and a `data.yaml` with uav = 0.

Use `--dry-run` to see `SPLIT.md` first.

On v3 the result is about 18.5k / 3.3k / 2.2k images, with no source shared between splits. Before, 58% of valid images shared a source with train.

## 4. Audit: `rgb_audit.py`

```
../.venv/bin/python src/data/rgb_audit.py --data <dataset> --out results/rgb_audit/<name>
```

It reports, per split:
- target sizes in model-input pixels;
- uav background (smooth / textured) and contrast;
- lighting;
- source concentration;
- leakage.

`results/rgb_audit/v3/AUDIT.md` is the baseline.
