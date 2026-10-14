# RGB dataset audit: `Fixed_wing_v3`

Classes ['bird', 'airplane', 'uav']. Sizes are sqrt(w·h) in model-input px at 640. Script: `detection/src/data/rgb_audit.py`.

## Counts

| | train | valid | test |
|---|---|---|---|
| images | 20,281 | 2,972 | 1,516 |
| empty images | 2,391 (11.8%) | 234 (7.9%) | 100 (6.6%) |
| bird boxes | 10,620 | 1,303 | 738 |
| airplane boxes | 1,226 | 234 | 174 |
| uav boxes | 10,595 | 2,346 | 1,192 |
| sources | 7,179 | 1,817 | 1,006 |
| top-10 sources' share | 20.3% | 32.3% | 30.4% |

## bird target size

| split | n | 0-8 | 8-16 | 16-32 | 32-64 | 64-128 | >=128 |
|---|---|---|---|---|---|---|---|
| train | 10,620 | 6.5% | 24.8% | 32.4% | 21.0% | 6.4% | 8.8% |
| valid | 1,303 | 0.0% | 15.8% | 31.2% | 26.6% | 7.4% | 18.9% |
| test | 738 | 1.1% | 13.8% | 37.7% | 23.8% | 6.0% | 17.6% |

## airplane target size

| split | n | 0-8 | 8-16 | 16-32 | 32-64 | 64-128 | >=128 |
|---|---|---|---|---|---|---|---|
| train | 1,226 | 0.7% | 2.8% | 8.0% | 13.1% | 19.9% | 55.5% |
| valid | 234 | 0.0% | 1.3% | 7.7% | 17.9% | 15.8% | 57.3% |
| test | 174 | 0.0% | 2.9% | 7.5% | 12.1% | 19.5% | 58.0% |

## uav target size

| split | n | 0-8 | 8-16 | 16-32 | 32-64 | 64-128 | >=128 |
|---|---|---|---|---|---|---|---|
| train | 10,595 | 0.3% | 4.9% | 6.7% | 16.5% | 26.7% | 44.9% |
| valid | 2,346 | 0.1% | 0.6% | 7.3% | 14.7% | 25.7% | 51.6% |
| test | 1,192 | 0.1% | 0.6% | 6.2% | 14.9% | 25.5% | 52.7% |

## uav: background and contrast

Ring = 0.75 × box size around the box. smooth = ring gray std < 8; low contrast = |box − ring| < 12 gray levels.

| uav boxes | train | valid | test | hole? |
|---|---|---|---|---|
| small <16 px, smooth bg (sky) | 99 (0.9%) | 5 (0.2%) | 7 (0.6%) | train < 100; valid under-represented |
| small <16 px, textured bg (cloud/terrain) | 455 (4.3%) | 11 (0.5%) | 1 (0.1%) | valid under-represented; test under-represented |
| small <16 px, low contrast | 117 (1.1%) | 9 (0.4%) | 7 (0.6%) |  |
| small, low contrast, textured | 40 (0.4%) | 5 (0.2%) | 1 (0.1%) | train < 100; test under-represented |
| 16-64 px, textured bg | 1,441 (13.6%) | 326 (13.9%) | 141 (11.8%) |  |
| >=64 px (close range) | 7,583 (71.6%) | 1,814 (77.3%) | 932 (78.2%) |  |
| cut by image edge | 1,628 (15.4%) | 431 (18.4%) | 224 (18.8%) |  |

## Scenes (images with a uav box)

| | train | valid | test |
|---|---|---|---|
| light: dark | 5.0% | 5.4% | 4.2% |
| light: normal | 88.6% | 88.2% | 89.9% |
| light: bright | 6.3% | 6.3% | 5.9% |
| comp: mostly smooth | 30.0% | 30.9% | 33.3% |
| comp: mixed | 43.3% | 40.2% | 40.3% |
| comp: cluttered | 26.7% | 28.9% | 26.4% |

## Leakage

- **valid**: 610 sources also in train (1,741 of 2,972 valid images, 58.6%); 410 valid images (13.8%) have a near-duplicate (dHash ≤ 4) in train.
- **test**: 357 sources also in train (861 of 1,516 test images, 56.8%); 185 test images (12.2%) have a near-duplicate (dHash ≤ 4) in train.

## Holes flagged automatically

- uav, small <16 px, smooth bg (sky): train < 100, valid under-represented
- uav, small <16 px, textured bg (cloud/terrain): valid under-represented, test under-represented
- uav, small, low contrast, textured: train < 100, test under-represented
- class **airplane**: only 1,226 train boxes
