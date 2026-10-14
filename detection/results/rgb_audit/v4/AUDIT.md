# RGB dataset audit: `Fixed_wing_v4_20261011_191146`

Classes ['uav', 'airplane', 'bird']. Sizes are sqrt(w·h) in model-input px at 640. Script: `detection/src/data/rgb_audit.py`.

## Counts

| | train | valid | test | test_boson_heldout |
|---|---|---|---|---|
| images | 23,426 | 3,100 | 1,516 | 100 |
| empty images | 4,406 (18.8%) | 338 (10.9%) | 99 (6.5%) | 0 (0.0%) |
| uav boxes | 17,110 | 2,314 | 1,168 | 100 |
| airplane boxes | 1,569 | 305 | 203 | 0 |
| bird boxes | 5,025 | 1,309 | 774 | 0 |
| sources | 7,197 | 1,818 | 1,006 | 2 |
| top-10 sources' share | 22.7% | 34.6% | 30.4% | 100.0% |

## uav target size

| split | n | 0-8 | 8-16 | 16-32 | 32-64 | 64-128 | >=128 |
|---|---|---|---|---|---|---|---|
| train | 17,110 | 5.0% | 16.8% | 15.5% | 16.0% | 18.7% | 28.0% |
| valid | 2,314 | 0.6% | 1.5% | 7.7% | 14.8% | 24.5% | 50.9% |
| test | 1,168 | 0.1% | 0.6% | 6.3% | 15.2% | 25.0% | 52.8% |
| test_boson_heldout | 100 | 4.0% | 54.0% | 42.0% | 0.0% | 0.0% | 0.0% |

## airplane target size

| split | n | 0-8 | 8-16 | 16-32 | 32-64 | 64-128 | >=128 |
|---|---|---|---|---|---|---|---|
| train | 1,569 | 0.5% | 2.3% | 8.9% | 12.7% | 17.4% | 58.2% |
| valid | 305 | 0.0% | 1.0% | 5.9% | 14.8% | 14.1% | 64.3% |
| test | 203 | 0.0% | 2.5% | 6.4% | 10.3% | 17.2% | 63.5% |
| test_boson_heldout | 0 | – | – | – | – | – | – |

## bird target size

| split | n | 0-8 | 8-16 | 16-32 | 32-64 | 64-128 | >=128 |
|---|---|---|---|---|---|---|---|
| train | 5,025 | 0.7% | 10.2% | 35.8% | 28.3% | 7.2% | 17.7% |
| valid | 1,309 | 0.0% | 15.8% | 30.8% | 27.4% | 7.3% | 18.7% |
| test | 774 | 1.0% | 13.4% | 37.1% | 24.8% | 6.6% | 17.1% |
| test_boson_heldout | 0 | – | – | – | – | – | – |

## uav: background and contrast

Ring = 0.75 × box size around the box. smooth = ring gray std < 8; low contrast = |box − ring| < 12 gray levels.

| uav boxes | train | valid | test | test_boson_heldout | hole? |
|---|---|---|---|---|---|
| small <16 px, smooth bg (sky) | 2,607 (15.2%) | 34 (1.5%) | 7 (0.6%) | 28 (28.0%) | valid under-represented; test under-represented |
| small <16 px, textured bg (cloud/terrain) | 1,111 (6.5%) | 13 (0.6%) | 1 (0.1%) | 30 (30.0%) | valid under-represented; test under-represented |
| small <16 px, low contrast | 2,293 (13.4%) | 37 (1.6%) | 7 (0.6%) | 7 (7.0%) | valid under-represented; test under-represented |
| small, low contrast, textured | 453 (2.6%) | 7 (0.3%) | 1 (0.1%) | 6 (6.0%) | valid under-represented; test under-represented |
| 16-64 px, textured bg | 2,603 (15.2%) | 330 (14.3%) | 142 (12.2%) | 23 (23.0%) |  |
| >=64 px (close range) | 8,003 (46.8%) | 1,746 (75.5%) | 909 (77.8%) | 0 (0.0%) | test_boson_heldout under-represented |
| cut by image edge | 1,727 (10.1%) | 380 (16.4%) | 204 (17.5%) | 0 (0.0%) | test_boson_heldout under-represented |

## Scenes (images with a uav box)

| | train | valid | test | test_boson_heldout |
|---|---|---|---|---|
| light: dark | 3.4% | 5.5% | 4.3% | 0.0% |
| light: normal | 91.7% | 88.6% | 89.8% | 50.0% |
| light: bright | 4.9% | 5.8% | 5.8% | 50.0% |
| comp: mostly smooth | 37.7% | 29.2% | 31.9% | 4.0% |
| comp: mixed | 36.9% | 41.4% | 41.0% | 93.0% |
| comp: cluttered | 25.4% | 29.4% | 27.1% | 3.0% |

## Leakage

- **valid**: 611 sources also in train (1,869 of 3,100 valid images, 60.3%); 415 valid images (13.4%) have a near-duplicate (dHash ≤ 4) in train.
- **test**: 357 sources also in train (861 of 1,516 test images, 56.8%); 185 test images (12.2%) have a near-duplicate (dHash ≤ 4) in train.
- **test_boson_heldout**: 0 sources also in train (0 of 100 test_boson_heldout images, 0.0%); 0 test_boson_heldout images (0.0%) have a near-duplicate (dHash ≤ 4) in train.

## Holes flagged automatically

- uav, small <16 px, smooth bg (sky): valid under-represented, test under-represented
- uav, small <16 px, textured bg (cloud/terrain): valid under-represented, test under-represented
- uav, small <16 px, low contrast: valid under-represented, test under-represented
- uav, small, low contrast, textured: valid under-represented, test under-represented
- uav, >=64 px (close range): test_boson_heldout under-represented
- class **airplane**: only 1,569 train boxes
- test_boson_heldout: dusk / low light 0.0% of images
- test_boson_heldout: only 0.0% empty (negative) images
