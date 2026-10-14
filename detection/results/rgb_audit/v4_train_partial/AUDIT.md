# RGB dataset audit: `Fixed_wing_v4_20261011_191146`

Classes ['uav', 'airplane', 'bird']. Sizes are sqrt(w·h) in model-input px at 640. Script: `detection/src/data/rgb_audit.py`.

## Counts

| | train |
|---|---|
| images | 5,037 |
| empty images | 542 (10.8%) |
| uav boxes | 3,859 |
| airplane boxes | 75 |
| bird boxes | 1,828 |
| sources | 1,648 |
| top-10 sources' share | 47.8% |

## uav target size

| split | n | 0-8 | 8-16 | 16-32 | 32-64 | 64-128 | >=128 |
|---|---|---|---|---|---|---|---|
| train | 3,859 | 0.4% | 8.1% | 11.0% | 22.8% | 25.9% | 31.8% |

## airplane target size

| split | n | 0-8 | 8-16 | 16-32 | 32-64 | 64-128 | >=128 |
|---|---|---|---|---|---|---|---|
| train | 75 | 0.0% | 2.7% | 40.0% | 18.7% | 1.3% | 37.3% |

## bird target size

| split | n | 0-8 | 8-16 | 16-32 | 32-64 | 64-128 | >=128 |
|---|---|---|---|---|---|---|---|
| train | 1,828 | 1.7% | 14.8% | 23.9% | 19.1% | 3.8% | 36.7% |

## uav: background and contrast

Ring = 0.75 × box size around the box. smooth = ring gray std < 8; low contrast = |box − ring| < 12 gray levels.

| uav boxes | train | hole? |
|---|---|---|
| small <16 px, smooth bg (sky) | 66 (1.7%) | train < 100 |
| small <16 px, textured bg (cloud/terrain) | 263 (6.8%) |  |
| small <16 px, low contrast | 64 (1.7%) | train < 100 |
| small, low contrast, textured | 23 (0.6%) | train < 100 |
| 16-64 px, textured bg | 778 (20.2%) |  |
| >=64 px (close range) | 2,225 (57.7%) |  |
| cut by image edge | 258 (6.7%) |  |

## Scenes (images with a uav box)

| | train |
|---|---|
| light: dark | 5.6% |
| light: normal | 90.4% |
| light: bright | 3.9% |
| comp: mostly smooth | 20.9% |
| comp: mixed | 50.8% |
| comp: cluttered | 28.3% |

## Leakage


## Holes flagged automatically

- uav, small <16 px, smooth bg (sky): train < 100
- uav, small <16 px, low contrast: train < 100
- uav, small, low contrast, textured: train < 100
- class **airplane**: only 75 train boxes
- class **bird**: only 1,828 train boxes
