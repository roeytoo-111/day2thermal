# RGB model comparison

### v5_test (1476 images)

| | fw4 | v5a_s0 |
|---|---|---|
| mAP50 | 0.932 | 0.878 |
| mAP50-95 | 0.723 | 0.653 |
| uav AP50 | 0.989 | 0.974 |
| uav AP50-95 | 0.832 | 0.785 |
| airplane AP50 | 0.839 | 0.718 |
| airplane AP50-95 | 0.700 | 0.570 |
| bird AP50 | 0.969 | 0.940 |
| bird AP50-95 | 0.638 | 0.604 |
| uav GT | 1082 | 1082 |
| uav recall@0.15 | 98.6 | 96.4 |
| uav found any class@0.15 | 98.6 | 97.8 |
| background->uav boxes@0.15 | 20 | 70 |
| no-uav images with a uav box@0.15 (%) | 2.4 | 10.4 |

### unseen (234 images)

| | fw4 | v5a_s0 |
|---|---|---|
| mAP50 | 0.928 | 0.919 |
| mAP50-95 | 0.655 | 0.660 |
| uav AP50 | 0.978 | 0.977 |
| uav AP50-95 | 0.770 | 0.762 |
| airplane AP50 | 0.853 | 0.840 |
| airplane AP50-95 | 0.608 | 0.640 |
| bird AP50 | 0.954 | 0.942 |
| bird AP50-95 | 0.588 | 0.580 |
| uav GT | 179 | 179 |
| uav recall@0.15 | 98.3 | 95.5 |
| uav found any class@0.15 | 98.3 | 96.1 |
| background->uav boxes@0.15 | 6 | 11 |
| no-uav images with a uav box@0.15 (%) | 0.000 | 3.3 |

### boson (100 images)

| | fw4 | v5a_s0 |
|---|---|---|
| mAP50 | 0.989 | 0.991 |
| mAP50-95 | 0.822 | 0.808 |
| uav AP50 | 0.989 | 0.991 |
| uav AP50-95 | 0.822 | 0.808 |
| uav GT | 100 | 100 |
| uav recall@0.15 | 94 | 98 |
| uav found any class@0.15 | 94 | 98 |
| background->uav boxes@0.15 | 1 | 3 |
| no-uav images with a uav box@0.15 (%) | 0.000 | 0.000 |

### flight1014 (174 images)

| | fw4 | v5a_s0 |
|---|---|---|
| mAP50 | 0.188 | 0.203 |
| mAP50-95 | 0.155 | 0.158 |
| uav AP50 | 0.564 | 0.564 |
| uav AP50-95 | 0.466 | 0.452 |
| airplane AP50 | 0.000 | 0.000 |
| airplane AP50-95 | 0.000 | 0.000 |
| bird AP50 | 0.000 | 0.045 |
| bird AP50-95 | 0.000 | 0.023 |
| uav GT | 69 | 69 |
| uav recall@0.15 | 69.6 | 66.7 |
| uav found any class@0.15 | 69.6 | 66.7 |
| background->uav boxes@0.15 | 25 | 14 |
| no-uav images with a uav box@0.15 (%) | 42.6 | 32.0 |

### oi_sky (129 images)

| | fw4 | v5a_s0 |
|---|---|---|
| mAP50 | 0.093 | 0.037 |
| mAP50-95 | 0.083 | 0.033 |
| uav AP50 | 0.009 | 0.008 |
| uav AP50-95 | 0.008 | 0.007 |
| airplane AP50 | 0.192 | 0.061 |
| airplane AP50-95 | 0.177 | 0.060 |
| bird AP50 | 0.080 | 0.042 |
| bird AP50-95 | 0.063 | 0.032 |
| uav GT | 2 | 2 |
| uav recall@0.15 | 0.000 | 0.000 |
| uav found any class@0.15 | 100 | 100 |
| background->uav boxes@0.15 | 30 | 32 |
| no-uav images with a uav box@0.15 (%) | 14.2 | 20.5 |
