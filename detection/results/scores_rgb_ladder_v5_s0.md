# RGB model comparison

### v5_test (1476 images)

| | fw4 | v5_s0 |
|---|---|---|
| mAP50 | 0.932 | 0.888 |
| mAP50-95 | 0.723 | 0.664 |
| uav AP50 | 0.989 | 0.972 |
| uav AP50-95 | 0.832 | 0.793 |
| airplane AP50 | 0.839 | 0.745 |
| airplane AP50-95 | 0.700 | 0.592 |
| bird AP50 | 0.969 | 0.947 |
| bird AP50-95 | 0.638 | 0.607 |
| uav GT | 1082 | 1082 |
| uav recall@0.15 | 98.6 | 96.2 |
| uav found any class@0.15 | 98.6 | 97.7 |
| background->uav boxes@0.15 | 20 | 57 |
| no-uav images with a uav box@0.15 (%) | 2.4 | 8.2 |

### unseen (234 images)

| | fw4 | v5_s0 |
|---|---|---|
| mAP50 | 0.928 | 0.907 |
| mAP50-95 | 0.655 | 0.652 |
| uav AP50 | 0.978 | 0.969 |
| uav AP50-95 | 0.770 | 0.775 |
| airplane AP50 | 0.853 | 0.817 |
| airplane AP50-95 | 0.608 | 0.597 |
| bird AP50 | 0.954 | 0.935 |
| bird AP50-95 | 0.588 | 0.584 |
| uav GT | 179 | 179 |
| uav recall@0.15 | 98.3 | 95.5 |
| uav found any class@0.15 | 98.3 | 96.1 |
| background->uav boxes@0.15 | 6 | 15 |
| no-uav images with a uav box@0.15 (%) | 0.000 | 8.3 |

### boson (100 images)

| | fw4 | v5_s0 |
|---|---|---|
| mAP50 | 0.989 | 0.990 |
| mAP50-95 | 0.822 | 0.807 |
| uav AP50 | 0.989 | 0.990 |
| uav AP50-95 | 0.822 | 0.807 |
| uav GT | 100 | 100 |
| uav recall@0.15 | 94 | 99 |
| uav found any class@0.15 | 94 | 99 |
| background->uav boxes@0.15 | 1 | 0.000 |
| no-uav images with a uav box@0.15 (%) | 0.000 | 0.000 |

### flight1014 (174 images)

| | fw4 | v5_s0 |
|---|---|---|
| mAP50 | 0.188 | 0.423 |
| mAP50-95 | 0.155 | 0.334 |
| uav AP50 | 0.564 | 0.633 |
| uav AP50-95 | 0.466 | 0.507 |
| airplane AP50 | 0.000 | 0.627 |
| airplane AP50-95 | 0.000 | 0.486 |
| bird AP50 | 0.000 | 0.009 |
| bird AP50-95 | 0.000 | 0.007 |
| uav GT | 69 | 69 |
| uav recall@0.15 | 69.6 | 66.7 |
| uav found any class@0.15 | 69.6 | 72.5 |
| background->uav boxes@0.15 | 25 | 12 |
| no-uav images with a uav box@0.15 (%) | 42.6 | 28.7 |

### oi_sky (129 images)

| | fw4 | v5_s0 |
|---|---|---|
| mAP50 | 0.093 | 0.141 |
| mAP50-95 | 0.083 | 0.127 |
| uav AP50 | 0.009 | 0.012 |
| uav AP50-95 | 0.008 | 0.011 |
| airplane AP50 | 0.192 | 0.212 |
| airplane AP50-95 | 0.177 | 0.191 |
| bird AP50 | 0.080 | 0.199 |
| bird AP50-95 | 0.063 | 0.178 |
| uav GT | 2 | 2 |
| uav recall@0.15 | 0.000 | 0.000 |
| uav found any class@0.15 | 100 | 100 |
| background->uav boxes@0.15 | 30 | 15 |
| no-uav images with a uav box@0.15 (%) | 14.2 | 10.2 |

### sim_tail (830 images)

| | fw4 | v5_s0 |
|---|---|---|
| mAP50 | 0.831 | 0.887 |
| mAP50-95 | 0.583 | 0.638 |
| uav AP50 | 0.960 | 0.954 |
| uav AP50-95 | 0.653 | 0.629 |
| airplane AP50 | 0.592 | 0.725 |
| airplane AP50-95 | 0.405 | 0.584 |
| bird AP50 | 0.939 | 0.982 |
| bird AP50-95 | 0.690 | 0.701 |
| uav GT | 631 | 631 |
| uav recall@0.15 | 97.1 | 95.4 |
| uav found any class@0.15 | 97.1 | 95.4 |
| background->uav boxes@0.15 | 45 | 55 |
| no-uav images with a uav box@0.15 (%) | 14.5 | 15.9 |
