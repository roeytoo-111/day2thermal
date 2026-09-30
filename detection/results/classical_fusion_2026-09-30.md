### 07-02 s1 (key 0708): 320 positive / 246 negative frames

| detector | recall @FF≤5% | recall @FF≤10% | recall @FF≤25% |
|---|---|---|---|
| classical only | 38.4% (rank≥0.95) | 38.4% (rank≥0.95) | 54.7% (rank≥0.9) |
| p2_noleak | 22.8% [19%–28%] (conf≥0.5) | 32.2% [27%–37%] (conf≥0.3) | 39.1% [34%–45%] (conf≥0.15) |
| p2_noleak + classical | 39.7% [34%–45%] (conf≥0.8, rank≥0.95) | 48.1% [43%–54%] (conf≥0.4, rank≥0.95) | 59.7% [54%–65%] (conf≥0.25, rank≥0.9) |

### 07-30 (test 2): 137 positive / 279 negative frames

| detector | recall @FF≤5% | recall @FF≤10% | recall @FF≤25% |
|---|---|---|---|
| classical only | 5.8% (rank≥0.99) | 5.8% (rank≥0.98) | 6.6% (rank≥0.95) |
| p2_noleak | 19.0% [13%–26%] (conf≥0.15) | 21.9% [16%–30%] (conf≥0.1) | 26.3% [20%–34%] (conf≥0.05) |
| p2_noleak + classical | 19.0% [13%–26%] (conf≥0.15, rank≥0.999) | 22.6% [16%–30%] (conf≥0.1, rank≥0.99) | 26.3% [20%–34%] (conf≥0.05, rank≥0.95) |
| p2_sessions_aug | 21.9% [16%–30%] (conf≥0.25) | 21.9% [16%–30%] (conf≥0.25) | 21.9% [16%–30%] (conf≥0.25) |
| p2_sessions_aug + classical | 22.6% [16%–30%] (conf≥0.25, rank≥0.995) | 23.4% [17%–31%] (conf≥0.25, rank≥0.99) | 24.1% [18%–32%] (conf≥0.25, rank≥0.95) |

