# Lab notebook

One file per working day: `YYYY-MM-DD.md`. Newest at the bottom of the index.
Each entry uses the same sections so days can be compared:

1. **Context:** where things stood at the start of the day
2. **Experiments:** what ran, on which data, with which settings, and where the outputs are
3. **Results:** numbers, always with n and 95% CI, on a named eval set
4. **Findings:** what we now believe, and how sure we are
5. **Decisions and exit criteria:** what we committed to, and when we stop
6. **Next steps**
7. **Artifacts:** code, data, and reports created or changed

## Standing rules (carry forward, edit when they change)

- **Detector eval set: the 613 GT frames of `thermal.mp4`, leak-free models only** (since 2026-09-29).
- **Evaluate only on data no model trained on.** Before any comparison, name the eval set and confirm it's clean for every model being compared. If it isn't clean for one of them, restrict to the subset that is.
- **Compare at matched confidence thresholds.** Log detections at conf ≥ 0.05 and score with `--conf_floor` (0.1 plus a sweep). Never compare runs whose inference used different floors.
- **Change one thing per run.** Bundled changes (leak fix + P2 head) can't be attributed.
- **No pairing or alignment claim from one visual match.** Verify across independent windows (`verify_stream_sync.py`) and multiple frames (`video_pairs calibrate`).
- **Set exit criteria before a run, not after seeing it.**

## Index

- [2026-09-28](2026-09-28.md): GAN review; leak found (45% of train is the eval video); leak-free dataset; synthetic copy-paste (option A); stream sync verified; segmented registration, 392 GAN pairs; first honest P2 numbers
- [2026-09-29](2026-09-29.md): eval set fixed to the 613 frames (leak-free models only); tail test dropped; no-P2 run skipped; GAN pairs re-split 280/112; go/no-go tool with a linear baseline bar (L1 ≤ 0.070); synthA v1 decomposed, not a win; GT misses tiny objects
