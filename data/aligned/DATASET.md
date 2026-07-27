# Correlated RGB + thermal stream — aligned pair set

The one experiment recorded in this repository: a **simultaneously recorded
day-RGB and LWIR thermal stream of the same scene**, temporally synchronised
and spatially registered into supervised training pairs.

This is the supervision set for the paired modes (`pix2pix`, `two_stage`).
Everything else that existed alongside it in the original working directory —
unpaired collections, autolabelled detector sets, generated thermal output —
is out of scope here and is not recorded.

## What is in git

`manifest.csv` only. **No pixels.** One row per aligned pair:

| column | meaning |
|---|---|
| `frame` | zero-padded frame id, identical in both modalities |
| `split` | `train` or `val` |
| `width`, `height` | 460 × 445 for every pair (post-warp auto-crop) |
| `thermal_dtype` | `uint8` — this set is **8-bit, not radiometric** |
| `thermal_mean`, `thermal_std`, `thermal_min`, `thermal_max` | per-frame thermal intensity stats |
| `rgb_mean` | per-frame RGB intensity |
| `rgb_sha256`, `thermal_sha256` | SHA-256 of each PNG |

The checksums are the point: they let you verify that an out-of-band copy of
the imagery is byte-identical to the set these splits were built from, without
committing ~50 MB of PNGs that git would store badly and that nobody would
diff.

## Composition

| | pairs | frames | thermal mean ± σ |
|---|---|---|---|
| train | 116 | 000200 – 000315 | 86.34 ± 3.72 |
| val | 200 | 000000 – 000199 | 83.07 ± 5.48 |
| **total** | **316** | 000000 – 000315 | |

Both splits are **contiguous frame ranges**, which is deliberate — see below.

## How the pairs were produced

1. **Temporal sync** — `day2thermal.extract_frames` decimates both streams to
   a common rate and applies a fixed `--offset-ms` (thermal clock − RGB clock),
   calibrated once against a hot-object event visible in both recordings.
   `--drop-static-thermal` discards frames frozen by the microbolometer's
   NUC/FFC shutter.
2. **Spatial registration** — `day2thermal.register` warps RGB **into the
   thermal frame** through a single homography, so thermal pixels are never
   resampled. One homography for the whole recording is valid for a rigidly
   mounted pair viewing a distant scene (no parallax). `--auto-crop` then
   trims to the rectangle where warped RGB is fully valid, giving the uniform
   460 × 445.
3. **Split** — contiguous temporal chunks, never a random shuffle. Adjacent
   video frames are near-duplicates; shuffling them across the split leaks
   train content into val and inflates every metric you would then report.

## Known gaps — read before using this set

* **`registration.json` was not preserved.** `register.py` writes the
  homography, crop rectangle and source resolutions next to the split; that
  file is absent from the source directory, so the exact warp used for these
  316 pairs is not recoverable from what survives. The pairs themselves are
  consistent, but the calibration cannot be reproduced or audited, and it
  cannot be reapplied to new frames from the same rig. Re-derive and commit it
  before extending this set.
* **8-bit, not radiometric.** `thermal_dtype` is `uint8`, so there is no
  absolute-temperature calibration. A model trained on this set learns
  *relative* thermal appearance only — ThermalGAN's calibrated-°C property is
  not available here. It also means the source AGC state is unknown; if
  per-frame auto-gain was active during recording, the regression target is
  non-stationary across the set.
* **`thermal_min`/`thermal_max` saturate at 0/255 on essentially every frame**,
  consistent with a stretched 8-bit encode. Do not read the intensity stats as
  temperature.
* **Single recording, single scene.** 316 pairs from one continuous stream is
  narrow. ThermalGAN's own headline dataset lesson was that training on one
  narrow split generalised poorly and forced the collection of a much more
  varied one — the fix was scene diversity, not more frames of the same scene.
  Treat this set as a bring-up/plumbing validation, not as sufficient
  supervision for a deployable translator.
* **val (200) is larger than train (116)**, and val is the *earlier* part of
  the recording. That is an unusual ratio which follows from the chunked split
  parameters used at the time; it is recorded here as fact, not endorsed.

## Regenerating the manifest

```bash
python - <<'PY'
import csv, glob, hashlib, os, cv2
rows = []
for split in ("train", "val"):
    for p in sorted(glob.glob(f"{split}/rgb/*.png")):
        name = os.path.basename(p); tp = f"{split}/thermal/{name}"
        rgb, th = cv2.imread(p, -1), cv2.imread(tp, -1)
        h, w = th.shape[:2]
        rows.append(dict(frame=os.path.splitext(name)[0], split=split, width=w, height=h,
            thermal_dtype=str(th.dtype), thermal_mean=round(float(th.mean()), 3),
            thermal_std=round(float(th.std()), 3), thermal_min=int(th.min()),
            thermal_max=int(th.max()), rgb_mean=round(float(rgb.mean()), 3),
            rgb_sha256=hashlib.sha256(open(p, "rb").read()).hexdigest(),
            thermal_sha256=hashlib.sha256(open(tp, "rb").read()).hexdigest()))
w = csv.DictWriter(open("manifest.csv", "w", newline=""), fieldnames=list(rows[0]))
w.writeheader(); w.writerows(rows)
PY
```

Run from the directory holding `train/` and `val/`.
