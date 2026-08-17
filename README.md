# day2thermal

**Day-RGB → LWIR thermal image synthesis for UAV imagery.**

A supervised paired image-translation pipeline that learns to turn day colour
video into synthetic longwave-infrared (thermal) video, so that a large
RGB-only detection dataset can be converted into thermal training data without
re-flying every mission with a thermal core.

The method implements **ThermalGAN** (Kniaz et al., ECCV Workshops 2018) —
see [Paper implemented](#paper-implemented) — adapted for the practical case of
having a *synchronised day + thermal video pair* as supervision and a large
RGB-only dataset to translate.

```
paired videos ──► extract_frames ──► register ──► train ──► infer ──► eval
 (day+thermal)     (temporal sync)   (spatial      (GAN)    (translate    (vs real
                                      alignment,            the big RGB    thermal
                                      train/val)            dataset)       holdout)
```

---

## Why this is hard (and what the paper contributes)

RGB→thermal is **one-to-many**. The same scene in the visible band can
correspond to wildly different thermal images depending on ambient
temperature, solar loading and sky state — an airframe at 09:00 and at 15:00
looks identical in colour and very different in LWIR. Any deterministic
regression therefore averages over incompatible answers.

ThermalGAN quantifies the damage: off-the-shelf GAN frameworks (pix2pix+noise,
cLR-GAN, cVAE-GAN, BicycleGAN) recover absolute object temperature only to
roughly **5 °C**, while the fine local contrast that carries the useful signal
lives at the **1 °C** scale. That gap is the paper's real insight.

The fix is a **two-stage frequency decomposition**:

| Stage | Learns | Character |
|---|---|---|
| **G₁**: RGB + temperature → Ŝ | the low-frequency thermal *base* (average object temperatures) | genuinely multimodal — conditioned on a temperature vector so the ambiguity becomes a **controllable input** rather than noise |
| **G₂**: RGB + Ŝ → R̂ | the high-frequency *relative contrasts* | near-invariant to ambient temperature ⇒ unimodal ⇒ a plain pix2pix backbone suffices |

Final image is the sum **B̂ = Ŝ + R̂**. The two generator/discriminator pairs
train independently. Confining multimodality to stage 1 leaves stage 2 with a
low-variance target, which is what buys back the 1 °C-scale fidelity.

The practical payoff of temperature conditioning: sweeping the condition at
inference produces one synthetic thermal set per ambient temperature — free
weather augmentation for detector training.

`research/DayToThermal_Research_Report.md` carries the full literature review,
the LWIR physics the pipeline has to respect (cold-sky effect, hot-component
distribution by propulsion type, microbolometer NUC/AGC artefacts), and the
reasoning behind every design decision.

---

## Paper implemented

> V. V. Kniaz, V. A. Knyaz, J. Hladůvka, W. G. Kropatsch, V. Mizginov.
> **"ThermalGAN: Multimodal Color-to-Thermal Image Translation for Person
> Re-Identification in Multispectral Dataset."**
> *ECCV 2018 Workshops*, LNCS vol. 11134, pp. 606–624. Springer, 2019.
> doi:10.1007/978-3-030-11024-6_46

```bibtex
@inproceedings{kniaz2018thermalgan,
  title     = {ThermalGAN: Multimodal Color-to-Thermal Image Translation
               for Person Re-Identification in Multispectral Dataset},
  author    = {Kniaz, Vladimir V. and Knyaz, Vladimir A. and
               Hlad{\r{u}}vka, Ji{\v{r}}{\'i} and Kropatsch, Walter G. and
               Mizginov, Vladimir},
  booktitle = {Computer Vision -- ECCV 2018 Workshops},
  series    = {Lecture Notes in Computer Science},
  volume    = {11134},
  pages     = {606--624},
  publisher = {Springer},
  year      = {2019},
  doi       = {10.1007/978-3-030-11024-6_46}
}
```

The paper targets person re-identification; this repository keeps the
**translation framework** (the durable contribution) and drops the ReID
matching machinery (SCA body segmentation, temperature-histogram and MSER
signature distances), which is not relevant to air-to-air UAV imagery.

Components inherited from the surrounding literature:

* **pix2pix** — Isola et al., CVPR 2017 — cGAN + L1 + PatchGAN template
  (`--mode pix2pix`, the baseline).
* **pix2pixHD** — Wang et al., CVPR 2018 — multi-scale discriminator
  (`--num-scales-d 2`), which helps when targets are small relative to frame.
* **BicycleGAN** — Zhu et al., NeurIPS 2017 — the cVAE-GAN + KL machinery
  ThermalGAN adapts (`--use-vae`, optional/experimental here).
* **CycleGAN** — Zhu et al., ICCV 2017 — unpaired fallback (`--mode cyclegan`)
  for when registration fails.

### Correspondence to the paper, and honest deviations

| Paper | This code | Why |
|---|---|---|
| G₁ predicts thermal segmentation Ŝ supervised by **pixel-level class annotations** | G₁ predicts a **Gaussian low-pass of the real thermal frame** (`--lowpass-sigma`) | No masks exist for this data. The low-pass base preserves the same low/high-frequency decomposition with zero labelling. If masks are added later (sky / airframe / hot parts), swap `S` in `train.py::step_two_stage`. |
| Random latent z replaced by a **temperature vector Tᵢ** | Scalar temperature plane (`T` = frame mean, or real ambient °C via `--temps-csv`) | Same controllability from one number; sweeping it at inference reproduces the paper's multimodal probe sets. |
| Full BicycleGAN (cVAE-GAN + cLR-GAN + KL) inside G₁ | `--use-vae` implements the cVAE-GAN + KL part only | Explicit T-conditioning already carries the multimodality; the dual-cycle adds complexity for limited benefit here. |
| L_thermal: latent-temperature-domain loss via encoder E | `--lambda-temp` · \|mean(Ŝ) − T\| | Simple, stable proxy with the same intent (global temperature consistency). |
| G₂ = pix2pix on 4-channel input (RGB + Ŝ), B̂ = Ŝ + R̂, generators trained independently | Identical (Ŝ detached for stage 2) | Faithful. |
| U-Net + one extra conv & deconv layer | `extra_head` (default on) adds the extra conv after the last deconv | Faithful in spirit at equal resolution. |
| Vanilla GAN loss, Adam lr 2e-4, β=(0.5, 0.999) | Same optimiser defaults; **LSGAN** by default (`--gan-mode vanilla` restores the paper) | LSGAN trains more stably. |

**Deliberately not inherited:** the FLIR ONE PRO sensor behind ThermalWorld is
160×120 and person/street-centric, so its learned priors do not transfer — the
architecture is reused, not the weights.

---

## Install

```bash
pip install -r requirements.txt
```

`torch`, `numpy`, `opencv-python-headless`, `tqdm`. Everything except
`train.py` / `infer.py` runs without PyTorch.

## 1. Extract synchronised frame pairs

```bash
python -m day2thermal.extract_frames \
    --rgb flight01_day.mp4 --thermal flight01_thermal.mp4 \
    --out data/flight01/raw --fps 5 --offset-ms 120 --drop-static-thermal
```

* `--rgb` / `--thermal` take a video file **or** a directory of frames
  (add `--rgb-fps` / `--thermal-fps` for directories).
* `--offset-ms` = thermal clock − RGB clock for the same event. Calibrate once
  per rig: put a hot object (soldering iron, lighter, hand warmer) in view,
  find that event in both recordings, subtract the times.
* `--drop-static-thermal` skips frames frozen by the microbolometer's NUC/FFC
  shutter events, which otherwise poison pairs.
* **Radiometric note:** if you can export 16-bit radiometric frames (FLIR
  SEQ → TIFF via `flirpy`/`exiftool`, or the camera SDK), point `--thermal` at
  that directory. Ordinary 8-bit video works too — you just lose absolute
  temperature calibration, which is ThermalGAN's headline property.

## 2. Register RGB onto the thermal geometry

```bash
# export one raw pair, pick >=4 correspondences, save as
# {"rgb": [[x,y],...], "thermal": [[x,y],...]}
python -m day2thermal.register --raw data/flight01/raw --out tmp --dump-pair 0

python -m day2thermal.register --raw data/flight01/raw \
    --out data/flight01/aligned --mode manual --points calib/points.json \
    --auto-crop --val-frac 0.1
```

* RGB is warped **into the thermal frame**, never the reverse — radiometric
  thermal pixels are never resampled.
* One homography for a whole recording is correct for a rigid camera pair
  observing distant scenes, i.e. exactly the air-to-air / high-altitude case
  (scene ≈ at infinity ⇒ no parallax). For close-range ground scenes, crop
  away the near field.
* `--mode ecc` attempts automatic edge-based alignment. Cross-modal ECC is
  fragile — `manual` is the recommended default. `day2thermal/pick_points.py`
  is an interactive picker.
* Train/val split is by **contiguous temporal chunks** (`--chunk`), not random
  shuffle — adjacent video frames are near-duplicates and would otherwise leak
  across the split and inflate validation scores.
* Writes `registration.json` (homography, crop, ECC correlation, source sizes)
  next to the split — **keep this file**, it is the calibration record.

### 2b. Already-registered pair folders (skip `register`)

If a source dataset ships pixel-aligned pairs organised as
`<session>/RGB[-k]/<i>.png` + `<session>/NIR[-k]/<i>.png` (same footprint,
same instant), `prep_aligned` maps it straight onto the training layout:

```bash
python -m day2thermal.prep_aligned --root ir-rgb-dataset \
    --rgb-prefix RGB --ir-prefix NIR --out data/agri_rgb_nir/aligned \
    --val-sessions canola_06082019,drybean-30072020,Lentwheat_29082018,wheat_27072019
```

* Pairs by filename inside each `RGB*`/`NIR*` sub-folder pair; crops both to
  their common size when they differ by a pixel or two (skips anything
  larger — that is a mis-registration, not an off-by-one); writes IR as
  single-channel 8-bit and hard-links RGB.
* Splits **by session** (whole flights held out), never inside one —
  `--val-sessions` explicit, or `--val-frac` picks evenly spaced sessions.
* Writes `split.json` recording what went where.
* The IR channel is whatever the source recorded — for a NIR (reflected,
  0.7–1.5 µm) dataset the model learns RGB→NIR, **not** thermal; the
  temperature conditioning of `two_stage` has no physical meaning there, so
  use `--mode pix2pix`.

## 3. Train

Baseline — strong and fast, start here:

```bash
python -m day2thermal.train --data-root data/flight01/aligned \
    --out runs/p2p --mode pix2pix --crop-size 256 --batch-size 8
```

ThermalGAN-style two-stage with temperature conditioning:

```bash
python -m day2thermal.train --data-root data/flight01/aligned \
    --out runs/2stage --mode two_stage --crop-size 256 \
    --lowpass-sigma 8 --lambda-temp 10 [--temps-csv weather.csv]
```

Useful flags: `--thermal-mode abs16 --tmin -20 --tmax 80` for radiometric
16-bit data (the model then learns **calibrated temperatures**);
`--num-scales-d 2` for the pix2pixHD multi-scale discriminator (helps small
distant targets); `--load-size 0` to train on native-resolution random crops
instead of squash-resize; `--use-vae` for the BicycleGAN latent
(experimental); `--temps-csv` maps `filename,celsius` if ambient temperature
was logged per flight.

## 4. Generate the synthetic thermal dataset

```bash
python -m day2thermal.infer --ckpt runs/2stage/checkpoints/latest.pt \
    --input my_big_rgb_dataset/images --out my_synthetic_thermal \
    --temps 0,15,30 --save-16bit
```

With a two-stage checkpoint, `--temps` sweeps the ambient condition and writes
one thermal set per value (`T0C/`, `T15C/`, …).

## 5. Evaluate against real thermal

```bash
python -m day2thermal.eval --pred preds/ --gt data/flight01/aligned/val/thermal
```

Reports L1 / RMSE / PSNR / SSIM, plus °C errors in `abs16` mode. Calibrate
expectations against the paper's bar: generic GANs ≈ 5 °C, useful fine
contrast needs ≈ 1 °C.

**The decisive test is not any of these metrics.** It is detector-in-the-loop:
train the UAV detector on the synthetic thermal set, evaluate it on a **real**
thermal holdout. See section 6 of the research report.

`day2thermal/check_thermalness.py` catches the unpaired-training failure mode
where the model learns a greyscale filter instead of a modality change — see
[Stage 5](#stage-5--check-the-output).

---

# Reproducing the results

The five steps above are the general API. This section is the **specific run**
that produced the synthetic thermal output for this project — the actual
commands, the actual hyperparameters, and what each stage emitted.

## What was recorded

One flight, two cameras running simultaneously:

| Source | Content | Native size |
|---|---|---|
| `day.ts` (`arsuf_day.ts`) | day RGB stream, H.264 MPEG-TS, 30 fps, 711.5 s | 3840 × 2160 |
| `thermal.ts` (`arsuf_thermal.ts`) | LWIR stream, same scene, same time, H.264 MPEG-TS, 25 fps, 8-bit AGC | 640 × 512 |
| `csi_.mp4` | the large **RGB-only** dataset to be translated | 1920 × 1080 |

Byte-exact identification of the two source streams (SHA-256, frame counts,
PTS statistics, embedded MISB ST 0601 KLV telemetry) is recorded in
[`data/streams/STREAMS.md`](data/streams/STREAMS.md) /
[`manifest.json`](data/streams/manifest.json). The KLV timestamps also yield
the inter-stream offset (`--offset-ms ≈ +893`) — see Stage 1 below.

`day.ts` + `thermal.ts` are the *supervision*. `csi_.mp4` is the *payload* —
the dataset that has no thermal counterpart and is the whole reason the
translator exists.

## Stage 0 — environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is required for training. The recorded run took **~15.5 h wall**
for 97 epochs (checkpoint timestamps run 22 Jul 15:13 → 23 Jul 06:55).
Inference runs fine on CPU with `--cpu`, just slowly.

## Stage 1 — split the correlated streams into frame pairs

```bash
python -m day2thermal.extract_frames \
    --rgb day.ts --thermal thermal.ts \
    --out data/raw --drop-static-thermal \
    --fps <RATE> --offset-ms <THERMAL_MINUS_RGB>
```

→ `data/raw/rgb/` and `data/raw/thermal/`, **4897 frames each**, named
`000000.png` … `004896.png`. The shared filename *is* the pairing.

The exact `--fps` and `--offset-ms` used for this run were not recorded.
The streams' embedded KLV telemetry (one shared encoder clock) gives
`--offset-ms ≈ +893` (thermal started ~0.9 s before day) — derivation in
[`data/streams/STREAMS.md`](data/streams/STREAMS.md); confirm it with the
hot-object procedure in step 1 above before trusting it to the frame. 4897
pairs over 711.5 s is consistent with `--fps 7`. The offset is rig-specific
and must be recalibrated if the cameras are remounted anyway.

## Stage 2 — two branches from the same raw frames

The raw pairs feed two independent experiments. **The shipped results came
from branch B.** Both are documented because the distinction matters when
reading the output.

### Branch A — paired / registered (the correlated-stream experiment)

```bash
python -m day2thermal.register --raw data/raw --out tmp --dump-pair 0
# pick >=4 correspondences -> calib/points.json
python -m day2thermal.register --raw data/raw --out data/aligned \
    --mode manual --points calib/points.json --auto-crop --val-frac 0.1
```

→ `data/aligned/{train,val}/{rgb,thermal}/` — **316 pairs at 460 × 445**
(116 train / 200 val). This is the set catalogued in
[`data/aligned/manifest.csv`](data/aligned/manifest.csv).

This branch is what unlocks `--mode pix2pix` and `--mode two_stage`, i.e. the
actual ThermalGAN method. **No training run against it survives**, and its
`registration.json` was not preserved — see
[`data/aligned/DATASET.md`](data/aligned/DATASET.md).

### Branch B — unpaired (what actually generated the output)

Registration throws away all but 316 of the 4897 pairs (the auto-crop keeps
only the region where both fields of view overlap). The unpaired route keeps
nearly all of them, at the cost of pixel-accurate supervision:

```bash
python -m day2thermal.prep_unpaired --raw data/raw --out data/unpaired \
    --width 640 --skip-first 900
```

→ `data/unpaired/rgb/` at **640 × 360** (downscaled once, so the loader stops
decoding 4K PNGs it would immediately crop away) and `data/unpaired/thermal/`
hard-linked unchanged at 640 × 512. **3997 frames each** — 4897 minus the 900
skipped lead-in.

## Stage 3 — train (CycleGAN, the recorded run)

```bash
python -m day2thermal.train --data-root data/unpaired --out runs/cyc \
    --mode cyclegan --load-size 286 --crop-size 256 --batch-size 4 \
    --lambda-cyc 10 --lambda-idt 5 --pool-size 50 \
    --num-scales-d 2 --norm instance --gan-mode lsgan \
    --thermal-mode rel8 --val-frac 0.1 --chunk 200 \
    --n-epochs 70 --n-epochs-decay 30 --save-freq 10 --seed 0
```

Every value above is transcribed from the run's own `config.json`, which
`train.py` writes into `--out` at startup. Resume an interrupted run with
`--resume runs/cyc/checkpoints/latest.pt`.

Emits `runs/cyc/checkpoints/epoch_0*.pt` (every 10 epochs, ~1.4 GB each) plus
`latest.pt`, and `runs/cyc/samples/eNNN_iNNNNN.png` progress strips — 870 of
them in the recorded run, which are the fastest way to see whether the model
is converging or collapsing.

## Stage 4 — translate the RGB-only dataset

Dump frames from the payload video (every 10th, ~3 fps from 30 fps source):

```bash
ffmpeg -i csi_.mp4 -vf "select=not(mod(n\,10))" -vsync 0 csi__frames/csi__%06d.jpg
```

→ **2973 frames at 1920 × 1080.**

```bash
python -m day2thermal.infer \
    --ckpt runs/cyc/checkpoints/latest.pt \
    --input csi__frames --out thermal_out
```

→ **2973 synthetic thermal PNGs at 640 × 360.** Note the output geometry:
CycleGAN preserves the *input* aspect, so these are 16:9 like the source RGB,
not 5:4 like the real thermal core. Downstream consumers that assume a
640 × 512 thermal frame need to letterbox.

With a **two_stage** checkpoint you would add `--temps 0,15,30` here to emit
one set per ambient temperature (`T0C/`, `T15C/`, …). `--temps` does nothing
for a cyclegan checkpoint — there is no temperature conditioning in that mode.

## Stage 5 — check the output

```bash
python -m day2thermal.check_thermalness \
    --ckpt runs/cyc/checkpoints/latest.pt --data-root data/unpaired --n 24
```

**Run this before trusting any unpaired output.** An unpaired translator has a
well-known shortcut: emit something close to the input's luminance. That
satisfies cycle-consistency perfectly (the mapping stays trivially invertible)
and partly fools the discriminator — but it is a greyscale filter, not a
modality change. Real LWIR is nearly uncorrelated with visible brightness,
because temperature is not albedo.

The check measures `corr(G(RGB), luminance(RGB))` against
`corr(real thermal, luminance(RGB))` over sampled frames. `r_gen` near `r_real`
means a genuine modality change; `r_gen` near 1 means the model learned to
desaturate. (`r_real` is computed on unregistered pairs, so treat it as a
ballpark reference, not an exact target.)

A real evaluation needs paired ground truth:

```bash
python -m day2thermal.eval --pred preds/ --gt data/aligned/val/thermal
```

which requires branch A, and is therefore **not available for the branch-B
output** — the unpaired route has no aligned holdout to score against. That
is the central limitation of the recorded run: nothing quantitative was
measured on it. The only honest verdict on branch-B output is
detector-in-the-loop (train on synthetic, evaluate on a **real** thermal
holdout), per section 6 of the research report.

## Pipeline at a glance

| Stage | Command | Out | Count | Size |
|---|---|---|---|---|
| 1 | `extract_frames` | `data/raw/{rgb,thermal}` | 4897 pairs | 3840×2160 / 640×512 |
| 2A | `register` | `data/aligned/{train,val}` | 316 pairs | 460×445 |
| 2B | `prep_unpaired` | `data/unpaired/{rgb,thermal}` | 3997 each | 640×360 / 640×512 |
| 3 | `train --mode cyclegan` | `runs/cyc/checkpoints` | 10 ckpts | ~1.4 GB each |
| 4 | `ffmpeg` + `infer` | `thermal_out` | 2973 | 640×360 |

**None of these artifacts are in git** — no frames, no checkpoints, no
generated thermal. `.gitignore` blocks all of it; the repo carries the code
that regenerates them plus the manifest that identifies the 316 aligned pairs.

## If you are starting from scratch

Run the [smoke test](#smoke-test-no-real-data-needed) first — it exercises
extract → register → train → infer → eval end to end on dummy data in about a
minute, and will surface a broken environment before you spend 15 h on a real
run. Then prefer **branch A + `--mode two_stage`**: it is the method this
repository is actually about, it gives you a scoreable val split, and it is
the only route to temperature-controllable output. Branch B is the fallback
for when registration cannot be made to work.

---

## Data

Only the **correlated (paired) RGB + thermal stream experiment** is recorded
here, and only as metadata — see [`data/aligned/DATASET.md`](data/aligned/DATASET.md)
and [`data/aligned/manifest.csv`](data/aligned/manifest.csv).

**No imagery is tracked in git.** The manifest carries per-frame SHA-256
checksums for both modalities plus intensity statistics, which is enough to
verify an out-of-band copy of the pixels is the exact set these splits were
built from. `.gitignore` blocks images, video, and checkpoints.

## Practical tips

* Lock the thermal camera's AGC, or record radiometric. Per-frame auto-gain
  makes the regression target non-stationary — the single most common way to
  waste a training run.
* Diversity beats quantity. The paper found that training on its narrow ReID
  split generalised poorly and had to collect the varied VOC split. Feed the
  model every background, altitude and time of day you have.
* **Audit hallucinations.** A GAN will happily paint hotspots that were never
  there. Compare detector false-positive rates on real vs synthetic thermal.
* This model is per-frame with no temporal consistency. For video-consistent
  output, temporally smooth the inputs or move to a vid2vid / temporal-
  discriminator setup.
* Synthetic thermal is a prior, not a measurement. RGB texture is only weakly
  informative about temperature; what the network genuinely exploits is shape,
  class, context, sky state and the ambient condition you hand it.

## Smoke test (no real data needed)

```bash
python tests/make_dummy_data.py --out /tmp/dummy
python -m day2thermal.extract_frames --rgb /tmp/dummy/rgb.mp4 \
    --thermal /tmp/dummy/thermal.mp4 --out /tmp/dummy/raw --fps 10
python -m day2thermal.register --raw /tmp/dummy/raw --out /tmp/dummy/aligned \
    --mode manual --points /tmp/dummy/points.json --auto-crop --val-frac 0.2 --chunk 10
python -m day2thermal.train --data-root /tmp/dummy/aligned --out /tmp/run \
    --mode pix2pix --load-size 72 --crop-size 64 --batch-size 4 \
    --n-epochs 2 --n-epochs-decay 0 --num-workers 0 --cpu
python -m day2thermal.infer --ckpt /tmp/run/checkpoints/latest.pt \
    --input /tmp/dummy/aligned/val/rgb --out /tmp/pred --cpu
python -m day2thermal.eval --pred /tmp/pred --gt /tmp/dummy/aligned/val/thermal
```

## Layout

```
day2thermal/          the pipeline package
  extract_frames.py     paired-video → synchronised frame pairs
  register.py           cross-modal homography, warp, temporal-chunk split
  pick_points.py        interactive correspondence picker
  prep_aligned.py       already-registered pair folders → train/val layout (session split)
  prep_unpaired.py      dataset prep for the CycleGAN fallback
  dataset.py            paired/unpaired loaders, thermal normalisation
  networks.py           U-Net generator (+extra head), PatchGAN / multi-scale D
  losses.py             GAN / L1 / KL / temperature losses
  train.py              pix2pix | two_stage | cyclegan training loops
  infer.py              batch translation, temperature sweep, 16-bit output
  eval.py               L1 / RMSE / PSNR / SSIM (°C in abs16 mode)
  check_thermalness.py  luminance-shortcut detector for unpaired models
research/             literature review, LWIR physics, design rationale
data/aligned/         metadata for the correlated-stream experiment (no pixels)
tests/                dummy-data generator for the end-to-end smoke test
```
