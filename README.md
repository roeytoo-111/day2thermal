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

`day2thermal/check_thermalness.py` is a quick sanity check on whether generated
frames have plausible thermal statistics before you spend a training run.

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
  prep_unpaired.py      dataset prep for the CycleGAN fallback
  dataset.py            paired/unpaired loaders, thermal normalisation
  networks.py           U-Net generator (+extra head), PatchGAN / multi-scale D
  losses.py             GAN / L1 / KL / temperature losses
  train.py              pix2pix | two_stage | cyclegan training loops
  infer.py              batch translation, temperature sweep, 16-bit output
  eval.py               L1 / RMSE / PSNR / SSIM (°C in abs16 mode)
  check_thermalness.py  quick plausibility check on generated frames
research/             literature review, LWIR physics, design rationale
data/aligned/         metadata for the correlated-stream experiment (no pixels)
tests/                dummy-data generator for the end-to-end smoke test
```
