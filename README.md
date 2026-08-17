# day2thermal

**Paired RGB → single-band infrared image translation for UAV imagery**
(day RGB → LWIR thermal is the design target; any pixel-aligned RGB → 1-channel
IR pair set — e.g. RGB → NIR — trains with the same code).

A supervised image-translation pipeline that learns to turn colour frames into
synthetic infrared frames, so that a large RGB-only dataset can be converted
into IR training data without re-flying every mission with an IR core.

The method implements **ThermalGAN** (Kniaz et al., ECCV Workshops 2018) —
see [Paper implemented](#paper-implemented) — adapted for the practical case of
having a *synchronised RGB + IR video pair* (or a folder of pre-registered
pairs) as supervision, plus a **pix2pix** baseline and a **CycleGAN** fallback.

```
                 ┌ extract_frames ─► register ──┐            (paired video)
 RGB + IR data ──┤ prep_aligned ────────────────┼─► train ─► infer ─► eval
                 └ prep_unpaired ───────────────┘   (GAN)    (translate  (vs real
                   (unpaired fallback)                        RGB set)    IR holdout)
```

Everything except `train.py` / `infer.py` / `check_thermalness.py` runs
without PyTorch (numpy + OpenCV only).

---

## Contents

1. [Why this is hard (and what the paper contributes)](#why-this-is-hard-and-what-the-paper-contributes)
2. [Paper implemented](#paper-implemented)
3. [How the models work — as implemented](#how-the-models-work--as-implemented)
4. [Install](#install)
5. [Usage, step by step](#usage-step-by-step)
6. [Worked example A — arsuf day + thermal streams](#worked-example-a--arsuf-day--thermal-streams)
7. [Worked example B — agricultural RGB → NIR pair set](#worked-example-b--agricultural-rgb--nir-pair-set)
8. [Data recorded in this repo](#data-recorded-in-this-repo)
9. [Practical tips](#practical-tips)
10. [Smoke test](#smoke-test-no-real-data-needed)
11. [Layout](#layout)

---

## Why this is hard (and what the paper contributes)

RGB→**thermal** is **one-to-many**. The same scene in the visible band can
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

**When the target is not thermal** (e.g. NIR, which is *reflected* light at
0.7–1.5 µm), there is no ambient-temperature ambiguity: the mapping is close to
deterministic given RGB, the two-stage decomposition and temperature
conditioning have no physical meaning, and `--mode pix2pix` is the right
model. See [worked example B](#worked-example-b--agricultural-rgb--nir-pair-set).

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
  (`--num-scales-d 2`).
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

## How the models work — as implemented

Everything below is what the code in `day2thermal/` actually does; file and
function names are given so it can be checked.

### Data model and normalisation (`dataset.py`, `utils.py::ThermalNorm`)

* A sample is `A` (RGB, 3×H×W) and `B` (IR, 1×H×W), both scaled to **[−1, 1]**
  (`x / 127.5 − 1`), matching the generator's `tanh` output.
* `--thermal-mode rel8` (default): 8-bit IR video / images are *relative*
  intensities. `abs16`: radiometric 16-bit counts → °C
  (`raw · raw_scale + raw_offset`, FLIR TLinear default 0.04 K/count) → linear
  map from `[tmin, tmax]` to [−1, 1]; the model then predicts *calibrated*
  temperatures. 3-channel IR files are collapsed to one channel on read.
* `T` (the temperature condition, used by `two_stage` only) is the value from
  `--temps-csv "filename,celsius"` if given, else **the mean of `B`** for that
  crop — read off the target, i.e. an oracle at training *and* validation
  time. At inference you supply it (`--temps`). Consequence: `two_stage` val L1
  is optimistic relative to deployment, where T is your guess.
* Paired loader (`PairedDayThermalDataset`): with `--load-size L` (default 286)
  both images are squash-resized to L×L then a random `--crop-size` window
  (default 256) is cut — **the same window from A and B**, which is what keeps
  them aligned; with `--load-size 0` the crop is taken at native resolution
  (no aspect distortion, only up-scaled if the frame is smaller than the crop).
  Augmentation: 50 % horizontal flip on both; brightness/contrast jitter
  (gain 0.85–1.15, bias ±15, applied to 80 % of samples) on **RGB only** — the
  IR target is never perturbed. Validation uses a centre crop, no augmentation.
* Unpaired loader (`UnpairedDayThermalDataset`): A and B are drawn
  independently. RGB is first downscaled to `--rgb-width` (default = thermal
  width) so a crop covers a similar angular field in both domains — otherwise
  the discriminator separates the domains by *zoom level* instead of modality.
* Splits are by **contiguous temporal chunk** (`register.py`, `--chunk`) or by
  **whole session** (`prep_aligned.py`), never a random shuffle: consecutive
  video frames are near-duplicates and would leak across the split.

### Networks (`networks.py`)

* **Generator — U-Net** (`UnetGenerator`): `num_downs = log2(crop_size)`
  stride-2 encoder stages (crop 256 → 8 stages, 3→64→128→256→512→512→512→512→512,
  1×1 bottleneck), a mirror decoder with skip concatenation at every level,
  InstanceNorm (default; `--norm batch|none`), LeakyReLU(0.2) down / ReLU up,
  optional dropout 0.5 in the inner 512-channel blocks (`--dropout`), and the
  ThermalGAN-style **extra head** (`ConvT(128→64) → norm → ReLU → Conv3×3(64→1)
  → tanh`; `--no-extra-head` for the plain pix2pix `ConvT(128→1) → tanh`).
  54.5 M parameters at ngf 64. Fully convolutional: `infer.py` accepts any size.
  `--crop-size` must be a power of two ≥ 32.
* **Discriminator — 70×70 PatchGAN** (`NLayerDiscriminator`, `--n-layers-d 3`):
  `k4s2 → k4s2 → k4s2 → k4s1 → k4s1(→1)`, receptive field 70 px, i.e. a 30×30
  map of per-patch logits for a 256×256 input; 2.8 M parameters. It is
  **conditional** wherever pairs exist: its input is `cat([A, B])`, so it
  judges "is this a plausible IR image *for this RGB*". `--num-scales-d 2`
  (`MultiScaleDiscriminator`, pix2pixHD) adds a second PatchGAN on the
  2×-average-pooled input (140-px effective receptive field); losses average
  over scales.
* **Temperature encoder** (`TemperatureEncoder`, only with `--use-vae`):
  5 stride-2 convs → global pool → (μ, log σ²) of an `--nz` = 8-dim latent.
* Weights initialised N(0, 0.02).

### Losses (`losses.py`) and optimiser

`GANLoss("lsgan")` = MSE of the patch map against 1 (real) / 0 (fake);
`"vanilla"` = BCE-with-logits. L1 weighted by `--lambda-l1` (100). Adam,
lr 2e-4, β = (0.5, 0.999), one optimiser per net; lr constant for
`--n-epochs` then linearly decayed to 0 over `--n-epochs-decay`.

### `--mode pix2pix` — one generator, one conditional discriminator (`train.py::step_pix2pix`)

```
fake = G(A)
D step:  L_D = ½ [ (D(A, fake.detach()) − 0)² + (D(A, B) − 1)² ]     # D updated
G step:  L_G = (D(A, fake) − 1)²  +  100 · ‖fake − B‖₁              # G updated, D frozen
```

Alternating updates every batch. `detach()` keeps D's loss from moving G;
`set_requires_grad(D, False)` keeps G's loss from moving D. The ½ on L_D slows
D relative to G. Why both terms: L1 alone regresses to the conditional median
(blurry, texture-less); the patch discriminator penalises exactly that and
forces sharp, plausible local structure, while L1 anchors global level and
layout to the ground truth. LSGAN keeps gradients alive for confidently
rejected fakes.

### `--mode two_stage` — ThermalGAN adaptation (`train.py::step_two_stage`)

```
S  = gaussian_blur(B, σ = --lowpass-sigma)      # low-frequency base (surrogate "thermal segmentation")
R  = B − S                                      # relative contrasts
Tpl = T broadcast to a 1×H×W plane
Ŝ  = G1([A, Tpl])                     D1 sees [A, Tpl, ·]  (5 ch)
     L_G1 = LSGAN + 100·‖Ŝ − S‖₁ + --lambda-temp·|mean(Ŝ) − T|  (+ --lambda-kl·KL with --use-vae)
R̂  = G2([A, Ŝ.detach()])            D2 sees [A, ·]       (4 ch)
     L_G2 = LSGAN + 100·‖R̂ − R‖₁
B̂  = clamp(Ŝ + R̂, −1, 1)
```

G2 is trained on the *predicted* base (detached), so it sees at training time
the same imperfect input it gets at test time, and the two GAN pairs stay
independent as in the paper. With `--use-vae`, `E(S)` gives (μ, log σ²),
`z = μ + σ·ε` is broadcast to `--nz` extra input planes of G1 (z = 0 at
inference). At inference `--temps` sweeps T.

### `--mode cyclegan` — unpaired fallback (`train.py::step_cyclegan`)

G: RGB→IR (3→1), F: IR→RGB (1→3), D_B on IR (1 ch), D_A on RGB (3 ch), both
**unconditional** (no pairs to condition on).

```
L_G,F = LSGAN(D_B(G(A))) + LSGAN(D_A(F(B)))
      + --lambda-cyc · [ ‖F(G(A)) − A‖₁ + ‖G(F(B)) − B‖₁ ]          # cycle consistency
      + --lambda-idt · [ ‖G(B⊗3) − B‖₁ + ‖F(mean_c(A)) − A‖₁ ]     # channel-adapted identity
D updated on an ImagePool(--pool-size) history of fakes.
```

The identity term anchors global intensity and prevents contrast inversion,
but it also nudges G towards a luminance-preserving map — the "greyscale
shortcut" that `check_thermalness.py` measures. `--lambda-idt 0` disables it.

### Training loop bookkeeping (`train.py::Trainer.fit`)

* `--out/config.json`: all args + thermal normalisation, written at start.
* Every `--log-every` iterations: loss line + a sample strip
  `--out/samples/eNNN_iNNNNN.png`. Columns — pix2pix: **RGB | real IR |
  generated**; two_stage: **RGB | real | S | Ŝ | B̂**; cyclegan: **RGB |
  unrelated real IR | G(RGB) | F(G(RGB)) | F(IR)**.
* Every epoch: `val L1` on up to `--val-max-batches` batches (paired modes:
  \|prediction − B\| on the **[−1, 1] scale**, i.e. twice the [0, 1] value
  `eval.py` reports; cyclegan: cycle-reconstruction L1, *not* a fidelity
  metric), then `checkpoints/latest.pt` (overwritten) and every `--save-freq`
  epochs `checkpoints/epoch_NNN.pt`. A checkpoint holds args, thermal
  normalisation, all nets and all optimiser states — `infer.py` rebuilds the
  architecture from it. Sizes: pix2pix ≈ 0.7 GB, two_stage ≈ 1.4 GB,
  cyclegan ≈ 1.4 GB (fp32 weights + Adam moments).
* `--resume PATH` restores nets, optimisers, epoch counter and fast-forwards
  the lr schedule.

### Inference (`infer.py`)

Rebuilds G (or G1+G2) from the checkpoint, reflect-pads the input to a
multiple of `crop_size`, translates, un-pads → output geometry = input
geometry. Writes one 8-bit PNG per input (same name); `--save-16bit` adds
`<name>_16bit.png` (calibrated counts in abs16, a 16-bit stretch in rel8).
`--temps` (two_stage only): °C in abs16, relative levels in [0, 1] in rel8
(mapped `t·2 − 1`); each value writes its own sub-folder (`T15C/`, `T0.5/`).
No `--temps` → T = 0 (mid-range). `--max-size` caps the long side first;
`--cpu` forces CPU. Device is CUDA when available.

### Evaluation (`eval.py`, `check_thermalness.py`)

`eval.py` matches prediction and ground-truth files by stem (ignoring a
`_16bit` suffix), resizes the prediction to the GT size if they differ, and
reports **L1, RMSE, PSNR, SSIM on the [0, 1] scale** (SSIM: 11×11 Gaussian
window, σ 1.5, C₁ = 0.01², C₂ = 0.03²); with `--thermal-mode abs16` also
`L1_degC` / `RMSE_degC`. `check_thermalness.py` compares
`corr(G(RGB), luminance(RGB))` with `corr(real IR, luminance(RGB))` to catch a
model that merely desaturates (r > 0.75 = shortcut, < 0.5 = decoupled).

---

## Install

```bash
pip install -r requirements.txt
```

`torch`, `numpy`, `opencv-python-headless`, `tqdm`. A CUDA GPU is required for
training in practice; inference works on CPU with `--cpu`.

---

## Usage, step by step

### 1. Extract synchronised frame pairs from two videos

```bash
python -m day2thermal.extract_frames \
    --rgb flight01_day.mp4 --thermal flight01_thermal.mp4 \
    --out data/flight01/raw --fps 5 --offset-ms 120 --drop-static-thermal
```

* `--rgb` / `--thermal` take a video file **or** a directory of frames (add
  `--rgb-fps` / `--thermal-fps` for directories). Frame time is
  `index / fps` — a constant-frame-rate assumption; check your streams'
  timestamps if in doubt.
* `--offset-ms` = thermal clock − RGB clock for the same event. Calibrate once
  per rig: put a hot object in view, find that event in both recordings,
  subtract the times. If both streams carry MISB KLV timestamps on a shared
  clock, the offset can be read from them (see worked example A).
* Pairs are accepted when both frames are within `--tol-ms` of the sample time
  (default: half the thermal frame period — the nearest thermal frame then
  always qualifies, and so does the nearest RGB frame as long as the RGB rate
  is at least the thermal rate; a slower RGB stream needs a larger `--tol-ms`).
  So the sync error per pair is bounded by ±`tol` plus your `--offset-ms`
  error. `--drop-static-thermal` skips frames whose thermal image is (nearly)
  identical to the previous one (mean |Δ| < 0.05 on the 0–255 scale) — the
  microbolometer's NUC/FFC shutter freezes. `--start`, `--duration`,
  `--max-pairs` limit the range.
* Output `out/rgb/000000.png …`, `out/thermal/000000.png …` — **the shared
  filename is the pairing**. Thermal is written single-channel.
* Radiometric note: if you can export 16-bit radiometric frames, point
  `--thermal` at that directory and later train with `--thermal-mode abs16`.
  Ordinary 8-bit video works too — you lose absolute temperature calibration.

### 2. Register RGB onto the thermal geometry

```bash
# export one raw pair, pick >=4 correspondences (day2thermal.pick_points),
# save as {"rgb": [[x,y],...], "thermal": [[x,y],...]} in NATIVE pixel coords
python -m day2thermal.register --raw data/flight01/raw --out tmp --dump-pair 0

python -m day2thermal.register --raw data/flight01/raw \
    --out data/flight01/aligned --mode manual --points calib/points.json \
    --auto-crop --val-frac 0.1
```

* RGB is warped **into the thermal frame** (`cv2.warpPerspective`, one 3×3
  homography for the whole recording), never the reverse — thermal pixels are
  never resampled. One homography is exact for a rigid pair viewing a distant
  scene (parallax ≈ f·b/Z px); for close-range ground scenes crop away the
  near field.
* `--mode manual` (recommended; RANSAC, 3-px threshold) · `--mode ecc`
  (the code's default; automatic ECC on Sobel edge maps, affine model — fragile
  across modalities) · `--mode identity` (scale + letterbox only).
* `--auto-crop` keeps the largest rectangle where the warped RGB is fully
  valid, so no black borders reach the network. `--crop x,y,w,h` sets it by
  hand.
* Every frame in `--raw` is written to `train/` or `val/` (this stage never
  discards frames); the split is by contiguous chunks of `--chunk` frames.
* Writes `registration.json` (homography, crop, ECC score, sizes) — **keep
  it**, it is the calibration record.

### 2b. Already-registered pair folders (skip `register`)

If a source dataset ships pixel-aligned pairs as
`<session>/RGB[-k]/<i>.png` + `<session>/NIR[-k]/<i>.png` (same footprint,
same instant), `prep_aligned` maps it straight onto the training layout:

```bash
python -m day2thermal.prep_aligned --root ir-rgb-dataset \
    --rgb-prefix RGB --ir-prefix NIR --out data/agri_rgb_nir/aligned \
    --val-sessions canola_06082019,drybean-30072020,Lentwheat_29082018,wheat_27072019
```

* Pairs by filename inside each `RGB*`/`NIR*` sub-folder pair (sub-folders
  pair by suffix: `RGB-1`↔`NIR-1`); crops both to their common size when they
  differ by ≤ `--max-size-diff` px (default 4 — the off-by-one crops common in
  source sets) and **skips** larger mismatches as mis-registrations; writes IR
  single-channel 8-bit and hard-links RGB (`--copy-rgb` to copy).
* Renames to `<session>[_k]_<index:06d>.png` so sorted order is
  session-then-frame; splits **by whole session** (`--val-sessions` explicit,
  or `--val-frac` picks evenly spaced sessions); writes `split.json`.
* `--dry-run` prints the plan without writing.

### 2c. Unpaired fallback prep

```bash
python -m day2thermal.prep_unpaired --raw data/raw --out data/unpaired --width 640 --skip-first 900
```

Downscales RGB once to `--width` (default: thermal width) and hard-links
thermal unchanged, for `--mode cyclegan` (which reads `root/rgb`, `root/thermal`
and does its own temporal-chunk split).

### 3. Train

Choose the mode:

| Data you have | Target | Mode |
|---|---|---|
| Registered pairs, 8-bit or radiometric thermal | thermal | `two_stage` (the ThermalGAN method; temperature-controllable) — after a `pix2pix` baseline |
| Registered pairs, non-thermal IR (NIR/SWIR reflectance) | NIR etc. | `pix2pix` |
| Same scenes, no usable registration | thermal | `cyclegan` + `check_thermalness` |

Baseline — start here:

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

Unpaired fallback:

```bash
python -m day2thermal.train --data-root data/unpaired --out runs/cyc \
    --mode cyclegan --load-size 286 --crop-size 256 --batch-size 4 \
    --lambda-cyc 10 --lambda-idt 5 --pool-size 50 --num-scales-d 2
```

Flags worth knowing (defaults in brackets): `--load-size` [286; 0 = native
crops], `--crop-size` [256, power of two], `--batch-size` [8],
`--num-scales-d` [1; 2 = multi-scale D], `--norm` [instance], `--gan-mode`
[lsgan], `--lambda-l1` [100], `--n-epochs` [100] + `--n-epochs-decay` [100],
`--save-freq` [10], `--log-every` [100], `--val-max-batches` [50],
`--num-workers` [4], `--thermal-mode` [rel8; abs16 with `--tmin/--tmax`], `--temps-csv`,
`--dropout`, `--no-extra-head`, `--use-vae`, `--resume`, `--max-iters`
(debug), `--cpu`, `--seed` [0]. Watch `--out/samples/*.png` and the per-epoch
`val L1` line.

### 4. Generate the synthetic IR dataset

```bash
python -m day2thermal.infer --ckpt runs/2stage/checkpoints/latest.pt \
    --input my_big_rgb_dataset/images --out my_synthetic_thermal \
    --temps 0,15,30 --save-16bit
```

One PNG per input image, same name and geometry (any input size; padded
internally). `--temps` sweeps the condition for a two-stage checkpoint (one
sub-folder per value) and does nothing for pix2pix/cyclegan. `--max-size` to
cap resolution, `--cpu` if no GPU. Note `latest.pt` is rewritten every epoch
while training runs — use an `epoch_NNN.pt` snapshot to avoid reading it
mid-write.

### 5. Evaluate against real IR

```bash
python -m day2thermal.infer --ckpt runs/p2p/checkpoints/latest.pt \
    --input data/flight01/aligned/val/rgb --out preds/p2p_val
python -m day2thermal.eval --pred preds/p2p_val --gt data/flight01/aligned/val/thermal \
    --out-json preds/p2p_val/metrics.json
```

Reports L1 / RMSE / PSNR / SSIM on [0, 1] (plus °C errors with
`--thermal-mode abs16`). Calibrate expectations against the paper's bar for
thermal: generic GANs ≈ 5 °C, useful fine contrast needs ≈ 1 °C.

**For thermal, the decisive test is not any of these metrics** — it is
detector-in-the-loop: train the UAV detector on the synthetic set, evaluate on
a **real** thermal holdout (research report, section 5). For unpaired models
run `check_thermalness` first:

```bash
python -m day2thermal.check_thermalness --ckpt runs/cyc/checkpoints/latest.pt --data-root data/unpaired --n 24
```

---

## Worked example A — arsuf day + thermal streams

The run that produced this project's synthetic thermal output — actual
commands, actual hyper-parameters, what each stage emitted.

### What was recorded

One session, two cameras running simultaneously:

| Source | Content | Native size |
|---|---|---|
| `day.ts` (`arsuf_day.ts`) | day RGB stream, H.264 MPEG-TS, 30 fps, 711.5 s | 3840 × 2160 |
| `thermal.ts` (`arsuf_thermal.ts`) | LWIR stream, same scene, same time, H.264 MPEG-TS, 25 fps, 8-bit AGC | 640 × 512 |
| `csi_.mp4` | the large **RGB-only** dataset to be translated | 1920 × 1080 |

Byte-exact identification of the two source streams (SHA-256, frame counts,
PTS statistics, embedded MISB ST 0601 KLV telemetry) is in
[`data/streams/STREAMS.md`](data/streams/STREAMS.md) /
[`manifest.json`](data/streams/manifest.json). The two KLV Precision Time
Stamp series share one encoder clock and give **`--offset-ms ≈ +893`**
(thermal started ~0.9 s before day; +897 at the end of the recording) — a
telemetry-derived value, to be confirmed with the hot-object procedure before
trusting it to the frame.

### Stage 0 — environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The recorded CycleGAN run took **~15.5 h wall** for 97 epochs (checkpoint
timestamps 22 Jul 15:13 → 23 Jul 06:55) on a single CUDA GPU.

### Stage 1 — split the correlated streams into frame pairs

```bash
python -m day2thermal.extract_frames \
    --rgb day.ts --thermal thermal.ts \
    --out data/raw --drop-static-thermal \
    --fps <RATE> --offset-ms <THERMAL_MINUS_RGB>
```

→ `data/raw/rgb/`, `data/raw/thermal/`: **4897 frames each**, `000000.png` …
`004896.png`. The exact `--fps` and `--offset-ms` were not recorded; 4897
pairs over 711.5 s is consistent with `--fps 7`, and the KLV analysis above
gives the offset.

### Stage 2 — two branches from the same raw frames

The raw pairs fed two independent experiments. **The shipped results came
from branch B.**

#### Branch A — paired / registered

```bash
python -m day2thermal.register --raw data/raw --out tmp --dump-pair 0
# pick >=4 correspondences -> calib/points.json
python -m day2thermal.register --raw data/raw --out data/aligned \
    --mode manual --points calib/points.json --auto-crop --val-frac 0.1
```

→ `data/aligned/{train,val}/{rgb,thermal}/` — **316 pairs at 460 × 445**
(116 train / 200 val), catalogued in
[`data/aligned/manifest.csv`](data/aligned/manifest.csv). The 316 pairs are
frames `000000`–`000315`: `register.py` writes every frame in `--raw`, so the
raw directory used for this branch held only those 316 pairs at the time
(e.g. an extraction capped with `--max-pairs`); the split is exactly what
`--chunk 200 --val-frac 0.1` yields on 316 frames (2 chunks, chunk 0 = frames
0–199 held out — hence val > train). The auto-crop from 640 × 512 down to
460 × 445 means the warped RGB footprint did not cover the thermal frame —
most simply because the day camera's field of view is *narrower* than the
thermal's (a strongly rotated mount would shrink the inscribed rectangle too;
the lost `registration.json` would settle it). **No training run against this
branch survives**, and its `registration.json` was not preserved — see
[`data/aligned/DATASET.md`](data/aligned/DATASET.md).

#### Branch B — unpaired (what actually generated the output)

```bash
python -m day2thermal.prep_unpaired --raw data/raw --out data/unpaired \
    --width 640 --skip-first 900
```

→ `data/unpaired/rgb/` at **640 × 360**, `data/unpaired/thermal/` hard-linked
at 640 × 512, **3997 frames each** (4897 minus the 900 skipped lead-in).

### Stage 3 — train (CycleGAN, the recorded run)

```bash
python -m day2thermal.train --data-root data/unpaired --out runs/cyc \
    --mode cyclegan --load-size 286 --crop-size 256 --batch-size 4 \
    --lambda-cyc 10 --lambda-idt 5 --pool-size 50 \
    --num-scales-d 2 --norm instance --gan-mode lsgan \
    --thermal-mode rel8 --val-frac 0.1 --chunk 200 \
    --n-epochs 70 --n-epochs-decay 30 --save-freq 10 --seed 0
```

Transcribed from the run's `config.json`. Emitted `runs/cyc/checkpoints/
epoch_0*.pt` (~1.4 GB each) + `latest.pt`, and 870 sample strips.

### Stage 4 — translate the RGB-only dataset

```bash
ffmpeg -i csi_.mp4 -vf "select=not(mod(n\,10))" -vsync 0 csi__frames/csi__%06d.jpg
python -m day2thermal.infer --ckpt runs/cyc/checkpoints/latest.pt \
    --input csi__frames --out thermal_out
```

→ **2973 frames** in, **2973 synthetic thermal PNGs at 640 × 360** out.
`infer.py` preserves input geometry, so the 1920 × 1080 frames were reduced
with `--max-size 640` (or pre-resized) — that flag was not recorded. The
output is 16:9 like the source RGB, not 5:4 like the thermal core; downstream
consumers that assume 640 × 512 need to letterbox. `--temps` does nothing for
a cyclegan checkpoint.

### Stage 5 — check the output

```bash
python -m day2thermal.check_thermalness \
    --ckpt runs/cyc/checkpoints/latest.pt --data-root data/unpaired --n 24
```

The unpaired route has no aligned holdout, so `eval.py` is **not available
for the branch-B output**; nothing quantitative was measured on it. The only
honest verdict is detector-in-the-loop.

### If you are starting from scratch on thermal

Run the [smoke test](#smoke-test-no-real-data-needed), then prefer **branch A
+ `--mode two_stage`**: it is the method this repository is about, it gives a
scoreable val split, and it is the only route to temperature-controllable
output. Branch B is the fallback when registration cannot be made to work.

---

## Worked example B — agricultural RGB → NIR pair set

A second, independent use of the same pipeline: a pre-registered UAV crop
dataset (`ir-rgb-dataset/`: canola, drybean, lentil-wheat, wheat; 15
sessions, 2018–2020; 8970 RGB↔**NIR** pairs at ≈1098 × 798, 8-bit, NIR stored
as replicated 3-channel grey). NIR is *reflected* near-infrared, so this
trains an **RGB→NIR reflectance** model — a different physical quantity from
thermal, and much more predictable from RGB (chlorophyll → bright NIR).

### Prep

```bash
python -m day2thermal.prep_aligned --root ir-rgb-dataset --rgb-prefix RGB --ir-prefix NIR \
    --out data/agri_rgb_nir/aligned \
    --val-sessions canola_06082019,drybean-30072020,Lentwheat_29082018,wheat_27072019
```

→ **7331 train / 1632 val** pairs; one whole session per crop held out; the
NIR frames' consistent +1 px were cropped away; 7 pairs skipped because RGB
and NIR sizes differed by 15–190 px (mis-cropped in the source). Recorded in
[`data/agri_rgb_nir/aligned/split.json`](data/agri_rgb_nir/aligned/split.json).

### Train

```bash
python -m day2thermal.train --data-root data/agri_rgb_nir/aligned --out runs/nir_p2p \
    --mode pix2pix --load-size 0 --crop-size 256 --batch-size 8 \
    --num-scales-d 2 --norm instance --gan-mode lsgan --lambda-l1 100 \
    --thermal-mode rel8 --n-epochs 40 --n-epochs-decay 20 --save-freq 10 \
    --log-every 200 --val-max-batches 50 --num-workers 4 --seed 0
```

Choices: `pix2pix` (no ambient ambiguity to model), native-resolution crops
(`--load-size 0`, no squash-resize — the crop-row texture is the content),
multi-scale D for structure beyond 70 px, session-level holdout for an honest
generalisation number.

**What was actually run (2026-08-17, RTX 5050 Laptop 8 GB, ~3.8 GB VRAM):**

* Sanity check first: 60 iterations, then `infer` + `eval` on 100 held-out
  images → L1 0.098 / RMSE 0.123 / PSNR 18.5 / SSIM 0.66 — RGB→NIR is learned
  almost immediately (vegetation bright, soil dark), the rest is texture.
* The full command above: 916 iterations/epoch at ≈0.23 s/iteration
  (≈215 s/epoch, so the planned 60 epochs ≈ 3.6 h). Per-epoch `val L1`
  ([−1, 1] scale, 50 batches of centre crops from the held-out sessions):

  | epoch | 1 | 2 | 3 | 4 | 5 | 6 |
  |---|---|---|---|---|---|---|
  | val L1 | 0.143 | 0.131 | 0.136 | 0.132 | 0.125 | 0.108 |

* **The run was cut off after epoch 6** by a restart of the machine and was
  deliberately not resumed. What survives locally (not in git — `runs/` is
  ignored): `runs/nir_p2p/checkpoints/latest.pt` (epoch 6, 0.72 GB),
  `config.json`, 33 sample strips. To continue it, re-issue the same command
  with `--resume runs/nir_p2p/checkpoints/latest.pt`; to train from scratch,
  use a fresh `--out`.

### Evaluate on the held-out sessions

```bash
python -m day2thermal.infer --ckpt runs/nir_p2p/checkpoints/latest.pt \
    --input data/agri_rgb_nir/aligned/val/rgb --out preds/nir_p2p_val
python -m day2thermal.eval --pred preds/nir_p2p_val --gt data/agri_rgb_nir/aligned/val/thermal \
    --out-json preds/nir_p2p_val/metrics.json
```

All 1632 val pairs at full 1098 × 798 (~0.1 s/image on the GPU, ~4 min in
total). `eval.py` numbers are on the [0, 1] scale — half the training-log
`val L1` values. Any checkpoint works here (`epoch_NNN.pt` snapshots are
written every `--save-freq` epochs once a run gets that far); for a finished
run, comparing the end of the flat-lr phase with `latest.pt` shows what the
decay phase bought.

### Generate NIR for any RGB folder

```bash
python -m day2thermal.infer --ckpt runs/nir_p2p/checkpoints/latest.pt \
    --input /path/to/rgb_images --out /path/to/synthetic_nir
```

One 8-bit single-channel PNG per input, same name and geometry. `--temps` is
ignored for a pix2pix checkpoint; `--save-16bit` only writes a 16-bit stretch
in `rel8` mode.

---

## Data recorded in this repo

**No imagery, video or checkpoints are tracked in git** — `.gitignore` blocks
all of it. The repository carries the code that regenerates them plus
manifests that identify the data:

* [`data/aligned/DATASET.md`](data/aligned/DATASET.md) +
  [`manifest.csv`](data/aligned/manifest.csv) — the 316 arsuf aligned pairs
  (per-frame SHA-256 of both modalities + intensity statistics).
* [`data/streams/STREAMS.md`](data/streams/STREAMS.md) +
  [`manifest.json`](data/streams/manifest.json) — the two arsuf source
  streams (SHA-256, frame counts, PTS statistics, KLV telemetry, sync offset).
* [`data/agri_rgb_nir/aligned/split.json`](data/agri_rgb_nir/aligned/split.json)
  — the session split of the agricultural RGB→NIR set.

The checksums let you verify an out-of-band copy is byte-identical to what
these records describe.

---

## Practical tips

* Lock the thermal camera's AGC, or record radiometric. Per-frame auto-gain
  makes the regression target non-stationary — the single most common way to
  waste a training run.
* Diversity beats quantity. The paper found that training on its narrow ReID
  split generalised poorly and had to collect the varied VOC split. Feed the
  model every background, altitude and time of day you have.
* Hold out whole flights/sessions for the number you report; adjacent frames
  leak.
* **Audit hallucinations.** A GAN will happily paint hotspots that were never
  there. Compare detector false-positive rates on real vs synthetic thermal.
* This model is per-frame with no temporal consistency. For video-consistent
  output, temporally smooth the inputs or move to a vid2vid / temporal-
  discriminator setup.
* Synthetic thermal is a prior, not a measurement. RGB texture is only weakly
  informative about temperature; what the network genuinely exploits is shape,
  class, context, sky state and the ambient condition you hand it.

## Smoke test (no real data needed)

Exercises extract → register → train → infer → eval end to end on dummy data
in about a minute:

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
  dataset.py            paired/unpaired loaders, IR normalisation
  networks.py           U-Net generator (+extra head), PatchGAN / multi-scale D, VAE encoder
  losses.py             GAN (LSGAN/vanilla) and KL losses
  train.py              pix2pix | two_stage | cyclegan training loops
  infer.py              batch translation, temperature sweep, 16-bit output
  eval.py               L1 / RMSE / PSNR / SSIM (°C in abs16 mode)
  check_thermalness.py  luminance-shortcut detector for unpaired models
  torch_utils.py        seeding, Gaussian blur, padding, ImagePool
  utils.py              torch-free I/O and ThermalNorm
research/             literature review, LWIR physics, design rationale
data/                 manifests only (no pixels): aligned/, streams/, agri_rgb_nir/
tests/                dummy-data generator for the end-to-end smoke test
```
