# Day RGB → Thermal Image Translation for UAV Datasets
### A deep analysis of ThermalGAN, a survey of the field, and the design rationale for the delivered pipeline

**Scope.** You have (a) a large day-RGB dataset of fixed-wing UAV imagery, (b) a smaller set of day and thermal videos recorded simultaneously of the same scenes, and (c) the goal of synthesizing a thermal version of the RGB dataset good enough to train downstream models. This report analyzes the paper you uploaded (Kniaz et al., *ThermalGAN*, ECCV 2018 Workshops, LNCS 11134, pp. 606–624), places it in the broader literature on visible-to-infrared translation, extracts the physics that the pipeline must respect, and justifies the concrete implementation that accompanies this report. One honesty note up front: this session has no web access, so the survey is drawn from the uploaded paper plus literature I know as of early 2026; all external citations should be verified before you quote them, and for anything published in the last year I recommend a follow-up pass with web search enabled.

---

## 1. The uploaded paper, in depth

### 1.1 Problem and central claim

ThermalGAN addresses cross-modality person re-identification: given a single color probe image, match the person against a gallery of real LWIR thermal images. The mechanism is to *translate* the color probe into thermal space and then match thermal-to-thermal using temperature histograms and MSER blobs as a "thermal signature." For your purposes the ReID matching machinery (Section 4.4 of the paper: SCA body segmentation, Bhattacharyya distance on temperature histograms, MSER distance) is peripheral; the durable contribution is the *translation framework* and the reasoning behind it.

### 1.2 Why color→thermal is one-to-many, and the 5 °C / 1 °C observation

The paper's foundational observation is that color-to-thermal translation is inherently **multimodal**: a person on a cold autumn day and on a hot summer afternoon can look identical in the visible band while their skin temperature differs substantially. Any deterministic RGB→thermal mapping therefore averages over incompatible answers. The authors quantify the consequence empirically: off-the-shelf GAN frameworks (pix2pix + noise, cLR-GAN, cVAE-GAN, cVAE-GAN++, BicycleGAN) predict absolute object temperature only to roughly **5 °C**, while the fine local contrasts that carry identity information (skin vs. cloth, eyes vs. brow) live at the **1 °C** scale. This gap between attainable global accuracy and required local accuracy is the paper's real insight, and it motivates everything else.

### 1.3 The two-stage decomposition

The fix is to split the prediction into a hard multimodal part and an easy unimodal part, then recombine:

**Stage 1 — thermal segmentation Ŝ.** A generator G₁: {Tᵢ, A} → Ŝᵢ predicts an image of *average object temperatures* ("thermal segmentation") from the color image A, conditioned on a **temperature vector Tᵢ** holding desired background and object temperatures. Architecturally G₁ is a modified BicycleGAN: the U-Net generator gains one extra convolutional and one extra deconvolutional layer for higher output resolution, and — the key twist — the random latent z is **replaced by the physically meaningful Tᵢ**, turning uncontrolled stochastic diversity into *controllable* diversity. The training objective (their Eq. 1) keeps the BicycleGAN inheritance: a VAE-GAN term that encourages multimodal coverage, an L1 term, the discriminator's adversarial term, a Kullback–Leibler term on the encoder E, and an additional **L₁ᵗʰᵉʳᵐᵃˡ latent-temperature-domain loss** tying the latent to actual temperatures.

**Stage 2 — relative contrasts R̂.** A second generator G₂: {Ŝᵢ, A} → R̂ᵢ predicts *relative* local temperature contrasts, conditioned on a 4-channel input (RGB + Ŝ). The authors observe (their Fig. 5) that relative contrasts are nearly invariant to ambient temperature — the same person filmed by cameras at 17 °C, 25 °C and 28 °C ambient produces almost identical R images — so this distribution is unimodal and a plain pix2pix backbone suffices. The final thermal image is the sum **B̂ = Ŝ + R̂**. The stated advantages: multimodality is confined to stage 1, and stage 2 sees a target with much lower variance and dynamic range, which directly improves the 1 °C-scale fidelity that matters. The two generator/discriminator pairs are trained **independently**.

### 1.4 The dataset lesson that matters as much as the architecture

The authors first tried training the translation GAN on their ReID split alone (15,118 pairs, 516 identities, sixteen FLIR ONE PRO cameras in one shopping-mall area) and found it **generalized poorly** — too few object classes and backgrounds. This forced them to collect the separate ThermalWorld **VOC split**: 5,098 aligned color–thermal pairs across four cities, all seasons, sun/rain/snow, object temperatures from −20 °C to +40 °C, with pixel-level annotations for ten classes. Translation quality was bought with *scene diversity*, not with more of the same footage. Transposed to your project: however many hours of one flight profile you have, variety of backgrounds, altitudes, weather and times of day will matter more than raw pair count.

### 1.5 Training and results

Training used PyTorch on a single GTX 1080 Ti — 76 h for G₁/D₁ and 68 h for G₂/D₂ — with minibatch SGD, Adam, lr 2·10⁻⁴, β₁ = 0.5, β₂ = 0.999 (the delivered code adopts these defaults). On generation quality, ThermalGAN achieved the best of both worlds among six baselines: highest AMT fooling rate (30.41 % vs. 28.12 % for BicycleGAN) *and* highest LPIPS diversity (0.167 vs. 0.153 for cVAE-GAN), where the baselines trade one for the other. On the downstream ReID task it set the state of the art on ThermalWorld: rank-1 19.48 / nAUC 41.84 single-shot and 22.59 / 57.35 multi-shot, ahead of TONE_1/2, HOG, and the SYSU one-stream/two-stream/zero-padding networks. Two side findings are worth registering: the two-stream network fine-tuned from *near*-infrared performed worst (NIR priors transfer badly to LWIR — your thermal camera is almost certainly LWIR, so beware NIR-pretrained anything), and only ThermalGAN outputs **calibrated temperatures in °C**, which is what enables physically meaningful signature matching.

### 1.6 Critical assessment

Strengths: the low/high-frequency decomposition is well-motivated and cheap; temperature conditioning converts a nuisance (multimodality) into a feature (controllable weather augmentation); calibrated-temperature output is genuinely differentiating; the evaluation includes both perceptual and task-level metrics. Limitations you should not inherit blindly: stage 1 requires **pixel-level semantic annotations** to supervise the thermal segmentation (expensive; your data has none); the FLIR ONE PRO sensor is 160×120 native, so the learned priors are low-resolution and person/street-centric; the model is strictly per-frame with no temporal consistency; two-stage systems compound errors (a wrong Ŝ misleads G₂); and 2018-era GAN training is less stable than modern recipes. The delivered implementation keeps the decomposition and the conditioning while removing the annotation dependency (Section 5).

---

## 2. LWIR physics your pipeline should respect

Thermal cameras in this class operate in the longwave infrared (8–14 µm) and measure emitted radiance — emissivity times the Planck function of surface temperature, plus a reflected-ambient component — not illumination. Several consequences are specific to UAV imagery and worth encoding as data-pipeline decisions rather than hoping the network learns them.

The **cold-sky effect** dominates air-to-air scenes: a clear sky has very low downwelling LWIR radiance, so its apparent temperature is tens of degrees below ambient, which is why aircraft against clear sky show strong positive contrast; overcast raises the background level and compresses contrast. **Hot components** differ by propulsion type: combustion engines and exhaust stacks are the brightest LWIR features of a fixed-wing UAV (the exhaust *plume* itself radiates mostly in MWIR CO₂ bands and is faint in LWIR — the hot metal is what you see), while electric aircraft show milder motor/ESC/battery warm spots. **Thin fast parts** — propellers — have low thermal mass and small optical cross-section and are close to invisible in LWIR. **Solar loading** warms airframe skins during the day, so the same aircraft has different signatures at 09:00 and 15:00 — a direct, physical justification for the paper's temperature-vector conditioning. Finally, **uncooled microbolometers** bring artifacts the pairing stage must handle: periodic NUC/FFC shutter events freeze or jump the video (the extractor's `--drop-static-thermal` flag), slow pixel time constants blur fast targets, and per-frame AGC makes 8-bit output a *non-stationary* function of scene temperature — lock the AGC span or, much better, record radiometric 16-bit and use the pipeline's `abs16` mode so the network learns calibrated temperatures as ThermalGAN did.

A sober expectation-setting note: RGB texture is only weakly informative about temperature. What the network can genuinely exploit is shape, class, context, sky state and the ambient condition you feed it; it will learn "airframe of this type against clear sky at ambient T looks like this," which is exactly what a detector-training dataset needs, but it cannot recover the true temperature of *this particular* engine at *this particular* moment. Synthetic thermal is a prior, not a measurement.

---

## 3. The surrounding literature

### 3.1 Paired translation (your regime)

pix2pix (Isola et al., CVPR 2017) established the conditional-GAN + L1 + PatchGAN template that everything here builds on; pix2pixHD (Wang et al., CVPR 2018) added coarse-to-fine generation, multi-scale discriminators and feature-matching loss — the multi-scale discriminator in particular helps when targets are small relative to the frame, as distant UAVs are (exposed as `--num-scales-d 2`). BicycleGAN (Zhu et al., NeurIPS 2017) contributed the cVAE-GAN + cLR-GAN machinery for multimodal outputs that ThermalGAN adapts. Beyond ThermalGAN itself, the closest published relative of your task is **InfraGAN** (Özkanoğlu & Ozer, *Pattern Recognition Letters* 2022), which trains visible→IR with an SSIM-based loss and a pixel-level discriminator and — importantly for you — evaluates on **VEDAI aerial imagery** as well as FLIR ADAS, demonstrating that paired vis→IR works at aerial viewpoints.

### 3.2 Unpaired and semi-supervised (your fallback if registration ever fails)

CycleGAN and the UNIT/MUNIT/DRIT++ family remove the alignment requirement at the cost of weaker pixel fidelity and a real risk of hallucinated content — acceptable for style, risky for radiometry. Two works are worth knowing even though their direction or setting differs: **PearlGAN** (Luo et al., *IEEE T-ITS* 2022) translates nighttime TIR→visible for driving using top-down attention and a structured-gradient-alignment loss whose edge-preservation idea transfers directly to vis→TIR; and Lee et al.'s **edge-guided multi-domain RGB→TIR** model (ICRA 2023) showed that *synthetic* TIR produced this way measurably improves downstream TIR tracking/detection — the clearest published precedent that your overall plan (train perception models on GAN-generated thermal) is sound. **DR-AVIT** (IEEE TGRS 2024, with the companion AVIID dataset) tackles *aerial* visible→infrared translation explicitly, using disentangled geometry/style representations to get diverse-and-realistic outputs; it is the closest aerial-specific precedent and worth replicating against once your baseline works.

### 3.3 Diffusion-era methods

Palette (Saharia et al., SIGGRAPH 2022) demonstrated that a single conditional DDPM beats task-specific GANs across image-to-image tasks; BBDM (Li et al., CVPR 2023) formulates translation as a Brownian bridge between domains; ControlNet (Zhang et al., ICCV 2023) makes it practical to condition a large pretrained latent-diffusion model on your RGB (or its edges) and fine-tune toward TIR output on modest data. There is also early work injecting physics into IR generation with diffusion (a 2024 arXiv line on physics-informed diffusion for infrared image generation, adding radiometric constraints to the objective — verify the exact reference before citing). The honest trade-off as of my knowledge: diffusion gives better texture realism and mode coverage, but sampling is 1–2 orders of magnitude slower (it matters when translating a whole dataset), calibrated-temperature output is harder to enforce, and small paired datasets favor the GAN's stronger supervised signal. Recommendation: GAN first for the radiometric branch and for bulk generation; consider a ControlNet/BBDM pass later as a phase-2 realism upgrade for 8-bit visual datasets.

### 3.4 Datasets and benchmarks to know

For paired RGB–thermal at aerial or surveillance viewpoints: DroneVehicle (aligned drone RGB-IR vehicle detection), VEDAI (aerial vehicles, visible+IR), LLVIP (aligned low-light surveillance pairs), KAIST multispectral pedestrians, FLIR ADAS, M3FD (fusion + detection), HIT-UAV (UAV-borne thermal detection). If your dataset is imagery **of** UAVs (air-to-air / counter-UAV — the phrase "thermal dataset of UAV" reads this way, and the pipeline supports either reading), the Anti-UAV thermal tracking benchmark series is the natural downstream evaluation target, and visible-band air-to-air sets such as Det-Fly or DUT Anti-UAV are candidates for translation once your model is trained. Finally, physics-based simulators — DIRSIG, OKTAL-SE's SE-WORKBENCH-IR, MuSES/TAIThermIR for vehicle signatures — are the classical alternative; a pragmatic hybrid is simulator output refined by a GAN trained on your real pairs (the SimGAN idea).

### 3.5 Method choice at a glance

| Method family | Needs aligned pairs | Multimodal / controllable | Calibrated °C output | Fit for your case |
|---|---|---|---|---|
| pix2pix / pix2pixHD | yes | no | yes (with abs16 targets) | **Baseline — start here** |
| ThermalGAN-style two-stage | yes | yes (temperature vector) | yes | **Main track** — weather-controllable augmentation |
| BicycleGAN | yes | yes (latent z) | possible | superseded by T-conditioning for this task |
| CycleGAN / MUNIT / DRIT++ | no | partly | unreliable | fallback if registration fails |
| ControlNet / BBDM / Palette (diffusion) | preferably | yes | hard | phase-2 realism upgrade; slow bulk inference |
| Physics simulators (+GAN refinement) | no | yes (by construction) | yes | complement for rare conditions (night, winter) |

---

## 4. Why the delivered pipeline is shaped the way it is

Because you own **synchronized day/thermal videos**, you are in the strongest possible regime — supervised paired translation — and should not pay the fidelity tax of unpaired methods. The delivered code therefore implements two tracks sharing one data pipeline. Track A is a clean pix2pix-class baseline (conditional PatchGAN + L1, LSGAN by default, optional multi-scale discriminator): it produces a trustworthy first result within a day of GPU time and gives you the reference number every later change must beat. Track B is the ThermalGAN-inspired two-stage model with one deliberate substitution: since you have no pixel-level annotations, the "thermal segmentation" Ŝ is replaced by a **Gaussian low-pass of the ground-truth thermal frame**, preserving the paper's low-frequency/high-frequency decomposition (multimodal base + unimodal contrasts) with zero labeling cost; the temperature vector becomes a scalar ambient-temperature plane (frame-mean by default, real logged °C via `--temps-csv` if you have weather records), and sweeping it at inference yields the multimodal thermal sets of the paper — repurposed as free weather augmentation for your detector data. The README's correspondence table lists every deviation from the paper explicitly.

Two properties of *your* data made specific engineering choices easy. First, a rigidly co-mounted camera pair observing distant scenes (air-to-air targets, ground from altitude) has essentially **zero parallax**, so a single homography per recording is not an approximation but nearly exact — hence the one-homography registration design, with RGB warped into the thermal frame so radiometric values are never resampled. Second, video pairs are temporally dense and autocorrelated, so the train/val split is done in contiguous temporal chunks; a random frame split would leak near-duplicates and inflate every metric you report.

---

## 5. Evaluation protocol

Report three tiers. Tier 1, pixel fidelity on the held-out real pairs: L1/RMSE (in °C if radiometric — recall the 5 °C vs 1 °C framing), PSNR, SSIM; the delivered `eval.py` computes these. Tier 2, distributional realism: FID/KID between synthetic and real thermal sets, plus a small human A/B in the spirit of the paper's AMT test. Tier 3 — the one that actually answers your question — **detector-in-the-loop**: train your UAV detector (a) on real thermal only, (b) on synthetic thermal only, (c) on real + synthetic, and evaluate all three on a real-thermal holdout from flights never seen by the GAN. The gap between (b) and (a) measures translation quality in task units; the lift of (c) over (a) measures the augmentation value you are ultimately buying. Include a hallucination audit: false-positive hotspots in synthetic frames that a detector learns to expect are the characteristic failure mode of GAN-generated training data.

---

## 6. Pitfalls checklist

Temporal sync drift over long clips (chunk recordings, re-check offset per flight); NUC/FFC frozen frames (drop them); per-frame AGC (lock it or go radiometric); train/val temporal leakage (chunked split, and hold out entire *flights* for the final test); overfitting to few backgrounds (the paper's VOC lesson — feed diversity); tiny targets drowned by background loss (train on crops around the target when boxes exist, or enable the multi-scale discriminator); night generalization (a day-trained model will not invent correct night thermal — if you need night, capture at least some night pairs or lean on a simulator for that slice); and per-frame flicker in generated video (acceptable for detector training on stills; move to temporal models if you need consistent clips).

---

## 7. References

Verify all of these before quoting — they are reproduced from memory (knowledge through early 2026) except [1], which is the uploaded paper.

[1] Kniaz, Knyaz, Hladůvka, Kropatsch, Mizginov. *ThermalGAN: Multimodal Color-to-Thermal Image Translation for Person Re-identification in Multispectral Dataset.* ECCV 2018 Workshops, LNCS 11134, pp. 606–624, 2019.
[2] Isola, Zhu, Zhou, Efros. *Image-to-Image Translation with Conditional Adversarial Networks* (pix2pix). CVPR 2017.
[3] Wang, Liu, Zhu, Tao, Kautz, Catanzaro. *High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs* (pix2pixHD). CVPR 2018.
[4] Zhu, Zhang, Pathak, Darrell, Efros, Wang, Shechtman. *Toward Multimodal Image-to-Image Translation* (BicycleGAN). NeurIPS 2017.
[5] Zhu, Park, Isola, Efros. *Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks.* ICCV 2017.
[6] Özkanoğlu, Ozer. *InfraGAN: A GAN architecture to transfer visible images to infrared domain.* Pattern Recognition Letters, 2022.
[7] Luo et al. *Thermal Infrared Image Colorization for Nighttime Driving Scenes with Top-Down Guided Attention* (PearlGAN). IEEE T-ITS, 2022.
[8] Lee et al. *Edge-guided Multi-domain RGB-to-TIR Image Translation for Training Vision Tasks with Challenging Labels.* ICRA 2023.
[9] *DR-AVIT: Toward Diverse and Realistic Aerial Visible-to-Infrared Image Translation* (+ AVIID dataset). IEEE TGRS, 2024.
[10] Saharia et al. *Palette: Image-to-Image Diffusion Models.* SIGGRAPH 2022.
[11] Li et al. *BBDM: Image-to-Image Translation with Brownian Bridge Diffusion Models.* CVPR 2023.
[12] Zhang, Rao, Agrawala. *Adding Conditional Control to Text-to-Image Diffusion Models* (ControlNet). ICCV 2023.
[13] Physics-informed diffusion for infrared image generation (2024 arXiv line; verify exact reference).
[14] Sun et al. *Drone-based RGB-Infrared Cross-Modality Vehicle Detection via Uncertainty-Aware Learning* (DroneVehicle). IEEE T-CSVT.
[15] Jia et al. *LLVIP: A Visible-Infrared Paired Dataset for Low-Light Vision.* ICCV Workshops 2021.
[16] Hwang et al. *Multispectral Pedestrian Detection: Benchmark Dataset and Baseline* (KAIST). CVPR 2015.
[17] Jiang et al. *Anti-UAV: A Large-Scale Benchmark for Vision-Based UAV Tracking* (thermal Anti-UAV series). IEEE T-MM / CVPR-W challenges.
[18] Shrivastava et al. *Learning from Simulated and Unsupervised Images through Adversarial Training* (SimGAN). CVPR 2017.
