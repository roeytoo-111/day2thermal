# Thermal (LWIR) Object Detection: Synthetic Data Generation, Domain Adaptation and Detector Development

**Period:** 2026-09-28 to 2026-10-11 (ongoing). **Role:** sole technical owner of the ML pipeline — data engineering, labeling tooling, model training, evaluation methodology, and cloud infrastructure.

Written for: a resume / portfolio write-up. This describes the technical work in detail — methods, tools built, experiments run, and results — so it can be trimmed into bullet points later, but with enough substance to speak to any of it in an interview.

---

## 1. Problem and constraints

Goal: a small-object detector (a fixed-wing drone) on a long-wave infrared (LWIR, 640×512) camera, paired with a separate 4K RGB camera on the same rig. Constraints that shaped every decision:

- **Real labelled thermal data was scarce** — initially one ~10-minute flight recording, later a handful of additional sessions from two different thermal camera/lens combinations.
- **No off-the-shelf thermal drone dataset existed** that matched the camera, altitude, and target size regime. External public datasets (surveyed in detail, see §7) were all rejected for concrete, measured reasons.
- **The target is tiny**: 3–20 px objects in a 640×512 frame, often only a few pixels wide even when the detector does catch them.
- **Strict evaluation discipline was imposed from day one** (self-imposed, not given): no train/test leakage, location-aware scoring (not just frame-level), confidence floors matched across models being compared, and — critically — **exit criteria written down before every experiment ran**, not after seeing the result. This discipline caught several false "wins" that later turned out to be noise or leakage (see §9).

---

## 2. Data leakage discovery and the honest baseline

The first thing I did on this project was audit the existing training data against the evaluation video, because the reported detector performance (~78% recall) looked too good relative to the stated difficulty.

**Method:** grouped training images by filename family and cross-referenced against existing perceptual-hash/SSIM similarity results. Found that a filename family covering 2,260 of ~5,000 training images (45% of the set) was in fact the *entire* evaluation video, sampled every 5th frame, with a recoverable frame offset (`frame ≈ 5.002·index + 14`).

**Fix:** rebuilt a leak-free training set by dropping the whole contaminated session (not just the exact-duplicate frames the automated similarity check had flagged — the near-duplicate neighbour frames needed removing too).

**Result:** re-scored the old models on the only truly unseen frames. Reported recall dropped from ~78% to **25–38%** depending on model and threshold. This became the project's honest baseline and the reference point for every later improvement claim. Documented clearly in the project notebook as "the previous ~78% recall was mostly memorization."

This also established a working principle that recurred constantly later: **a result that looks too good is treated as a leak or a methodology bug until proven otherwise**, and is checked before being reported as an improvement.

---

## 3. Evaluation methodology (built from scratch, iterated five times)

Off-the-shelf detection metrics were not sufficient for this problem and were progressively replaced:

1. **Frame-level recall → location-aware recall.** Early scoring counted a frame as "caught" if the detector fired *anywhere* in a labelled-positive frame. A model that fires everywhere gets free recall this way (measured: one generator variant hit 94.6% recall with 79% false-fire under this metric). Rebuilt ground truth as **box-level**, with a custom labeling tool (below), and a hit rule of "detection center within max(8px, 0.75× the box's long side) of the GT center, or IoU ≥ 0.1" (IoU alone is unstable at these target sizes).

2. **Single-threshold comparison → recall at matched false-fire rate.** Different models/runs naturally pick different operating points. Built a scorer that reports **recall at FF ≤ 5%/10%/25%** — the only way to compare models fairly when they fire at different rates. Logging convention was standardized: always log at a low confidence floor (0.05) and filter downstream, never discard low-confidence detections at inference time, because "you can't recover detections you never logged."

3. **Coarse threshold grid → exact thresholds.** Initially scored at 13 fixed confidence thresholds. Rebuilt the scorer to try *every distinct score value* as a candidate threshold, so "recall at FF ≤ x%" is computed exactly rather than approximated — this removed a real source of discretization noise once I discovered (step 5 below) how large the remaining noise floor was.

4. **A confidence-floor mismatch bug**: one run's detections had been logged starting at confidence 0.10, another's at 0.05 — a 30% of one model's detections were silently excluded from a "fair" comparison. Fixed by having the scorer auto-detect the common floor across all models being compared and refuse to score below it.

5. **Discovering and quantifying run-to-run noise.** After several training runs where a data change appeared to help by ~5 percentage points, I ran the *same* training recipe twice with only the random seed changed, specifically to find out whether 5pp meant anything. **Finding: seed noise alone moved recall by up to 15–20 percentage points on some evaluation subsets** (particularly on a session recorded with different camera/lighting conditions than training). This was a significant and somewhat uncomfortable result — it invalidated several earlier single-run conclusions — and it changed the project's decision rule going forward: **every real conclusion required ≥2 training seeds per configuration, and a difference only counted if it exceeded the observed seed-to-seed spread for that specific evaluation subset.** This is documented explicitly in the notebook as a policy change, with the old "+5pp passes" conclusions marked as superseded.

6. **Ground-truth background classification.** Built a classifier (`gt_background.py`) that automatically tags each ground-truth drone box as "against sky" or "against terrain" using a robust (median-absolute-deviation) local-contrast measure in a ring around the box, specifically excluding a halo around the target itself (an earlier naive version miscounted the target's own thermal glow as background clutter). This let me split evaluation sets by operational regime (the interceptor's camera looks upward at the sky; terrain is a secondary case) and catch a flipped conclusion: the aggregate metric was dominated by a regime the model already handled fine.

7. **Wilson confidence intervals** reported throughout, and small-sample warnings built into the scorer output (e.g. "fewer than 20 negative frames, false-fire not measurable").

---

## 4. Custom ground-truth labeling tooling

Built an interactive OpenCV-based labeling application (`gt_boxes.py`, several hundred lines) from scratch, iterated through multiple bug-fix rounds based on direct user (my manager's) feedback:

- **Model-free candidate proposals**: a classical small-target detector (motion-compensated temporal median over ±2/4/6 neighbor frames, aligned by global phase correlation to cancel camera pan, plus white/black top-hat filtering, normalized by local clutter) generates up to 6 numbered candidate boxes per frame. Critically, **candidates never come from any model being evaluated** — using a model's own output as its ground truth would be circular.
- **Dual-panel review**: thermal frame alongside the *registered* (homography-warped) RGB frame from the paired day camera, so the labeler can cross-reference a day-camera view of the same instant to resolve ambiguous thermal blobs. Added a full-resolution RGB zoom-tile mode (`t` key) for cases where even the registered thermal-scale RGB crop wasn't clear enough.
- **Click-to-fit and drag-to-draw**: clicking near a target auto-fits a tight box using a polarity-aware local segmentation (handles both bright-on-dark and dark-on-bright targets); dragging draws an exact box when auto-fit fails.
- **Verification/resumability mode**: re-review previously labeled frames with the old box pre-filled, flag "suspect" frames (e.g., a labeled-empty frame adjacent to a labeled-drone frame — a likely click/keystroke error pattern I identified from the data itself), and a `--redo-capped` mode to re-examine every box that had hit an old auto-fit size cap.
- **Sequential-decode frame reader**: discovered that `cv2`'s seek-based frame access (`CAP_PROP_POS_FRAMES`) silently disagreed with sequential decoding by tens to over a hundred frames on some variable-frame-rate source files. Built a `FrameReader` abstraction that always decodes sequentially and caches, with a one-time remap-and-verify tool (`remap_seek_frames.py`, using per-frame MD5 matching) to fix previously-mislabeled data. Verified the fix was byte-exact on two datasets (539/539 and 235/235 frames matched).
- **"Kalman-style" assisted labeling** (added in response to user feedback that manual labeling was slow and error-prone): a constant-velocity predictor draws a search-region circle on the review UI using the nearest already-labeled frames (interpolating between two known positions, or extrapolating from one using estimated velocity), and lists in-gate detector candidates first. Validated this before shipping it: leave-one-out testing on existing hand-labels showed the true location fell inside the predicted circle in 80–94% of cases depending on the video, with a median circle radius of ~30px in a 640×512 frame — i.e. it narrows the visual search to roughly 1% of the frame most of the time.
- **UI/ergonomics fixes driven by direct usability feedback**: when the user reported a reviewed crop was rendering off-screen for large (near, high-resolution) targets, traced it to a fixed magnification factor producing an oversized window, and rewrote the zoom logic to adaptively shrink magnification to fit a configurable max window size while keeping mouse-coordinate mapping correct — verified with a headless synthetic test (large box, small box, confirming output image dimensions) before handing it back.

---

## 5. Synthetic data generation, approach 1: copy-paste compositing

Measured the real detection failure mode precisely before designing a fix: training-set drone boxes had a median size of 40px, while even the drones the existing detector *did* catch in the evaluation video had a median size of ~13px (a >3x, later measured as 6.5x, train/test size mismatch), and the detector's recall on genuinely hard (dark-on-terrain) cases was close to 0%.

Built a copy-paste synthetic data generator (`synth_copy_paste.py`, later generalized into `paste_on_backgrounds.py`) that composites small real-drone sprites onto real thermal backgrounds:

- **Two sprite sources**: real drone crops (extracted from labeled boxes, downscaled — downscaling a large real drone gives a physically plausible small blob with genuine shape structure) and a parametric generator (smooth elliptical body + 1–4 random hot-spot blobs, for shape diversity beyond the limited real drone types available).
- **Physically-motivated compositing**: pastes are composited in *contrast space* (local background + α·contrast), not simple alpha blending, with contrast sampled from the measured real-drone distribution at each target size, a small Gaussian PSF blur to mimic optics, and matched sensor noise injected only inside the pasted region — followed by re-encoding through the same JPEG pipeline as the source images, so synthetic images don't carry a detectable "too clean" signature.
- **Iterated after a failure mode was found**: version 1 (placement anywhere) taught the model "any small bright point = drone" — it started firing on static terrain hot spots. Diagnosed via false-positive contact sheets, then fixed in v2 by restricting placement to visually "sky-like" regions (measured local roughness below a threshold, no pre-existing bright speck nearby) — this made real terrain hot-spots act as implicit hard negatives rather than being paved over.
- **Automated QA tooling**: every generation run produces a numeric manifest of every paste (position, size, contrast, source) and an auto-generated contact-sheet image for visual spot-checking before the data is ever used for training.
- **Background-removal pipeline for new source footage**: when pasting onto new real backgrounds that might already contain a real (unlabeled) drone, built a two-stage removal: an existing detector flags candidate real-drone locations, inpainting removes them — but a later refinement discovered that over-aggressive removal (low confidence threshold, large inpainting masks) was leaving smooth "smear" artifacts on *textured terrain* that didn't match real image statistics; fixed by computing a local high-frequency "roughness" measure around each candidate removal and **dropping the whole background image** rather than inpainting when the removal site sits on texture (rather than smooth sky), since inpainting can't be trusted to be invisible there.

**Lock-step verification engineering**: when later building two parallel experiment arms that needed to differ *only* in background source (real vs. a generative-model translation of the same scene, see §6), engineered the paste placement logic to decide all random choices (position, size, visibility) from the *real* image first, then apply the identical composite operation to the alternate background — and wrote an automated verification that confirmed the two output datasets were byte-for-byte identical when fed the same background, isolating the one experimental variable cleanly.

**Impact (measured with the evaluation methodology from §3):** copy-paste synthetic data produced the first statistically real (exceeding measured seed noise) improvement in the project — roughly **+8 percentage points of recall at matched false-fire rate** on two independent held-out videos, confirmed across multiple experiment generations.

---

## 6. Synthetic data generation, approach 2: generative RGB→thermal translation

In parallel, investigated whether a learned image-to-image model could translate the paired RGB video into realistic synthetic thermal imagery (the two cameras were time-synchronized and spatially registered — see §8), as a complementary or superior data source to copy-paste.

### 6.1 Stream synchronization and spatial registration (supporting infrastructure)

Before any pairing could be trusted, built and validated:

- **Time synchronization** via ego-motion cross-correlation: decoded both the RGB and thermal video streams at low resolution, estimated global frame-to-frame motion via phase correlation (not pixel-value comparison, which is meaningless across sensing modalities), and cross-correlated the two motion-magnitude time series over a lag search range, using multiple independent time windows spread through the recording to confirm a consistent lag (a single match isn't trustworthy — a policy I set explicitly after an early false pairing claim based on one lucky match). Later extended to fit a *linear drift* model when two recorders' clocks ran at very slightly different rates — caught via a near-miss on the fixed-offset tolerance check, not by assumption.
- **Spatial registration**: a day→thermal homography estimated per time-segment (the rig's relative camera pose was found to step discretely over time, ~1–1.5° at points, rather than being fixed — one global homography measurably failed), including a thermal lens distortion model (division model) fit by a coarse-to-fine search over distortion parameters, refined with ECC (Enhanced Correlation Coefficient) alignment on edge maps.
- When a second-generation hardware rig was introduced partway through the project with different relative camera geometry, the existing registration search code silently failed (zero solutions) because it implicitly assumed one camera's field of view contained the other's — diagnosed from first principles (field-of-view ratio arithmetic) and fixed by adding a mirrored search mode for the inverted-containment case.
- **Verification discipline carried through**: every registration result was checked by overlaying RGB-derived edges on the thermal image and visually confirming horizon lines, roads, and other structure coincided — including catching one session where the metric scores looked acceptable in isolation but the user (reviewing overlay images I generated) spotted a dataset shift that the automated metric had missed.

### 6.2 Generative model and go/no-go discipline

Trained a ControlNet-conditioned (RGB → thermal) diffusion model on a frozen Stable Diffusion backbone (only the ControlNet, initialized from the base model's encoder, was trained — this is what made training tractable with a few hundred to ~1,500 image pairs).

Every training run was framed as a **pre-registered go/no-go test**, with numeric pass/fail criteria written down and reasons for each criterion documented *before* the run, not retrofitted afterward:

- **C1 (not a trivial filter)**: the generated image's correlation with raw RGB luminance must be low and close to real thermal's own correlation with luminance — otherwise the model has just learned a greyscale/brightness remap, not real thermal physics.
- **C2 (beats a trivial baseline)**: must out-perform a simple linear-regression colour-to-thermal mapping fit on the same training pairs, measured by L1 and SSIM on held-out test pairs. "If it can't beat a linear brightness map, it learned nothing thermal."
- **C3 (qualitative)**: correct thermal physics on inspection — cold sky gradient, bright clouds matching RGB cloud shapes, drones rendered with correct (bright) polarity.

**Result arc across iterations** (each a distinct, documented experiment):
1. First model (pix2pix architecture): passed on same-day held-out data, but **failed outright (C1 failure) when tested on a genuinely unseen day** — quality collapsed on footage of different terrain content.
2. Diagnosed the *root cause* before trying to patch it: wrote a content-coverage analysis tool that classified every training/test image pair by RGB sky-fraction, and found the training set contained **zero terrain-dominated scenes** while the failing test day was 95% terrain/mixed. The real failure mode was content coverage, not "unseen day" per se, and not a capacity or architecture problem (ruled out via validation-loss-still-improving and the the fact that only a small ControlNet adapter was being trained on top of a frozen, far larger base model).
3. Second-generation diffusion model, retrained with terrain-containing data: **still failed** the "beats linear baseline" criterion (C2) even on data covering the missing terrain content — it only tied a trivial per-pixel colour formula. Applied the pre-committed exit rule: *"if it fails even with content covered, stop investing in translator pixel-quality at this data scale"* — explicitly declining to chase further architecture or post-processing fixes once the pre-set bar wasn't met, to avoid open-ended effort with no clear stopping point.
4. **Ran a cheap falsification test before fully abandoning the direction**: applied standard camera-style image enhancement (CLAHE contrast enhancement, unsharp masking) to the model's raw output with zero fitting to the target images, to check whether the failure was merely "needs post-processing" rather than structural. Result: texture *energy* could be matched this way, but edge-placement accuracy didn't move and pixel error got *worse* — confirming the missing information (fine thermal texture driven by surface-emissivity detail not present in the 7x-downscaled RGB input) could not be recovered by post-processing, only by better training data or a fundamentally different approach.
5. **Reframed the evaluation question itself**: recognized that pixel-space metrics (L1, SSIM) are the wrong judge for a *data generator* whose real purpose is downstream detector training — a generator can produce plausible-but-misplaced texture that scores poorly on pixel metrics while still being useful training signal. Designed a **decisive detector-in-the-loop experiment** instead: two otherwise-identical synthetic training sets, one pasting real drones onto *real* thermal backgrounds, one onto the *same scenes* translated by the generative model, holding every other variable fixed (the lock-step engineering from §5) — letting the actual downstream detector decide, rather than a pixel metric.

### 6.3 Detector-in-the-loop verdict

- **On backgrounds from the model's own training distribution**: the translated-background arm **tied** the real-background arm on detector performance (within measured noise) — a genuinely informative null result after the earlier pixel-metric failures, since it showed the generator wasn't *useless*, just not better than real data where real data existed.
- **Scaled up to translate genuinely novel, out-of-training-distribution RGB footage** (different source videos, different visual content — heavy cloud cover, farmland, a city skyline) as new synthetic backgrounds: this was a **clear regression**, not a tie — the resulting detector's false-positive rate roughly doubled on completely real, untranslated evaluation video. Diagnosed the likely cause: content far from the generator's training distribution produces plausible-but-wrong thermal textures/artifacts that happen to resemble the pasted-drone signal closely enough to teach the detector to fire more broadly.
- **Final, well-evidenced conclusion** (two independent confirmations: the pixel-metric failure and the detector-in-the-loop regression): the generative translator is safe to use only on content resembling its own training distribution, where it adds no measurable value over real data, and is actively harmful extrapolated beyond it. Fully closed out this research direction with a clear rationale, redirecting effort to real-data collection and the approaches in §5 and §9.

---

## 7. External dataset evaluation

Surveyed and rejected (with specific, documented reasons — not a blanket "didn't look") every candidate public dataset suggested as an alternative data source:

- Multiple agricultural/farmland RGB-thermal datasets (tens of GB each): thermal and RGB were captured as **separate, unpaired, differently-dated flights** despite superficially living in the same dataset — confirmed by checking image counts and filenames, not just folder names.
- A German urban thermal dataset: correct image content type but captured at night in early March with no solar heating — wrong underlying physics for a daytime, sun-heated-terrain use case.
- A building-facade RGB-thermal dataset: correctly paired and aligned, but wrong scene type (close-range walls vs. aerial oblique terrain).
- A Caltech aerial RGB-thermal dataset: the only one that actually matched the content need (natural terrain, synchronized/rectified pairs) — but the license (CC BY-NC, confirmed by checking the actual data repository rather than trusting a secondary internal document that had the license wrong) restricted it to research-only use, ruling it out for this use case.
- Checked whether an internal RGB-only detector's existing detections on other recorded flights could be reused as labels by proxy — investigated via frame-content matching (did two candidate video files actually show the same flight) and concluded, with quantitative evidence (frame-similarity scores indistinguishable from random), that they did not, closing off a potential shortcut rather than quietly assuming it would work.

**Conclusion, reached independently for each dataset and confirmed consistent**: the specific missing content (sun-heated natural terrain, viewed obliquely, by this camera's own sensor processing characteristics) existed only in this project's own flight data — external data could not substitute.

---

## 8. Real-label mining from a secondary RGB detector

Once a production-grade RGB-only drone detector existed (owned by a teammate), engineered a pipeline to turn its detections on the *day* camera into labeled training data for the *thermal* camera, exploiting the stream synchronization and registration already built:

- **Geometric label transfer**: map an RGB detection box through the registered homography and lens-distortion model into the corresponding thermal frame. Validated with a round-trip self-test (map a known thermal box forward to RGB and back) on 182 boxes — recovered the original location with a median corner error of 1px.
- **Dual-detector agreement as an automatic precision filter**: rather than trusting the RGB detector's output directly (measured precision as low as 10% on some footage — mostly false alarms, not drones), only accepted an RGB-derived label as "confirmed positive" when the thermal detector *also* fired on the same location. This is a form of co-training / pseudo-labeling with a precision guarantee, which I validated empirically before trusting it: on a sample of **629 human-reviewed boxes**, this agreement rule achieved **99.5% precision** at a chosen confidence threshold, while still retaining ~90% of the true labels — turning an otherwise-unusable ~2,000 raw detections into ~550 high-confidence automatic labels without additional human review.
- **Found and fixed a silent labeling bug through this process**: the secondary detector's output format had a third classification outcome (besides "confirmed"/"rejected") for cases where its first stage was confident enough to skip a second verification stage entirely — my label-transfer code only recognized the two documented outcomes and was silently discarding every label in the third category. Diagnosed via systematic visual spot-checking (zooming into claimed label locations across multiple confidence levels and finding *no visible target in any of them* — a strong signal something structural was wrong, not just noisy data) before concluding it was a real bug rather than legitimate low-quality detections, then fixed and re-verified with the same visual check (now finding plausible drone targets consistently).
- **Human-in-the-loop verification tooling**: for borderline cases, built a lightweight accept/reject review tool (separate from the main labeling tool, since the data shape was different — single candidate box per frame rather than multiple) with keyboard shortcuts, resumable state, and frame-by-frame playback for ambiguous (moving-target vs. sensor noise) cases.

**Measured impact**: this real-label pipeline produced the project's largest clean improvement when combined with a model architecture change (§9) — the resulting model showed a statistically real (seed-noise-exceeding) gain on the primary evaluation video.

---

## 9. Detector architecture and training experiments

Ran a structured sequence of controlled experiments, each isolating exactly one variable from the previous best configuration, with results tracked in a running comparison table across ~15 model variants over the project. Selected highlights:

- **Found and fixed a silent training confound**: a popular training framework's "auto" optimizer-selection setting silently switched between two different optimizers (and learning rates) depending on dataset size — meaning two experiments that appeared to isolate "does more data help" were actually also changing the optimizer. Diagnosed by inspecting training logs across runs, not assumed. Fixed by pinning the optimizer explicitly in every run from that point forward, and flagged all earlier comparisons made with the confound present as unreliable.
- **Isolated the effect of a company-internal, RGB-domain-pretrained backbone** (a drone detector trained on visible-light imagery) as a thermal-model initialization, versus a generic/COCO-pretrained backbone — found the RGB-pretrained initialization gave a consistent, large improvement on the hardest evaluation subset (dark drones against terrain, the project's historically worst case): roughly 2–4x relative improvement, holding up across multiple random seeds.
- **Diagnosed an architecture-specific small-object weakness**: a newer YOLO-family architecture (without an extra high-resolution detection head) under-performed an older, smaller architecture specifically on small sky-background targets despite otherwise-superior capacity — traced to its detection-head resolution, confirmed by adding a small-object ("P2") detection head to the newer architecture, which recovered (and then exceeded) the missing performance.
- **Caught and rejected a result from a failure-mode analysis, not just a benchmark number**: a configuration that scored best on two of five evaluation metrics was found, on inspecting its actual false-positive detections, to be firing on the aircraft's own airframe edges, ground-support structures, and cloud fragments at full image resolution — a specific overfitting signature invisible in the aggregate recall/false-fire numbers alone. Rejected the configuration despite its favorable headline numbers, and recorded the diagnostic images and reasoning in the project log for future reference (e.g., checking two specific frames as possible existing label errors it surfaced).
- **Final adopted model** combined: an RGB-pretrained backbone, a small-object detection head, and the mined real-label data (§8) — validated across 2–3 random seeds, beating the previous best model on every evaluation subset by a margin exceeding the measured per-subset seed-noise range (i.e., a result I was confident was real, not noise), including a >10 percentage point gain on the hardest held-out video.
- **Tracked and reported the compute-cost trade-off honestly alongside the accuracy win**: the adopted model used roughly 3x the parameters and compute of the previous architecture — noted explicitly as a cost not yet validated against the target embedded hardware's real-time constraint, rather than treating the accuracy win as a free upgrade.

---

## 10. Response to a mid-project hardware/domain shift

Partway through the project, new recordings arrived from an upgraded thermal camera — the camera planned for actual deployment, which the team confirmed was a different sensor/lens combination from everything trained on so far. Rather than assuming the existing pipeline would transfer, I:

- **Quantitatively characterized the domain shift** before doing anything else: compared the two cameras' angular resolution (derived from the registration geometry, not a datasheet) and found the new camera had **~2.2x more pixels per degree of field of view** (confirmed by a teammate's independent visual observation, which matched the measurement) and measurably softer edges/different image statistics.
- **Re-ran the full synchronization, registration, and false-sensor-overlay-detection pipeline** from scratch for the new recordings, discovering and handling several new data-quality issues specific to the new hardware: corrupted recording files, a sensor-specific color overlay burned into some frames that needed automatic detection and exclusion from training (an initial overlay-detection heuristic over-fired on ordinary image noise and had to be tightened based on a visual QA pass), and video container metadata reporting an incorrect frame rate that silently caused frame-duplication during decoding until caught by comparing decoded frame counts against the container's true packet count.
- **Built a new human-verified held-out evaluation set specifically for the new camera** before training anything, recognizing that the old evaluation videos (different camera) could only serve as a regression check, not a measure of real target performance.
- **Set up a background-automated monitoring and reporting loop**: configured a scheduled background process to check training/evaluation job status roughly every 25 minutes, automatically score newly-completed model runs against pre-registered pass/fail criteria the moment their outputs existed, update the project log, and send a status email — so that multi-hour unattended training runs on a cloud GPU instance produced timely, criteria-based decisions rather than requiring constant manual polling, while avoiding notification spam by only reporting on genuinely new results.
- **Adapted the synthetic data pipeline to the new domain** (increasing the new camera's data weight, generating a "zoom" augmentation that resamples old-camera training images to approximate the new camera's angular resolution, mining additional hard-negative training examples — specifically **false-positive detections from the RGB detector at a deliberately low confidence threshold**, per a direct engineering suggestion from leadership, to build a harder negative-mining set rather than relying only on naturally-occurring negatives), each evaluated as its own isolated experiment against the new held-out ground truth before being adopted or discarded.

---

## 11. Infrastructure and tooling

- **Cloud GPU lifecycle management**: scripted, idempotent start/stop of a cloud GPU instance (explicit cost-consciousness — never left running idle, start/stop verified by reading status back rather than trusting command output, after once losing an entire overnight work window to an unverified start command).
- **Multi-stage training queues**: bash orchestration scripts chaining dataset-build → train → inference → evaluation steps sequentially on a single GPU (to avoid idle time or memory contention from parallel jobs), with file-based "done" markers so a monitoring process could safely detect stage completion and a watchdog auto-shutdown to bound worst-case cost if anything hung.
- **Mid-flight queue surgery**: on at least two occasions, needed to change the remaining experiment plan while jobs were already mid-run on the remote machine (responding to new stakeholder priorities) — did this live by identifying and killing only the specific orchestrating process (not the training subprocess it had spawned, which could safely keep running) and launching a revised queue script, verifying the GPU stayed continuously utilized throughout the swap.
- **Reproducible dataset and run provenance**: every generated dataset ships with a JSON config recording every parameter used to build it; every notebook entry documents exact commands, not just results, so any number in the project history can be regenerated.
- **Automated status reporting**: a lightweight email-notification helper (SMTP, app-password auth) for sending concise, infrequent progress updates during long unattended runs, used in combination with the scheduled monitoring loop above — deliberately rate-limited ("don't spam") by only emailing on genuinely new, scoreable results.
- **Internal stakeholder-facing deliverables built directly from the pipeline's own provenance metadata**: packaged a subset of the mined/generated training data (confirmed detections, mined hard negatives, held-out test crops) into a self-contained, documented folder — with an explanatory README written for a non-project audience explaining exactly how each image category was produced and its reliability — so a manager/stakeholder could inspect or independently retrain on the same data without needing the full pipeline context.

---

## 12. Headline results (all numbers measured, mean recall at 1–25% false-fire unless stated)

**Thermal detector on the deployed (new) camera, two held-out recordings with human ground truth (22 and 122 drone frames), 2 training seeds each:**

| | recording 08_58_15 | recording 10_15_59 |
|---|---|---|
| previous best (YOLO26s-P2, RGB-pretrained backbone) | 62.6 | 45.5 |
| **adopted model (M0): + new-camera labels + pastes on new real backgrounds** | **92.7** | **58.0** |

- Gain **+30 / +12.5 points**, confirmed with a second seed; run-to-run spread on the new camera fell from 15–17 points to 2–4.
- Cost: 0730 old-camera video −14 points, 0708 −3; the old camera is not the deployed one.
- Variants tested and **not adopted** (pre-registered rule: ≥ 5 points on both sets): ×3 weighting of new-camera data + new-camera sprites (M2), + human-reviewed boxes (M4, a tie on the new camera and better on old-camera videos), initialisation from the fine-tuned RGB weights (M6, worse on old-camera videos).
- Progression on the old camera (07-02 test video, sky drones, recall at 10% false-fire): 55% (pinned-optimizer baseline) → 77% after copy-paste data, real small-drone labels, the RGB-pretrained backbone and the P2 head.

**RGB detector (fine-tune of the company YOLO26s on Fixed_wing_v3, 20,281 images, + 995 new-camera tiles, 40 epochs, about 3.7 minutes per epoch on one L4):**
- v3 test, drone class: mAP50 0.973 → 0.978, mAP50-95 0.730 → 0.774 (tighter boxes); airplane and bird improved as well.
- 100 held-out new-camera drone tiles: recall 96.9% → 93.5%, precision 95.1% → 96.9%; at a matched number of false alarms on 190 drone-free tiles it is not better (the test favours the old model, which picked those tiles). Reported as such, not as a win.

**Image generator on the new camera (ControlNet continued on 1,503 pairs from 18 sessions):**
- Raw pixel error versus a linear colour baseline on the 218 held-out-session pairs: 1.18x (old model) → 1.03x; the pre-set bar was ≤ 0.85x, so it is a NO-GO as written.
- After removing per-image brightness (the camera's automatic gain), it beats the baseline on sky scenes by 37% (0.062 vs 0.098), edge correlation 0.30 vs 0.27. It learns structure but not absolute level.
- Min-SNR loss weighting: tie. The limit is data, not training stability.

---

## 13. Findings that changed the plan late in the project

- **Label audit with model output as a probe:** inspecting the false alarms of the best model on empty frames of a held-out recording showed 16 of 21 were one small bright dot tracked across 14 frames labelled "nothing" — the drone itself. A missed-label pattern that only surfaced because the evaluation looked at *what* the errors were, not only how many. Ground truth was left unedited and sent back for re-review; the sensitivity of the ranking to those frames was reported (ranking among the top models stayed undecided, gap to the older baseline stayed).
- **A symlinked image folder silently made the trainer read the source labels (old class order):** caught within one minute of launch by reading the label-scan line of the training log, before any GPU time was wasted; fixed with hard links and a unit test.
- **A "dataset" mismatch caught by reading the checkpoint metadata:** the production RGB model recorded its own training set (an older dataset version), which corrected an assumption about what the fine-tune started from.
- **Counting labels with `cat` merges files without a trailing newline;** two of my own early dataset counts were wrong because of it and were corrected against a per-file count.

---

## 14. Working style and judgment demonstrated throughout

A few patterns that recurred across the whole project and are arguably the most transferable part of it:

- **Pre-registering success criteria before running an experiment**, every time, and actually honoring an exit rule when the criteria weren't met rather than rationalizing a near-miss as a win.
- **Treating a surprisingly good result as a bug to investigate, not a success to report** — this caught the original data leak, a mislabeled-class bug, a frame-indexing bug, and a training-confound bug, all of which would have otherwise produced confidently wrong conclusions.
- **Quantifying noise before trusting a signal** — the seed-variance investigation was unprompted and changed the project's entire decision-making bar afterward.
- **Diagnosing root causes instead of patching symptoms** — e.g., tracing a generative model's failure to a specific, fixable data-coverage gap rather than tuning hyperparameters against a vague "it doesn't work," or tracing a false-positive spike to the exact pixel content it was triggered by.
- **Converting every major finding into a reusable tool** (background classifiers, drift detectors, agreement-based auto-labelers, assisted-labeling UI) rather than one-off scripts, so the pipeline's capability compounded over the project instead of resetting with each new dataset.
- **Honest, specific reporting upward**: every status update to the project owner included explicit uncertainty (confidence intervals, "this is inside measured noise," "this makes it worse on X even though it helps Y"), compute/cost trade-offs stated plainly, and corrections issued proactively when an earlier number turned out to be wrong (e.g., a mis-counted dataset statistic, a timeline error caused by an infrastructure mistake) rather than left uncorrected.
