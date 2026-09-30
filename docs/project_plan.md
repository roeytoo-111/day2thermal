> **Status (2026-09-30):** superseded for day-to-day work by [`notebook/`](../notebook/README.md) (current state and plan: `notebook/2026-09-30.md` §9). Kept for the roadmap and approach comparison below.

# Thermal Detection — Generation & Detection Plan

Working plan for the day2thermal / drone-thermal-detection effort. Goal: generate usable synthetic thermal training data, then train/benchmark a detector on it.

---

## Roadmap

| # | Step | Notes |
|---|------|-------|
| 1 | Resolve the current data issue | Run `check_video_size.py`; check candidate datasets (MMFW-UAV, LRDDv3) for domain gaps and actual relevance before committing to either |
| 2 | Analyze current data and current GAN output | Understand domain gaps, failure modes beyond overfitting, and what to take into account when choosing new data (see Ground truth section, below, for the specific checks) |
| 3 | Train (or run current YOLO/RF-DETR model) on current data to get a baseline | Establishes the number everything else has to beat |
| 4 | Analyze MMFW-UAV (or other outsourced data): decide on a smaller subset for training | Make sure the domain is diverse, matches our domain, and the subset is balanced |
| 5 | Split up and work on image generation in parallel | Phase 1 approaches only for now, see **Candidate generation approaches**, below |
| 6 | Re-run YOLO/RF-DETR testing, benchmark against baseline | Use metrics from the research report + any new ones we define |
| 7 | Test knowledge distillation on the YOLO/RF-DETR model to improve it further | Use metrics from the research report + any new ones we define |
| 8 | Write the full pipeline | Candidate idea: combine the classic contrast-filtering paper (DVWELCM) for long-range/small targets with YOLO for closer targets, using the classic method to extract ROIs and handing off to YOLO/RF-DETR once targets are large enough to have discernible shape |

---

## Ground truth: what we can use it for

1. **Training loss.** Standard supervised signal, L1/perceptual/etc. between generated and real thermal. Current GAN already does this; any future candidate (ControlNet, LoRA-ControlNet) would use it the same way.
2. **Evaluation ground truth (distinct from #1).** A slice of real pairs held out and *never* trained on, reserved purely to score finished models (Tier 1: L1/RMSE/PSNR/SSIM vs. real thermal). Scoring a model on pairs it trained on measures memorization, not generalization, with only ~316 real pairs total, we need to explicitly fix which slice is eval-only (e.g. the existing contiguous val chunk) and keep it identical across every candidate model so the bake-off numbers are actually comparable.
3. **Domain-gap diagnostic for external datasets.** Use our real pairs to characterize what the RGB↔thermal relationship actually looks like on our footage (intensity ranges, correlation structure), then check whether MMFW-UAV / LRDDv3 samples look statistically similar before sinking time into registering or training on them.
4. **Correlation inspection.** Two distinct checks:
   - *RGB-luminance vs. thermal-intensity correlation, per real pair* (Pearson correlation, pixel-by-pixel, plus scatter/joint histograms across pairs). Tells us how "cheatable" the mapping is on our own footage, consistently very high correlation everywhere would mean a lot of the signal is recoverable from brightness alone (reframing this more as a registration/volume problem); correlation that collapses in specific regions (metal vs. foliage vs. sky) tells us exactly where real thermal physics diverges from a naive brightness mapping, i.e. what a generator actually has to learn rather than shortcut.
   - *Generated-vs-real correlation on the held-out eval set*, as a residual/error map rather than one aggregate number, and specifically overlaid against the drone bounding box region. Error concentrated on background costs little for detection; error concentrated on the drone itself directly threatens Tier 3 (mAP), so this version of the check is the one that predicts whether generation quality will actually matter for the real task.
   - *(Optional third check)* Temporal stability, track these correlation stats frame-by-frame across a sequence instead of per-image in isolation. Drift would point at calibration/registration decay over time rather than a modeling problem.
5. **Reusable calibration.** Once registration is sorted, the homography/registration parameters derived from this footage should be reusable on future flights with the same rig, a one-time cost, separate from anything about generation or loss.
6. **Real detector training/eval data in its own right.** Independent of generation entirely: our real paired thermal frames are legitimate data for YOLO/RF-DETR too, both for eventual real+synthetic blending (Tier 3) and as the only trustworthy source for a final real-world eval set.

---

## Data

Two candidate datasets uploaded to the drive:

| Dataset | Size | Notes | Paper | Data |
|---|---|---|---|---|
| **MMFW-UAV** | ~147k images (147,417 total, split across zoom/wide-angle/thermal sensors) | Air-to-air, but **fixed-wing UAVs only**. Captured via 3-sensor gimbal, so still needs registration. Available now. | [nature.com](https://www.nature.com/articles/s41597-025-04482-2) | [scidb.cn](https://www.scidb.cn/en/detail?dataSetId=edb443cfcfa142019499bbfd68542cc9) |
| **LRDDv3** | ~29.6k IR images paired with RGB | Drone-detecting-drone (same problem framing as ours), long-range focus, RGB/thermal already roughly co-registered. Access requested ~1 week turnaround expected. | [arxiv.org](https://arxiv.org/html/2605.25942v1) | [drexel.edu](https://research.coe.drexel.edu/ece/imaple/lrddv3/) |

**Interim strategy:** Start with MMFW-UAV in the meantime and see where that gets us, check for domain gap and actual relevance to our target drones before investing further.

**Production data path:** the above datasets, since they are non-commercial, are good for R&D/testing only, production training needs different data. Candidate: simulate our own with **AirSim**, it has a working thermal/IR camera pipeline, but it requires manually building per-environment temperature/emissivity/camera-response data before capture (not a flip-a-flag feature, real setup work per scene). Simulated thermal output should go through the same domain-gap validation as MMFW-UAV/LRDDv3 (see **Ground truth**, above) before being trusted, physically-modeled doesn't automatically mean realistic enough to close the gap.

---

## Candidate generation approaches

All keep Roey's existing inference interface (RGB image [+ scalar temperature] → thermal image); no text conditioning needed since that's not how the current pipeline is used.

Split into two phases to avoid over-committing compute/time before we know if the cheap options are enough.

### Phase 1 — starting now

| # | Approach | Notes | Ideal Pairs | Compute / GPU Type | Est. GCP Cost (Total Training) |
|---|---|---|---|---|---|
| 1 | **Current GAN model** (Roey's pix2pix / two-stage ThermalGAN) | Existing baseline; keep improving once the data issue is resolved. Fast inference, but lower physical accuracy. | 1,000 – 5,000 pairs | 1x L4 (24GB) | $8 – $15 (~12-20 hours on an L4 instance at ~$0.70/hr) |
| 2 | **LoRA-adapted ControlNet** (CtrLoRA) | Cheaper, designed for low-thousands-of-images regimes — a closer fit for our realistic near-term data volume. | 2,000 – 10,000 pairs | 1x L4 (24GB) or 1x A10G | $15 – $40 (~24-48 hours on an L4 instance at ~$0.70 - $0.85/hr) |

### Phase 2 — on hold, revisit only if Phase 1 results are underwhelming

| # | Approach | Notes | Ideal Pairs | Compute / GPU Type | Est. GCP Cost (Total Training) |
|---|---|---|---|---|---|
| 3 | **Full ControlNet fine-tune** on a pretrained Stable Diffusion backbone | Higher ceiling, needs more data/compute. RGB is fed as a spatial condition, same mechanism as depth/canny ControlNets. | 10,000 – 20,000 pairs | 2x to 4x A100 (80GB) | $345 – $830 *( GCP A100 80GB on-demand is ~$5.07/hr, ~$2/hr on Lambda, ~$1.49/hr on RunPod/Jarvislabs* |
| 4 | **Flow-based generative model** (ThermalGen) | State-of-the-art on published benchmarks, but trained on 8+ combined datasets — treat as a reference/benchmark rather than something to fine-tune from scratch at our scale. | 30,000 – 100,000+ pairs | 4x to 8x A100 / H100 | $1,200 – $3,000+ (multi-day multi-node run, out of scope for current infrastructure) |

**Rationale:** #1 and #2 are the lowest-cost, lowest-risk options and directly testable against our current (small) real dataset — start there, get real Tier 1–3 numbers, and only escalate to #3/#4 if those results don't clear the bar. #3 and #4 both assume data volumes we don't have yet (10k+ and 30k+ pairs respectively) — revisit once the MMFW-UAV/LRDDv3 domain-gap check and any registration fix land, since that's what determines whether either is even viable.

---

## Benchmark metrics

From the research report, plus anything else we agree on at the meeting:

- **Tier 1 — Pixel accuracy:** L1, RMSE (°C where calibrated), PSNR, SSIM
- **Tier 2 — Visual realism:** FID/KID + human perceptual A/B
- **Tier 3 — Task impact (the one that actually matters):** train detector on (real + synthetic) vs. real-only, measure mAP lift on an unseen/held-out real flight