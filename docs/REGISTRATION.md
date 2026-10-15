# Image registration in this repo (day camera ↔ thermal camera), and what carries over to map ↔ camera matching

Written for: Daniel and the team building the map-to-camera matcher. Code: `day2thermal/video_pairs.py` (`calibrate`, `extract`), `detection/src/data/{verify_stream_sync,sync_ts,make_shared_registration}.py`.

## 1. What it does, and what it is not

It aligns two cameras on one rigid rig that look at the same scene at the same time: a 4K day camera and a 640×512 thermal camera. It finds, for each stretch of the recording, a 3×3 matrix H that maps day pixels to thermal pixels, plus the thermal lens distortion.

- It matches **edges**: the horizon, roads and field borders, buildings, tree lines and cloud edges. Nothing is detected semantically. There is no "find the streets" step, and streets work only because they make strong edges.
- It is a **camera-to-camera** registration, with an approximately known scale and nearly the same viewpoint. It is not a map-to-camera localiser (section 5 says what that needs).
- Time alignment comes first (section 2). The geometry is solved only on frames where the rig is nearly still.

## 2. Pipeline

```
 sync (ego-motion)  ->  pick still frames  ->  per frame: lens-undistort thermal, coarse edge-NCC search, ECC refine
        ->  choose the lens coefficient  ->  segment in time  ->  verify  ->  registration.json  ->  extract / label / map boxes
```

### 2.1 Clock offset: `verify_stream_sync.py`, `sync_ts.py`
Pixel similarity cannot compare RGB with thermal, but both streams shift together when the rig pans. So:
1. Decode both videos small and estimate the global shift per frame by phase correlation.
2. Put both velocity signals on one grid (the `.ts` version uses each frame's own `pts`, since the day stream's frame gaps vary by 31–35 ms).
3. Cross-correlate over a lag range.

An offset is accepted only if several independent windows agree with the global peak, the peak clears the sidelobes, and the lag does not drift. The same regression gives the thermal/day pixel-scale ratio, which seeds the next step.

### 2.2 Which frames to solve
`pick_still_frames`: from the ego-motion cache, frames in the quietest 40% of the recording, spread evenly (default 100 frames). Motion blur and a small timing error would otherwise look like a geometric offset.

### 2.3 Per-frame solve (`solve_frames`)
1. **Lens.** Undistort the thermal frame with a division model: `x_u = c + (x_d − c) / (1 + λ r_d²)`. It has a closed form in both directions. λ < 0 is barrel distortion.
2. **Edge maps** (`edge_map`): Gaussian blur, Sobel magnitude, normalised by the 99.5th percentile. Edges are the only structure the two modalities share.
3. **Coarse search** (`coarse_search`, or `coarse_search_inside` when the thermal view lies *inside* the day view, as on the Boson rig):
   - Shrink the day frame by the seed scale ± 18% (9 values) and rotate it ±5° (7 values).
   - Slide one edge map over the other with normalised cross-correlation (`cv2.matchTemplate`, `TM_CCOEFF_NORMED`) in a ±90 px window.
   - Keep the best similarity. The peak margin, score minus the best score elsewhere, measures how unique the match is.
4. **Refinement** (`ecc_refine`): ECC (`cv2.findTransformECC`) on edge maps at 0.25× day resolution, affine by default, 200 iterations. It returns H (day full-res → undistorted thermal) and the ECC correlation `cc`.

### 2.4 Choosing the lens coefficient
`calibrate` solves a subset of frames for each λ in {−0.075 … −0.2}. The right λ is the one at which **temporally adjacent solutions agree most** (their H differ by ≤ 3 px over a grid of day points). A wrong λ makes H depend on where the structure is in the frame. For the Boson rig λ = −0.1: 84% of adjacent pairs agree, against 64% at −0.075, 75% at −0.125 and 22% at −0.15.

### 2.5 Segmenting in time (`segment_timeline`)
The day camera is **not** rigid with respect to the thermal over a whole recording: the relative pose steps by about 1.5° and 13 px between stretches. So:
- Solutions with `cc` < 0.2 are dropped.
- A solution that disagrees (> 3 px) with the running segment starts a new one only if the *next* good solution agrees with it. A lone disagreement is an outlier (near-field parallax or a bad fit).
- A segment is **verified** with at least 3 agreeing frames. Its H is the median of its members' mapped footprint.
- The gaps between segments are excluded.

### 2.6 Verification
It uses the exact remap that `extract` will use:
- Per-cell residual by phase correlation in a 4×3 grid.
- Edge NCC per frame.
- Verdict PASS needs the segments to cover at least 50% of the recording, the median deviation from the own segment to be ≤ 1.5 px, and every cell residual ≤ 2 px.
- Overlay previews are written for human review.

### 2.7 Sessions that cannot calibrate (`make_shared_registration.py`)
Featureless sky and short clips give no edges to match. The Oct-06 rig geometry measured on well-textured sessions is reused: 26 segments over 6 sessions agree on scale 0.3005 ± 0.001 thermal px per day px, rotation 0.15° ± 0.1° and translation (−253, −41) ± 2 px. These sessions are marked `SHARED_H`, and every extracted pair is still checked by edge NCC.

## 3. How the result is used
- **Pairs for training** (`extract`): warp RGB into the thermal geometry, so thermal pixels are never resampled. Keep a pair only if the scene changed enough since the last one and the thermal frame is not a freeze.
- **Boxes across cameras** (`build_boson_labels.py`): a day box goes to thermal by undistortion and H, and the other way by the inverse. The check was 2.6 px median error on 103 day/thermal co-detections for 08_58_15.

## 4. What this relies on, and where it fails
- A rigid rig and a known approximate scale.
- Edges present in both modalities. The method fails on blue sky and works on horizon, roads, buildings, trees and cumulus edges.
- A distant scene, so parallax is small. Near-field ground gives residuals.
- A calm moment (sync and still-frame selection).
- A known time offset. Without it, the geometry cannot be separated from the timing.

## 5. Map ↔ camera: what carries over and what is missing
The user's goal: "given a map and a camera input, decide whether it is the same place".

**Carries over:**
- The **edge-NCC coarse search followed by ECC refinement**, as the verification stage once a candidate pose exists.
- The **lens model**, solved by agreement between neighbouring frames, instead of calibrating it separately.
- **Temporal segmentation and outlier rejection**: a single frame never decides alone. Require agreement across frames before trusting a match.
- **Quality gates**: peak margin, ECC cc, per-cell residual, edge NCC. They give a score for "same place or not" instead of a forced answer.

**Missing:**
1. **Viewpoint.** A map is a nadir, roughly orthographic view. The camera looks obliquely, so the relation is a perspective warp that depends on altitude, pitch, roll and terrain relief. Here the two cameras look the same way, so a similarity plus ECC was enough. For map matching, first warp the camera frame to nadir using the pose estimate (IMU/GPS, altitude, camera intrinsics), which also seeds the scale, and then match. Terrain relief needs a DEM or a flat-ground assumption with a residual allowance.
2. **Global search.** Our search covers ±18% scale, ±5° rotation and ±90 px. The map case starts from a GPS prior with an error of tens to hundreds of metres, and a heading that the IMU may know. It needs a coarse-to-fine pyramid, a heading search, and a candidate list instead of one best score.
3. **Appearance gap.** A map is a rendering or satellite image from another season, light and time. Roads, field borders, coastlines and building outlines survive; texture and shadows don't. Our gradient-magnitude maps tolerate this less than a learned descriptor.
4. **Few edges.** Over sea, desert or sky the method has nothing to match (we faced this on blue sky). The matcher should say "no structure" instead of returning a number. `structure_frac` does that here.
5. **Repetition.** Field grids and regular roads give several equally good peaks. The peak margin is the first defence; combine it with the motion prior from the flight path.

**Reasonable build:** use our approach as the final verification, and put a learned local-feature matcher or a road/coastline segmentation chamfer in front of it for the global step; or use ours directly with a pose prior to test the "same place" hypothesis.
