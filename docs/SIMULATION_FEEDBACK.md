# Simulator renders: what the pictures show, and which parameters to vary

Written for: Daniel and whoever runs the simulator. Source: the 7,120 `img_*` renders in Fixed_wing_v4 (1,495 distinct pictures, ~5 Roboflow copies each, two frame sequences 0–499 and 1–1000), 5,792 uav boxes. Statistics: `detection/src/data/rgb_audit.py` (train split, sim vs everything else). Colour numbers compare the renders with the 2026-10-14 flight tiles. Whether the renders *help* is being measured by the ablation in notebook 2026-10-15 §8; this document is about the pictures.

## What the renders do well
- **Small targets:** median drone 16 px (10th–90th percentile 8–52 px) at 640. The real data median is 104 px. 78% of the small (< 16 px) drones in the whole training set are simulated.
- **Backgrounds:** a good mix of sky over cumulus and look-down terrain (fields, roads, desert, hills, sea), with sun in a few frames and one greyscale frame.
- **Low contrast:** drone against its surroundings median 9.6 grey levels (60% below 12), against 15 for the rest.

## Where they differ from our real flights
| | simulator | real (2026-10-14 flights unless stated) |
|---|---|---|
| Sky colour | saturated blue (mean HSV saturation 46, blue-minus-red 15.6) | hazy grey-white overcast (saturation 13, blue-minus-red 2.9) |
| Clouds | crisp, fair-weather cumulus with sharp edges | stratus and haze, edges soft or absent |
| Distance haze | little: far terrain keeps its colour | strong: horizon fades into the sky |
| Lighting range | almost no dark or bright frames (0.8% below mean luminance 60, 0.7% above 190) | 3.9% dark and 7.0% bright over the whole real set |
| Drone appearance | dark grey/black silhouette, one airframe shape per sequence | both: dark silhouettes and bright white ones where the sun hits the wing, depending on the sun angle |
| Background texture | 34% of drones over textured background | 73% |
| Other flying things | 66 bird and 9 airplane labels in 7,120 images | birds, manned aircraft with contrails, kites |
| Image path | clean | H.264 at 4K, downscaled ×6, then sensor noise and auto-exposure |

## Parameters to vary, in order of expected value
1. **Haze / atmospheric extinction (visibility 3–30 km).** The biggest gap. It makes distant drones low-contrast, which is our measured failure (small drones against cloud and overcast).
2. **Cloud cover:** thin cirrus, full stratus overcast, and no clouds, in addition to the cumulus. Real flights are mostly overcast.
3. **Sun azimuth and elevation relative to the camera**, including back-lit (silhouette against a glaring sky) and front-lit (bright wing). Include low sun and dusk.
4. **Airframe colour, albedo and specular coating:** white, grey, black, orange; matte and glossy. Specular glints make a drone bright at some angles.
5. **Drone variety:** flying wing, conventional fixed-wing, VTOL, and also **manned aircraft** (airliner at 5–15 km, light plane) with contrails, plus birds and kites, as separate labelled classes. Distant manned aircraft are the second weakest class.
6. **Look-down terrain:** coastline, buildings and road grids at different heights and seasons; keep both.
7. **Camera model:** auto-exposure changes, vignetting, lens flare, H.264 compression, sensor noise and 4K → 640 downscale blur applied after rendering. Rendering at 4K and downscaling like deployment is the cleanest.
8. **Frame content:** include frames **without a drone** (hard negatives: clouds, contrails, power lines, distant birds). Render at the real 16:9 frame size, not square crops, so the horizon and sky share the real proportions.

## Label hygiene (checked in the Roboflow versions)
- Some renders have a drone labelled bird or airplane, and some frames labelled empty hold a few-pixel drone. Export the class and a visibility flag from the simulator instead of labelling by hand. Where a drone is below, say, 4 px or occluded, leave it unlabelled.
- Keep the sequence order so train/test can be split by time. The 0–499 and 1–1000 sequences are the only way to split without leakage.
