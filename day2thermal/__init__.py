"""day2thermal: RGB (day) -> LWIR thermal image translation pipeline for UAV data.

Implements a supervised paired-translation pipeline inspired by
ThermalGAN (Kniaz et al., ECCV Workshops 2018) with practical adaptations
for synchronized day/thermal video recordings:

* frame extraction + temporal synchronization from paired videos
* cross-modality spatial registration (manual homography / ECC)
* single-stage pix2pix baseline (RGB -> thermal)
* two-stage ThermalGAN-style model:
    G1: RGB + temperature plane -> low-frequency thermal base ("thermal
        segmentation" surrogate, annotation-free)
    G2: RGB + base -> relative thermal contrasts
    thermal = base + contrasts
* batch inference to translate an RGB-only dataset into thermal, optionally
  sweeping ambient temperatures to produce multimodal outputs
* evaluation (L1 / RMSE / PSNR / SSIM, in degrees C for radiometric data)
"""

__version__ = "0.1.0"
