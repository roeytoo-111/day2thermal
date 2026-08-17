# Source streams — correlated day RGB + LWIR recording ("arsuf")

These two MPEG-TS files are the *supervision* behind everything under
`data/aligned/` and the raw input to `day2thermal.extract_frames`. This
document records what they are, byte-exactly, so an out-of-band copy can be
verified and the extraction re-run. `manifest.json` next to this file holds
the same facts machine-readably.

| | `arsuf_day.ts` | `arsuf_thermal.ts` |
|---|---|---|
| role | day RGB — network input **A** | LWIR thermal — network target **B** |
| bytes | 2 262 554 620 (2.26 GB) | 544 279 740 (544 MB) |
| SHA-256 | `a377e0a8…d4b9` | `880471d9…c614` |
| container | MPEG-TS, 1 program: H.264 video + KLV data | same |
| video | H.264 Main, yuv420p, **3840 × 2160**, 30 fps nominal | H.264 Main L3.0, yuv420p, **640 × 512**, 25 fps |
| frames (measured) | 21 348, effective 30.0012 fps, PTS spacing 18.8–41.5 ms (jitter, no drops) | 17 813, exactly 25.000 fps, PTS spacing 40.000 ms |
| PTS span | 3600.9676 → 4312.5050 s (711.54 s) | 3600.0000 → 4312.4800 s (712.48 s) |
| bit rate | 25.4 Mbit/s | 6.1 Mbit/s |
| KLV | MISB ST 0601, 21 347 packets @ 30 Hz | MISB ST 0601, 17 812 packets @ 25 Hz |
| sensor position (ST 0601 tags 13/14/15) | 32.1974 N, 34.8148 E, 53.5 → 54.7 m | same, alt starts at unpopulated 0.0 m |

Full SHA-256:

```
a377e0a892e5579ca062eb1d93ffec475fbe3bf2d294126c4bdae722c71ad4b9  arsuf_day.ts
880471d93ed09f3fe85e38a63099d7b999ab0c0d9e86b695b887bde496c7c614  arsuf_thermal.ts
```

## Things the pipeline needs to know about these files

* **Thermal is 8-bit AGC video, not radiometric.** The H.264 luma plane *is*
  the thermal image; there is no 16-bit counterpart. Train with the default
  `--thermal-mode rel8`; `abs16` / calibrated °C is not available from this
  recording. This is why `data/aligned/manifest.csv` reports `uint8` and
  0/255 saturation.
* **Frame rates differ (30 vs 25 fps).** `extract_frames` handles this by
  resampling both to `--fps` on a common time axis with a ±20 ms tolerance
  (half the thermal period). The recorded 4 897 raw pairs over 711.5 s imply
  `--fps ≈ 7` after tolerance/static drops.
* **`extract_frames` assumes constant fps (`t = idx / fps`).** Verified OK
  here: thermal PTS spacing is exactly 40 ms; day PTS jitters ±5 ms around
  33.3 ms but has no gaps, so index-time drifts < 70 ms end-to-end.
* **The KLV telemetry gives the inter-stream offset directly.** Both files
  carry MISB ST 0601 Precision Time Stamps (tag 2) on one shared encoder
  clock (their series end within 10 ms of the same value). Thermal frame 0 is
  stamped 582.362509 s, day frame 0 is 583.255841 s, so on the index-time
  axis that `extract_frames` uses:

  ```
  --offset-ms  ≈  +893   (thermal minus RGB; +897 at end of recording, 4 ms drift)
  ```

  i.e. the thermal recording started ~0.9 s before the day recording and RGB
  frame 0 pairs with thermal frame ~22. **This replaces the "not recorded"
  offset in README → Reproducing → Stage 1, but it is derived from telemetry
  and has not been validated by the hot-object procedure. Do that check
  before trusting it to the frame.**

## Regenerating these numbers

```bash
sha256sum arsuf_day.ts arsuf_thermal.ts
ffprobe -v error -show_streams -show_format arsuf_day.ts
# frame count / PTS statistics (video packets only, no decode)
ffprobe -v error -select_streams v:0 -show_entries packet=pts_time -of csv=p=0 arsuf_day.ts | grep -v '^$'
# KLV: demux the data stream, parse SMPTE 336M / MISB 0601 local sets, read tag 2 (8-byte µs timestamp)
ffmpeg -v error -i arsuf_day.ts -map 0:d:0 -c copy -f data - | xxd | head
```

## Where the bytes live

**Not in git** — by decision (2026-08-17), consistent with the rest of this
repository: `.gitignore` blocks `*.ts`, and 2.8 GB of H.264 cannot be pushed
to the GitHub remote in any case (100 MiB object limit; the day file also
exceeds the 2 GB Git-LFS / Release-asset ceilings without splitting). This
manifest is the tracked record; the bytes are kept out-of-band (at the time
of writing: `/home/me/shared/rec/` on the recording workstation). Verify any
copy against the SHA-256 above before extracting frames.
