#!/usr/bin/env python3
import os
import subprocess
from pathlib import Path

# Mapping prefix -> (start_time, end_time)
# Time format: HH:MM:SS or MM:SS
CLIPPING_CONFIG = {
    "2026-06-23": ("00:05:00", "00:13:00"),
    "2026-07-15": ("00:02:00", "00:10:00"),
    "2026-07-30": ("00:06:00", "00:17:00"),
}


def clip_video(input_path: Path, output_path: Path, start: str, end: str):
    """Clips video using ffmpeg stream copy (fast, no re-encoding)."""
    cmd = [
        "ffmpeg",
        "-y",                  # Overwrite output without asking
        "-ss", start,          # Fast seek to start time
        "-to", end,            # Cut until end time
        "-i", str(input_path), # Input file
        "-c", "copy",          # Stream copy (no re-encoding, extremely fast)
        str(output_path),
    ]

    print(f"🎬 Clipping {input_path.name} [{start} -> {end}]...")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode == 0:
        print(f"✅ Saved to {output_path.name}")
    else:
        print(f"❌ Error processing {input_path.name}:\n{result.stderr.decode()}")


def main():
    target_dir = Path(".")
    output_dir = target_dir / "clipped"
    output_dir.mkdir(exist_ok=True)

    # Supported video extensions
    video_extensions = {".mp4", ".ts", ".mkv", ".avi", ".mov"}

    for file_path in target_dir.iterdir():
        if file_path.suffix.lower() not in video_extensions:
            continue

        filename = file_path.name
        for date_prefix, (start_time, end_time) in CLIPPING_CONFIG.items():
            if filename.startswith(date_prefix):
                out_name = f"{file_path.stem}_clipped{file_path.suffix}"
                out_path = output_dir / out_name
                clip_video(file_path, out_path, start_time, end_time)
                break


if __name__ == "__main__":
    main()