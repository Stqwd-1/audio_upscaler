# Copyright 2026 Stanislav Suharkov
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert a FLAC library to MP3 (for building training/degradation datasets).

Usage:
    python convert_library.py [input_dir] [output_dir]
    python convert_library.py [library_dir] info
"""
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import soundfile as sf


def convert_flac_to_mp3(input_dir, output_dir, bitrate="320k"):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    flac_files = list(input_path.glob("*.flac"))
    if not flac_files:
        print(f"No FLAC files found in {input_dir}")
        return

    print(f"Found {len(flac_files)} FLAC files")
    converted = 0
    errors = 0

    for flac_file in flac_files:
        mp3_file = output_path / f"{flac_file.stem}.mp3"
        if mp3_file.exists():
            continue

        try:
            result = subprocess.run([
                "ffmpeg", "-y", "-i", str(flac_file),
                "-codec:a", "libmp3lame", "-b:a", bitrate,
                str(mp3_file)
            ], capture_output=True, text=True, timeout=30)

            if result.returncode == 0:
                converted += 1
                if converted % 10 == 0:
                    print(f"Converted {converted}/{len(flac_files)}")
            else:
                errors += 1
                print(f"Error: {flac_file.name}: {result.stderr[:100]}")
        except subprocess.TimeoutExpired:
            errors += 1
            print(f"Timeout: {flac_file.name}")
        except FileNotFoundError:
            print("ERROR: ffmpeg not found!")
            print("Install ffmpeg: https://ffmpeg.org/download.html")
            return

    print(f"\nDone! Converted: {converted}, Errors: {errors}")


def show_library_info(library_dir):
    library_path = Path(library_dir)
    flac_dir = library_path / "flac"
    mp3_dir = library_path / "mp3"

    flac_files = list(flac_dir.glob("*.flac")) if flac_dir.exists() else []
    mp3_files = list(mp3_dir.glob("*.mp3")) if mp3_dir.exists() else []

    print(f"\n=== Library Info ===")
    print(f"FLAC files: {len(flac_files)}")
    print(f"MP3 files: {len(mp3_files)}")

    if flac_files:
        total_size = sum(f.stat().st_size for f in flac_files)
        print(f"Total FLAC size: {total_size / (1024 * 1024 * 1024):.2f} GB")

        print(f"\nSample files:")
        for f in flac_files[:5]:
            try:
                info = sf.info(str(f))
                duration = info.duration
                print(f"  {f.name}: {duration:.1f}s, {info.samplerate}Hz, {info.channels}ch")
            except Exception:
                print(f"  {f.name}: (info unavailable)")


if __name__ == "__main__":
    library_dir = sys.argv[1] if len(sys.argv) > 1 else "library"

    if len(sys.argv) > 2 and sys.argv[2] == "info":
        show_library_info(library_dir)
    else:
        convert_flac_to_mp3(f"{library_dir}/flac", f"{library_dir}/mp3")