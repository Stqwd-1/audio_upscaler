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

"""Compare two audio files and report basic similarity metrics.

The reference file is resampled to the sample rate of the second file
(if they differ) so that the comparison happens in a common domain.

Usage:
    python compare.py reference.wav upscaled.wav
"""
import math
import sys

import numpy as np
import soundfile as sf
from scipy import signal


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return audio.mean(axis=1)


def _resample_to(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    g = math.gcd(src_sr, dst_sr)
    up, down = dst_sr // g, src_sr // g
    return signal.resample_poly(audio, up, down, axis=0)


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python compare.py reference.wav upscaled.wav")
        sys.exit(1)

    ref, sr1 = sf.read(sys.argv[1])
    out, sr2 = sf.read(sys.argv[2])

    print(f"Reference: {len(ref)} samples @ {sr1} Hz, shape={ref.shape}")
    print(f"Upscaled:  {len(out)} samples @ {sr2} Hz, shape={out.shape}")

    if sr1 != sr2:
        ref = _resample_to(ref, sr1, sr2)
        print(f"Reference resampled to {sr2} Hz: {len(ref)} samples")

    ref_mono = _to_mono(ref)
    out_mono = _to_mono(out)

    print(f"RMS reference: {np.sqrt(np.mean(ref_mono ** 2)):.6f}")
    print(f"RMS upscaled:  {np.sqrt(np.mean(out_mono ** 2)):.6f}")

    min_len = min(len(ref_mono), len(out_mono))
    diff = np.abs(ref_mono[:min_len] - out_mono[:min_len])
    mse = np.mean(diff ** 2)

    print(f"Max abs diff:  {np.max(diff):.6f}")
    sig_power = np.mean(ref_mono[:min_len] ** 2)
    if mse > 0 and sig_power > 0:
        print(f"SNR: {10 * np.log10(sig_power / mse):.1f} dB")


if __name__ == "__main__":
    main()