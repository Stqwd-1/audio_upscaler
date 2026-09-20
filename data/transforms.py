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

"""CPU-based audio transforms for robust training.

- MP3Compression: Simulates MP3 coding artifacts
- BandwidthLimiter: Low-pass filter via resampling
- QuantizationNoise: Bit-depth reduction
"""
import io
import logging
import random

import torch
import torchaudio

logger = logging.getLogger(__name__)


class MP3Compression:
    """Apply MP3 compression artifacts."""
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def __call__(self, waveform):
        """waveform: (Channels, Time)"""
        if random.random() < 0.2:
            return waveform

        buffer = io.BytesIO()
        try:
            w_cpu = waveform.cpu()
            encoding_sr = min(self.sample_rate, 48000)
            if encoding_sr < self.sample_rate:
                resampler = torchaudio.transforms.Resample(self.sample_rate, encoding_sr)
                w_cpu = resampler(w_cpu)

            torchaudio.save(buffer, w_cpu, encoding_sr, format="mp3")
            buffer.seek(0)
            loaded_w, loaded_sr = torchaudio.load(buffer, format="mp3")

            if loaded_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(loaded_sr, self.sample_rate)
                loaded_w = resampler(loaded_w)

            min_len = min(loaded_w.shape[1], waveform.shape[1])
            loaded_w = loaded_w[:, :min_len]
            if loaded_w.shape[1] < waveform.shape[1]:
                loaded_w = torch.nn.functional.pad(loaded_w, (0, waveform.shape[1] - loaded_w.shape[1]))

            return loaded_w.to(waveform.device)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"MP3 Compression failed: {e}")
            return waveform


class BandwidthLimiter:
    """Low-pass filter via resampling."""
    def __init__(self, sample_rate, min_cutoff=4000, max_cutoff=16000):
        self.sample_rate = sample_rate
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff

    def __call__(self, waveform):
        if random.random() < 0.2:
            return waveform

        cutoff = random.randint(self.min_cutoff, self.max_cutoff)
        down_sr = cutoff * 2
        if down_sr >= self.sample_rate:
            return waveform

        resampler_down = torchaudio.transforms.Resample(self.sample_rate, down_sr)
        resampler_up = torchaudio.transforms.Resample(down_sr, self.sample_rate)

        down = resampler_down(waveform)
        up = resampler_up(down)

        min_len = min(up.shape[1], waveform.shape[1])
        up = up[:, :min_len]
        if up.shape[1] < waveform.shape[1]:
            up = torch.nn.functional.pad(up, (0, waveform.shape[1] - up.shape[1]))

        return up


class QuantizationNoise:
    """Simulate lower bit depth."""
    def __init__(self, min_bits=8, max_bits=14):
        self.min_bits = min_bits
        self.max_bits = max_bits

    def __call__(self, waveform):
        if random.random() < 0.2:
            return waveform

        bits = random.randint(self.min_bits, self.max_bits)
        q_levels = 2 ** bits
        return torch.round(waveform * (q_levels / 2)) / (q_levels / 2)
