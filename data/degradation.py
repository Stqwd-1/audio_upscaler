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

"""Advanced GPU-based audio degradation pipeline.

Simulates real-world degradation:
- Reverb (convolution with random IR)
- EQ / Coloration (random biquad filters)
- Clipping (soft/hard)
- Bandwidth Limiting (resampling-based low-pass)
- Additive Noise (Gaussian with SNR control)
"""
import random

import torch
import torch.nn.functional as F
from torch import nn


class AdvancedDegradation(nn.Module):
    def __init__(self, sample_rate=44100, max_cutoff=16000):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_cutoff = max_cutoff

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (Channels, Time) or (Batch, Channels, Time)"""
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        if random.random() < 0.3:
            waveform = self.apply_reverb(waveform)
        if random.random() < 0.3:
            waveform = self.apply_random_eq(waveform)
        if random.random() < 0.3:
            waveform = self.apply_noise(waveform)
        if random.random() < 0.2:
            waveform = self.apply_clipping(waveform)
        waveform = self.apply_bandwidth_limit(waveform)

        return waveform.squeeze(0) if squeeze else waveform

    def apply_reverb(self, waveform):
        device = waveform.device
        reverb_len = int(random.uniform(0.1, 0.5) * self.sample_rate)
        if reverb_len == 0:
            return waveform

        t = torch.linspace(0, 1, reverb_len, device=device)
        decay = torch.exp(-t * random.uniform(5, 15))
        ir = torch.randn(1, 1, reverb_len, device=device) * decay.view(1, 1, -1)
        ir = ir / torch.norm(ir) * 0.5

        n_ch = waveform.shape[1]
        padded = F.pad(waveform, (reverb_len - 1, 0))
        # Convolve each channel separately
        wet = torch.cat([F.conv1d(padded[:, ch:ch+1, :], ir) for ch in range(n_ch)], dim=1)

        mix = random.uniform(0.1, 0.4)
        return (1 - mix) * waveform + mix * wet[..., :waveform.shape[-1]]

    def apply_random_eq(self, waveform):
        import torchaudio
        center_freq = random.uniform(200, 8000)
        gain = random.uniform(-10, 10)
        q = random.uniform(0.5, 2.0)
        return torchaudio.functional.equalizer_biquad(waveform, self.sample_rate, center_freq, gain, q)

    def apply_clipping(self, waveform):
        threshold = random.uniform(0.5, 0.9)
        if random.random() < 0.5:
            return torch.clamp(waveform, -threshold, threshold)
        return torch.tanh(waveform / threshold) * threshold

    def apply_noise(self, waveform):
        snr_db = random.uniform(20, 50)
        noise = torch.randn_like(waveform)
        sig_power = waveform.pow(2).mean()
        noise_power = noise.pow(2).mean()
        scale = torch.sqrt(sig_power / (noise_power * (10 ** (snr_db / 10)) + 1e-10))
        return waveform + noise * scale

    def apply_bandwidth_limit(self, waveform):
        cutoff = random.randint(4000, self.max_cutoff)
        if cutoff >= self.sample_rate // 2 * 0.95:
            return waveform

        orig_len = waveform.shape[-1]
        down = F.interpolate(waveform, scale_factor=cutoff * 2 / self.sample_rate, mode='linear', align_corners=False)
        up = F.interpolate(down, size=orig_len, mode='linear', align_corners=False)
        return up
