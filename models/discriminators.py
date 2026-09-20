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

# Multi-Period and Multi-Scale discriminators follow HiFi-GAN (Kong et al.,
# 2020): https://github.com/jik876/hifi-gan (MIT License).

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn.utils.parametrizations import weight_norm


class DiscriminatorP(nn.Module):
    def __init__(self, period: int, kernel_size: int = 5, stride: int = 3):
        super().__init__()
        self.period = period
        self.convs = nn.ModuleList()
        channels = [1, 32, 128, 512, 1024, 1024]
        for i in range(len(channels) - 1):
            self.convs.append(
                weight_norm(
                    nn.Conv2d(channels[i], channels[i + 1], (kernel_size, 1), (stride, 1), padding=(2, 0))
                )
            )
        self.conv_post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorS(nn.Module):
    def __init__(self, resolution: int):
        super().__init__()
        self.convs = nn.ModuleList()
        channels = [1, 128, 128, 256, 256, 512, 512, 1024, 1024]
        for i in range(len(channels) - 1):
            stride = 1 if i == len(channels) - 2 else 2
            self.convs.append(spectral_norm(nn.Conv1d(channels[i], channels[i + 1], 15, stride, 7)))
        self.conv_post = spectral_norm(nn.Conv1d(1024, 1, 3, 1, 1, bias=False))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorSpectral(nn.Module):
    """Operates on magnitude spectrograms (2D)."""
    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleList([
            spectral_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            spectral_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            spectral_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            spectral_norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            spectral_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = spectral_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        fmap = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, periods: list[int] | None = None):
        super().__init__()
        if periods is None:
            periods = [2, 3, 5, 7, 11]
        self.discriminators = nn.ModuleList([DiscriminatorP(p) for p in periods])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    @torch.no_grad()
    def score_candidate(self, y_hat: torch.Tensor, **kwargs) -> float:
        """Score a candidate using all sub-discriminators. Returns mean sigmoid score."""
        mono = y_hat.mean(dim=1, keepdim=True) if y_hat.shape[1] > 1 else y_hat
        scores = []
        for d in self.discriminators:
            score, _ = d(mono)
            scores.append(torch.sigmoid(score).mean().item())
        return sum(scores) / max(len(scores), 1)


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, resolutions: list[int] | None = None):
        super().__init__()
        if resolutions is None:
            resolutions = [1024, 2048, 4096]
        self.discriminators = nn.ModuleList([DiscriminatorS(r) for r in resolutions])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

    @torch.no_grad()
    def score_candidate(self, y_hat: torch.Tensor, **kwargs) -> float:
        """Score a candidate using all sub-discriminators. Returns mean sigmoid score."""
        mono = y_hat.mean(dim=1, keepdim=True) if y_hat.shape[1] > 1 else y_hat
        scores = []
        for d in self.discriminators:
            score, _ = d(mono)
            scores.append(torch.sigmoid(score).mean().item())
        return sum(scores) / max(len(scores), 1)


class MultiResolutionDiscriminator(nn.Module):
    """Multi-Resolution Discriminator — operates on spectrograms at different STFT resolutions.

    - Computes magnitude STFT at 3 resolutions
    - Each goes through a spectral discriminator
    - Returns sigmoid scores (0-1 realism probability)
    """
    def __init__(self, resolutions=None):
        super().__init__()
        if resolutions is None:
            resolutions = [[1024, 120, 600], [2048, 240, 1200], [512, 50, 240]]
        self.resolutions = resolutions
        self.discriminators = nn.ModuleList([DiscriminatorSpectral() for _ in resolutions])

    def forward(self, y: torch.Tensor, y_hat: torch.Tensor):
        y_disc_r, y_disc_g = [], []
        fmap_r, fmap_g = [], []

        # Mono for STFT
        y_mono = y.mean(dim=1) if y.dim() == 3 and y.shape[1] > 1 else y.squeeze(1)
        y_hat_mono = y_hat.mean(dim=1) if y_hat.dim() == 3 and y_hat.shape[1] > 1 else y_hat.squeeze(1)

        for i, (n_fft, hop, win) in enumerate(self.resolutions):
            window = torch.hann_window(win, device=y.device, dtype=y.dtype)

            y_spec = torch.abs(torch.stft(y_mono, n_fft, hop, win, window=window, return_complex=True))
            y_spec = y_spec.unsqueeze(1)  # (B, 1, Freq, Time)

            y_hat_spec = torch.abs(torch.stft(y_hat_mono, n_fft, hop, win, window=window, return_complex=True))
            y_hat_spec = y_hat_spec.unsqueeze(1)

            score_r, f_r = self.discriminators[i](y_spec)
            score_g, f_g = self.discriminators[i](y_hat_spec)

            y_disc_r.append(score_r)
            y_disc_g.append(score_g)
            fmap_r.append(f_r)
            fmap_g.append(f_g)

        return y_disc_r, y_disc_g, fmap_r, fmap_g

    @torch.no_grad()
    def score_candidate(self, y_hat: torch.Tensor, chunk_seconds: float = 5.0, sr: int = 44100) -> float:
        """Score a single candidate waveform.

        Returns average realism score (0-1) across all resolutions.
        Processes in chunks for long audio.
        """
        chunk_size = int(chunk_seconds * sr)
        total_len = y_hat.shape[-1]

        if total_len <= chunk_size:
            return self._score_chunk(y_hat, sr)

        total_score = 0.0
        num_chunks = 0

        for i in range(0, total_len, chunk_size):
            end = min(i + chunk_size, total_len)
            chunk = y_hat[..., i:end]
            if chunk.shape[-1] < 1024:
                continue
            total_score += self._score_chunk(chunk, sr)
            num_chunks += 1

        return total_score / max(1, num_chunks)

    def _score_chunk(self, chunk: torch.Tensor, sr: int = 44100) -> float:
        """Score a single chunk across all resolutions."""
        total_score = 0.0
        count = 0

        # Mono
        if chunk.dim() == 3 and chunk.shape[1] > 1:
            chunk = chunk.mean(dim=1, keepdim=True)

        for i, (n_fft, hop, win) in enumerate(self.resolutions):
            if chunk.shape[-1] < n_fft:
                chunk = F.pad(chunk, (0, n_fft - chunk.shape[-1]))

            window = torch.hann_window(win, device=chunk.device, dtype=chunk.dtype)
            spec = torch.abs(torch.stft(chunk.squeeze(1), n_fft, hop, win, window=window, return_complex=True))
            spec = spec.unsqueeze(1)

            device = next(self.discriminators[i].parameters()).device
            spec = spec.to(device)

            score, _ = self.discriminators[i](spec)
            prob = torch.sigmoid(score).mean()
            total_score += prob.item()
            count += 1

        return total_score / max(1, count)
