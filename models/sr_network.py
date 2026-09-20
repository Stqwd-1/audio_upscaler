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

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    """GroupNorm that adapts groups count to channel count."""
    return nn.GroupNorm(min(groups, channels), channels)


class FiLMConditioning(nn.Module):
    """Feature-wise Linear Modulation.

    Applies affine transformation (scale, shift) to features based on
    a conditioning vector. Used to condition the model on metadata like
    original sample rate.
    """
    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x: (B, C, T), cond: (B, cond_dim)"""
        params = self.proj(cond)  # (B, C*2)
        gamma, beta = params.chunk(2, dim=-1)  # each (B, C)
        gamma = gamma.unsqueeze(-1)  # (B, C, 1)
        beta = beta.unsqueeze(-1)   # (B, C, 1)
        return x * (1 + gamma) + beta


class NoiseInjection(nn.Module):
    """Learnable noise injection for high-frequency hallucination."""
    def __init__(self, channels: int):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            noise = torch.randn_like(x)
            return x + self.weight * noise
        return x


class GatedResBlock(nn.Module):
    """WaveNet-style gated residual block with noise injection.

    Architecture: tanh(signal) * sigmoid(gate) — controls information flow.
    Optional FiLM conditioning for sample-rate-aware processing.
    """
    def __init__(self, channels: int, dilation: int = 1, cond_dim: int = 0):
        super().__init__()
        padding = (3 - 1) * dilation // 2

        # Signal path
        self.conv_signal = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        # Gate path
        self.conv_gate = nn.Conv1d(channels, channels, 3, padding=padding, dilation=dilation)
        # Output
        self.conv_out = nn.Conv1d(channels, channels, 3, padding=1)

        self.noise1 = NoiseInjection(channels)
        self.noise2 = NoiseInjection(channels)

        self.norm1 = _gn(channels)
        self.norm2 = _gn(channels)

        # FiLM conditioning
        self.use_film = cond_dim > 0
        if self.use_film:
            self.film1 = FiLMConditioning(cond_dim, channels)
            self.film2 = FiLMConditioning(cond_dim, channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        residual = x

        # Signal + Gate
        s = self.norm1(self.conv_signal(x))
        if self.use_film and cond is not None:
            s = self.film1(s, cond)
        s = self.noise1(s)
        g = self.conv_gate(x)

        # Gated activation: tanh * sigmoid
        h = torch.tanh(s) * torch.sigmoid(g)

        # Output
        out = self.norm2(self.conv_out(h))
        if self.use_film and cond is not None:
            out = self.film2(out, cond)
        out = self.noise2(out)

        return out + residual


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.norm = _gn(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, 4, stride=2, padding=1)
        self.conv = nn.Conv1d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.norm = _gn(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        h = self.up(x)
        min_len = min(h.shape[-1], skip.shape[-1])
        h = h[..., :min_len]
        skip = skip[..., :min_len]
        h = torch.cat([h, skip], dim=1)
        return self.act(self.norm(self.conv(h)))


class SRNetwork(nn.Module):
    """Stereo/multi-channel audio super-resolution U-Net.

    Features:
    - Gated Residual Blocks (WaveNet-style) for better high-freq generation
    - Noise Injection for stochastic high-frequency hallucination
    - Global Residual Learning (learns difference from input)
    - FiLM Conditioning for sample-rate-aware processing
    - in_channels / out_channels = 2 for stereo, 1 for mono
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: list[int] = None,
        num_res_blocks: int = 8,
        cond_dim: int = 0,
    ):
        super().__init__()
        if channel_multipliers is None:
            channel_multipliers = [1, 2, 4, 8]

        channels = [base_channels * m for m in channel_multipliers]
        self.cond_dim = cond_dim

        # Conditioning encoder (optional)
        if cond_dim > 0:
            self.cond_encoder = nn.Sequential(
                nn.Linear(1, 32),
                nn.SiLU(),
                nn.Linear(32, cond_dim),
            )
        else:
            self.cond_encoder = None

        self.input_conv = nn.Conv1d(in_channels, base_channels, 7, padding=3)

        self.down_blocks = nn.ModuleList()
        in_ch = base_channels
        for out_ch in channels:
            self.down_blocks.append(DownBlock(in_ch, out_ch))
            in_ch = out_ch

        self.bottleneck = nn.Sequential(
            *[GatedResBlock(channels[-1], dilation=2 ** (i % 4), cond_dim=cond_dim)
              for i in range(num_res_blocks)]
        )

        self.up_blocks = nn.ModuleList()
        reversed_channels = list(reversed(channels))
        for i in range(len(reversed_channels)):
            in_ch = reversed_channels[i]
            if i < len(reversed_channels) - 1:
                skip_ch = reversed_channels[i + 1]
                out_ch = reversed_channels[i + 1] if i < len(reversed_channels) - 2 else base_channels
            else:
                skip_ch = base_channels
                out_ch = base_channels
            self.up_blocks.append(UpBlock(in_ch, skip_ch, out_ch))

        self.output_conv = nn.Conv1d(base_channels, out_channels, 7, padding=3)

    def forward(self, x: torch.Tensor, cond: torch.Tensor = None) -> torch.Tensor:
        """x: (B, C, T), cond: (B, 1) optional conditioning (normalized sample rate)"""
        skip_connections = []

        h = self.input_conv(x)
        skip_connections.append(h)

        for down in self.down_blocks:
            h = down(h)
            skip_connections.append(h)

        # Encode conditioning vector
        cond_vec = None
        if self.cond_encoder is not None:
            if cond is None:
                # Default: no conditioning (zeros)
                cond = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            cond_vec = self.cond_encoder(cond)  # (B, cond_dim)

        h = self.bottleneck(h) if cond_vec is None else self.bottleneck[0](h, cond_vec)
        for block in self.bottleneck[1:]:
            h = block(h, cond_vec) if cond_vec is not None else block(h)

        for i, up in enumerate(self.up_blocks):
            skip = skip_connections[-(i + 2)]
            h = up(h, skip)

        out = self.output_conv(h)
        min_len = min(out.shape[-1], x.shape[-1])
        return out[..., :min_len] + x[..., :min_len]
