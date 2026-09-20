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
    return nn.GroupNorm(min(groups, channels), channels)


class SpectralConvBlock(nn.Module):
    """Conv block for 2D spectrogram processing."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = _gn(out_ch)
        self.norm2 = _gn(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.act(self.norm2(self.conv2(h)))
        return h


class SpectralUNet(nn.Module):
    """U-Net operating on magnitude spectrograms.

    Input: (batch, 1, freq_bins, time_frames) — log magnitude spectrogram
    Output: (batch, 1, freq_bins, time_frames) — enhanced log magnitude

    Used as second stream in hybrid architecture:
    - Waveform model generates audio
    - Spectral model refines the spectrogram
    - Phase from waveform model is preserved
    """
    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        ch = base_channels

        # Encoder
        self.enc1 = SpectralConvBlock(in_channels, ch)
        self.enc2 = SpectralConvBlock(ch, ch * 2)
        self.enc3 = SpectralConvBlock(ch * 2, ch * 4)
        self.enc4 = SpectralConvBlock(ch * 4, ch * 8)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = SpectralConvBlock(ch * 8, ch * 8)

        # Decoder
        self.up4 = nn.ConvTranspose2d(ch * 8, ch * 8, 2, stride=2)
        self.dec4 = SpectralConvBlock(ch * 16, ch * 4)

        self.up3 = nn.ConvTranspose2d(ch * 4, ch * 4, 2, stride=2)
        self.dec3 = SpectralConvBlock(ch * 8, ch * 2)

        self.up2 = nn.ConvTranspose2d(ch * 2, ch * 2, 2, stride=2)
        self.dec2 = SpectralConvBlock(ch * 4, ch)

        self.up1 = nn.ConvTranspose2d(ch, ch, 2, stride=2)
        self.dec1 = SpectralConvBlock(ch * 2, ch)

        self.output = nn.Conv2d(ch, in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = self._match_size(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.output(d1)

    def _match_size(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Pad/crop x to match target spatial dimensions."""
        if x.shape[2:] != target.shape[2:]:
            x = F.interpolate(x, size=target.shape[2:], mode='bilinear', align_corners=False)
        return x
