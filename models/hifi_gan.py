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

# The generator and ResBlock below follow HiFi-GAN (Kong et al., 2020,
# "HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity
# Speech Synthesis"): https://github.com/jik876/hifi-gan (MIT License) — the
# original code was released under MIT; this module re-implements it.

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class ResBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: list[int]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilation:
            self.convs1.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=d, padding=(kernel_size - 1) * d // 2))
            )
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, dilation=1, padding=(kernel_size - 1) // 2)))
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class Generator(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        upsample_rates: list[int] | None = None,
        upsample_kernel_sizes: list[int] | None = None,
        upsample_initial_channel: int = 512,
        resblock_kernel_sizes: list[int] | None = None,
        resblock_dilation_sizes: list[list[int]] | None = None,
    ):
        super().__init__()
        if upsample_rates is None:
            upsample_rates = [8, 5, 3]
        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = [16, 11, 7]
        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = [3, 7, 11]
        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(nn.Conv1d(in_channels, upsample_initial_channel, 7, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k, u, padding=(k - u) // 2
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        ch = upsample_initial_channel // (2 ** len(upsample_rates))
        for i in range(len(upsample_rates)):
            ch_new = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(ResBlock(ch_new, k, d))

        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, padding=3, bias=False))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.leaky_relu(self.conv_pre(x), LRELU_SLOPE)
        for i in range(self.num_upsamples):
            h = F.leaky_relu(self.ups[i](h), LRELU_SLOPE)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](h)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](h)
            h = xs / self.num_kernels
        h = F.leaky_relu(h)
        h = torch.tanh(self.conv_post(h))
        return h

    def remove_weight_norm(self):
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)
