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
import torchaudio
from torch import nn


class MelLoss(nn.Module):
    def __init__(self, sample_rate: int = 44100, n_mels: int = 80, loss_weight: float = 4.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.loss_weight = loss_weight
        self.n_fft = 2048
        self.hop_length = 512
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels=n_mels, sample_rate=sample_rate, n_stft=self.n_fft // 2 + 1
        )

    def get_mel_spectrogram(self, x: torch.Tensor) -> torch.Tensor:
        window = torch.hann_window(self.n_fft, device=x.device, dtype=x.dtype)
        spec = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                         win_length=self.n_fft, window=window, return_complex=True)
        power_spec = spec.abs().pow(2)
        mel_spec = self.mel_scale(power_spec)
        return mel_spec

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        if y_pred.dim() == 3:
            b, c, t = y_pred.shape
            y_pred = y_pred.reshape(b * c, t)
            y_true = y_true.reshape(b * c, t)
        min_len = min(y_pred.shape[-1], y_true.shape[-1])
        mel_pred = torch.log(self.get_mel_spectrogram(y_pred[..., :min_len]) + 1e-7)
        mel_true = torch.log(self.get_mel_spectrogram(y_true[..., :min_len]) + 1e-7)
        return self.loss_weight * nn.functional.l1_loss(mel_pred, mel_true)
