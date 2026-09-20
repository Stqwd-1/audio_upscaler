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


class SpectralLoss(nn.Module):
    """Multi-resolution STFT loss with Spectral Convergence.

    Components:
    - L1 magnitude loss
    - Spectral Convergence (normalized Frobenius norm)
    """
    def __init__(
        self,
        fft_sizes: list[int] = None,
        hop_sizes: list[int] = None,
        win_sizes: list[int] = None,
        loss_weight: float = 4.5,
    ):
        super().__init__()
        if fft_sizes is None:
            fft_sizes = [1024, 2048, 4096]
        if hop_sizes is None:
            hop_sizes = [120, 240, 480]
        if win_sizes is None:
            win_sizes = [600, 1200, 2400]

        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes
        self.loss_weight = loss_weight

    def stft(self, x: torch.Tensor, n_fft: int, hop_length: int, win_length: int) -> torch.Tensor:
        if x.dim() == 3:
            b, c, t = x.shape
            x = x.reshape(b * c, t)
        window = torch.hann_window(win_length, device=x.device, dtype=x.dtype)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window, return_complex=True)
        return torch.abs(spec)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        l1_loss = 0.0
        sc_loss = 0.0

        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            min_len = min(y_pred.shape[-1], y_true.shape[-1])
            pred_spec = self.stft(y_pred[..., :min_len], n_fft, hop, win)
            true_spec = self.stft(y_true[..., :min_len], n_fft, hop, win)

            # L1 magnitude loss
            l1_loss += F.l1_loss(pred_spec, true_spec)

            # Spectral Convergence: ||target - pred||_F / ||target||_F
            sc = torch.norm(true_spec - pred_spec, p="fro") / (torch.norm(true_spec, p="fro") + 1e-7)
            sc_loss += sc

        n = len(self.fft_sizes)
        return self.loss_weight * (l1_loss / n + sc_loss / n)


class LSDLoss(nn.Module):
    """Log-Spectral Distance — for validation/metrics.

    LSD = sqrt(mean((log(S_pred) - log(S_true))^2))
    Lower is better. < 2.0 = excellent, > 4.0 = poor.
    """
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Match lengths
        min_len = min(y_pred.shape[-1], y_true.shape[-1])
        y_pred = y_pred[..., :min_len]
        y_true = y_true[..., :min_len]

        if y_pred.dim() == 3:
            b, c, t = y_pred.shape
            y_pred = y_pred.reshape(b * c, t)
            y_true = y_true.reshape(b * c, t)

        window = torch.hann_window(self.n_fft, device=y_pred.device, dtype=y_pred.dtype)

        pred_stft = torch.stft(y_pred, self.n_fft, self.hop_length, self.n_fft, window=window, return_complex=True)
        true_stft = torch.stft(y_true, self.n_fft, self.hop_length, self.n_fft, window=window, return_complex=True)

        pred_mag = torch.abs(pred_stft) + 1e-7
        true_mag = torch.abs(true_stft) + 1e-7

        lsd = torch.sqrt(torch.mean((torch.log(pred_mag) - torch.log(true_mag)) ** 2, dim=-1))
        return lsd.mean()


class HFExciterLoss(nn.Module):
    """High-Frequency Harmonic Exciter Loss.

    Targets 16kHz-48kHz range to encourage the generator
    to produce realistic high-frequency harmonics for studio-quality restoration.

    Components:
    1. HF Energy Loss: penalizes when pred has less HF energy than target
    2. HF Phase Coherence: penalizes phase inconsistency in HF range
    3. HF Spectral Convergence: normalized Frobenius norm on HF bins
    """
    def __init__(
        self,
        sample_rate: int = 44100,
        hf_start_freq: int = 16000,
        n_fft: int = 4096,
        loss_weight: float = 2.0,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.hf_start_freq = hf_start_freq
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.loss_weight = loss_weight
        self.hf_bin_start = int(hf_start_freq * n_fft / sample_rate)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        min_len = min(y_pred.shape[-1], y_true.shape[-1])
        pred = y_pred[..., :min_len]
        true = y_true[..., :min_len]
        if pred.dim() == 3:
            b, c, t = pred.shape
            pred = pred.reshape(b * c, t)
            true = true.reshape(b * c, t)

        window = torch.hann_window(self.n_fft, device=pred.device, dtype=pred.dtype)
        pred_stft = torch.stft(pred, self.n_fft, self.hop_length, self.n_fft,
                               window=window, return_complex=True)
        true_stft = torch.stft(true, self.n_fft, self.hop_length, self.n_fft,
                               window=window, return_complex=True)

        # Extract HF bins (above hf_start_freq)
        pred_hf = pred_stft[..., self.hf_bin_start:, :]
        true_hf = true_stft[..., self.hf_bin_start:, :]

        pred_mag = torch.abs(pred_hf)
        true_mag = torch.abs(true_hf)

        # 1. HF Energy Loss: penalize under-energetic HF
        energy_diff = F.relu(true_mag.mean() - pred_mag.mean())
        energy_loss = energy_diff ** 2

        # 2. HF Phase Coherence
        pred_phase = torch.angle(pred_hf)
        true_phase = torch.angle(true_hf)
        phase_loss = F.l1_loss(pred_phase, true_phase)

        # 3. HF Spectral Convergence
        sc = torch.norm(true_mag - pred_mag, p="fro") / (torch.norm(true_mag, p="fro") + 1e-7)

        return self.loss_weight * (energy_loss + phase_loss + sc)


class HighFrequencyLoss(nn.Module):
    """High-Frequency Suppression Loss — penalizes white noise in the upper spectrum.

    Unlike HFExciterLoss (which encourages HF energy), this loss punishes the
    generator for placing energy in frequency bands where the target is quiet.
    The discriminator cheating manifests as flat, high-energy white noise across
    15–48 kHz; this loss forces the generator to produce smooth, natural spectral
    roll-off instead.

    Loss = L1( log|STFT_pred|_HF , log|STFT_true|_HF )
    """
    def __init__(
        self,
        n_fft: int = 2048,
        sample_rate: int = 96000,
        cutoff_freq: float = 15000.0,
        loss_weight: float = 5.0,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = n_fft // 4
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.loss_weight = loss_weight
        self.hf_bin_start = int(cutoff_freq * n_fft / sample_rate)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        min_len = min(y_pred.shape[-1], y_true.shape[-1])
        pred = y_pred[..., :min_len]
        true = y_true[..., :min_len]

        if pred.dim() == 3:
            b, c, t = pred.shape
            pred = pred.reshape(b * c, t)
            true = true.reshape(b * c, t)

        window = torch.hann_window(self.n_fft, device=pred.device, dtype=pred.dtype)
        pred_stft = torch.stft(pred, self.n_fft, self.hop_length, self.n_fft,
                               window=window, return_complex=True)
        true_stft = torch.stft(true, self.n_fft, self.hop_length, self.n_fft,
                               window=window, return_complex=True)

        # HF bins: from cutoff_freq to Nyquist
        pred_hf = pred_stft[..., self.hf_bin_start:, :]
        true_hf = true_stft[..., self.hf_bin_start:, :]

        pred_log_mag = torch.log(torch.abs(pred_hf) + 1e-7)
        true_log_mag = torch.log(torch.abs(true_hf) + 1e-7)

        return self.loss_weight * F.l1_loss(pred_log_mag, true_log_mag)
