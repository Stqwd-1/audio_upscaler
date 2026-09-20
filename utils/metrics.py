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

"""Audio quality metrics.

- LSD (Log-Spectral Distance): Lower is better (< 2.0 = excellent)
- SSIM (Structural Similarity): Higher is better (max 1.0)
- PSNR (Peak Signal-to-Noise Ratio): Higher is better
- SI-SDR (Scale-Invariant SDR): Higher is better
"""
import torch
import torch.nn.functional as F


def calculate_lsd(ref_spec, deg_spec):
    """Log-Spectral Distance between two spectrograms.

    Args:
        ref_spec: Reference magnitude spectrogram
        deg_spec: Generated magnitude spectrogram
    Returns:
        float: LSD value (lower is better)
    """
    if not isinstance(ref_spec, torch.Tensor):
        ref_spec = torch.tensor(ref_spec)
    if not isinstance(deg_spec, torch.Tensor):
        deg_spec = torch.tensor(deg_spec)

    ref_log = torch.log10(ref_spec + 1e-7)
    deg_log = torch.log10(deg_spec + 1e-7)

    diff = (ref_log - deg_log) ** 2
    lsd = torch.mean(torch.sqrt(torch.mean(diff, dim=-2)))
    return lsd.item()


def calculate_ssim(ref_spec, deg_spec):
    """Structural Similarity Index on spectrograms.

    Args:
        ref_spec: Reference spectrogram
        deg_spec: Generated spectrogram
    Returns:
        float: SSIM value (higher is better, max 1.0)
    """
    min_val = min(ref_spec.min(), deg_spec.min())
    max_val = max(ref_spec.max(), deg_spec.max())

    ref_norm = (ref_spec - min_val) / (max_val - min_val + 1e-7)
    deg_norm = (deg_spec - min_val) / (max_val - min_val + 1e-7)

    if ref_norm.dim() == 2:
        ref_norm = ref_norm.unsqueeze(0).unsqueeze(0)
        deg_norm = deg_norm.unsqueeze(0).unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_x = F.avg_pool2d(ref_norm, 3, 1, 1)
    mu_y = F.avg_pool2d(deg_norm, 3, 1, 1)

    sigma_x = F.avg_pool2d(ref_norm ** 2, 3, 1, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(deg_norm ** 2, 3, 1, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(ref_norm * deg_norm, 3, 1, 1) - mu_x * mu_y

    ssim_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ssim_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim_map = ssim_n / ssim_d
    return torch.mean(ssim_map).item()


def calculate_psnr(pred, target, max_val=1.0):
    """Peak Signal-to-Noise Ratio in dB.

    Args:
        pred: Predicted waveform (B, C, T) or (C, T)
        target: Target waveform (same shape)
        max_val: Maximum signal value (1.0 for float audio)
    Returns:
        float: PSNR in dB (higher is better, >40 = excellent)
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    min_len = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :min_len]
    target = target[..., :min_len]
    mse = F.mse_loss(pred, target)
    return (10 * torch.log10(max_val ** 2 / (mse + 1e-10))).item()


def calculate_si_sdr(pred, target):
    """Scale-Invariant Signal-to-Distortion Ratio in dB.

    SI-SDR = 10 * log10(||s_target||^2 / ||noise||^2)
    where s_target = projection of pred onto target direction.

    Args:
        pred: Predicted waveform (B, C, T) or (C, T)
        target: Target waveform (same shape)
    Returns:
        float: SI-SDR in dB (higher is better, >20 = good)
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    min_len = min(pred.shape[-1], target.shape[-1])
    pred = pred[..., :min_len]
    target = target[..., :min_len]
    if pred.dim() == 3:
        b, c, t = pred.shape
        pred = pred.reshape(b * c, t)
        target = target.reshape(b * c, t)
    scale = (pred * target).sum(dim=-1, keepdim=True) / (target.pow(2).sum(dim=-1, keepdim=True) + 1e-8)
    s_target = scale * target
    noise = pred - s_target
    si_sdr = 10 * torch.log10(
        s_target.pow(2).sum(dim=-1) / (noise.pow(2).sum(dim=-1) + 1e-8) + 1e-8
    )
    return si_sdr.mean().item()
