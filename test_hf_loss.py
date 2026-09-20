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

"""Quick smoke test for HighFrequencyLoss."""
import torch
import sys
sys.path.insert(0, ".")

from losses.spectral import HighFrequencyLoss

def test_high_frequency_loss():
    sr = 96000
    duration = 1.0
    n_samples = int(sr * duration)
    t = torch.linspace(0, duration, n_samples)

    # "Clean" signal: 1 kHz sine — no energy above 15 kHz
    clean = torch.sin(2 * torch.pi * 1000 * t).unsqueeze(0).unsqueeze(0)  # (1, 1, N)

    # "Noisy" signal: same sine + white noise (will have flat HF energy)
    noise = 0.3 * torch.randn_like(clean)
    noisy = (clean + noise).requires_grad_(True)

    hf_loss = HighFrequencyLoss(n_fft=2048, sample_rate=sr, cutoff_freq=15000, loss_weight=5.0)

    loss_val = hf_loss(noisy, clean)
    assert loss_val.item() > 0.0, f"Loss should be > 0, got {loss_val.item()}"

    # Backward must work
    loss_val.backward()
    assert noisy.grad is not None, "Gradient must not be None"
    assert noisy.grad.abs().sum() > 0, "Gradient must be non-zero"

    # Two identical signals should give ~0 loss
    loss_zero = hf_loss(clean.clone(), clean.clone())
    assert loss_zero.item() < 0.01, f"Identical signals loss should be ~0, got {loss_zero.item()}"

    print(f"PASS: HF loss (noisy vs clean) = {loss_val.item():.4f}")
    print(f"PASS: HF loss (clean vs clean) = {loss_zero.item():.6f}")
    print(f"PASS: Gradient shape = {noisy.grad.shape}, sum = {noisy.grad.abs().sum().item():.6f}")

if __name__ == "__main__":
    test_high_frequency_loss()
