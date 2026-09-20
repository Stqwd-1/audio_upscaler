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

"""Fast, CPU-only smoke tests.

These only cover the non-GPU parts of the pipeline (metrics, degradation,
dataset bookkeeping) so they run anywhere in a few seconds.
"""
import random

import numpy as np
import pytest
import soundfile as sf
import torch

from data.dataset import AudioDataset
from data.degradation import AdvancedDegradation
from utils.metrics import calculate_lsd, calculate_psnr, calculate_si_sdr


def test_psnr_same_signal_is_large():
    x = torch.randn(2, 16000)
    assert calculate_psnr(x, x) > 90


def test_si_sdr_scale_invariant():
    x = torch.randn(2, 16000)
    assert calculate_si_sdr(x, 2.0 * x) > 80


def test_lsd_identical_spectrograms_is_zero():
    spec = torch.abs(torch.randn(1025, 100))
    assert calculate_lsd(spec, spec) < 1e-6


def test_degradation_is_deterministic_with_fixed_seeds():
    torch.manual_seed(7)
    random.seed(7)
    x = torch.randn(2, 16000)
    deg = AdvancedDegradation(sample_rate=16000)

    torch.manual_seed(7)
    random.seed(7)
    a = deg(x)
    torch.manual_seed(7)
    random.seed(7)
    b = deg(x)

    assert torch.allclose(a, b)


def test_degradation_preserves_shape_and_dtype():
    deg = AdvancedDegradation(sample_rate=44100)
    x = torch.randn(2, 44100)
    out = deg(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype


def _write_tone_wav(path, sr=44100, seconds=1.0):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * 440 * t)
    stereo = np.stack([tone, tone], axis=1).astype("float32")
    sf.write(str(path), stereo, sr)
    return path


def test_dataset_detects_files_lazily(tmp_path):
    wav = _write_tone_wav(tmp_path / "tone.wav")

    ds = AudioDataset(
        str(tmp_path),
        sample_rate=44100,
        segment_length=1.0,
        low_sample_rate=16000,
        augment=False,
    )
    assert len(ds) == 1

    item = ds[0]
    assert item["audio"].shape == (2, 44100)
    assert item["low_audio"].shape == (2, ds.segment_samples)


def test_dataset_rejects_non_audio(tmp_path):
    (tmp_path / "notes.txt").write_text("not audio")
    with pytest.raises(RuntimeError):
        AudioDataset(str(tmp_path), sample_rate=44100, augment=False)


def test_dataset_skips_corrupt_files_lazily(tmp_path):
    good = _write_tone_wav(tmp_path / "ok.wav")
    (tmp_path / "broken.wav").write_bytes(b"RIFF\x00\x00\x00\x00 not real wav")

    ds = AudioDataset(str(tmp_path), sample_rate=44100, augment=False)
    assert len(ds) == 1
    assert good.name in str(ds.files[0])

def test_train_shapes_batch2():
    import torch
    b, c, t = 2, 2, 44096
    # Simulate output after ISTFT: (B, samples)
    reconstructed = torch.randn(b, t)
    # What train.py does
    reconstructed = reconstructed.view(b, -1, reconstructed.shape[-1])
    if reconstructed.shape[1] == 1 and c > 1:
        reconstructed = reconstructed.expand(-1, c, -1)
    assert reconstructed.shape == (b, 2, t)

def test_default_config_loads():
    import yaml
    from pathlib import Path
    config_path = Path('configs/default.yaml')
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    assert 'model' in config
    assert 'data' in config
    assert 'training' in config

