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

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

SUPPORTED_FORMATS = {".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus", ".aiff", ".aif"}


def load_audio(path: str | Path, target_sr: int | None = None, mono: bool = False) -> tuple[np.ndarray, int]:
    """Load audio file. Returns (audio, sr).

    audio shape: (samples,) for mono, (samples, channels) for multi-channel.
    If mono=True, multi-channel audio is mixed down to mono.
    """
    path = Path(path)
    try:
        if path.suffix.lower() in {".wav", ".flac", ".aiff", ".aif"}:
            audio, sr = sf.read(path, dtype="float32")
        else:
            audio, sr = _load_via_ffmpeg(path, target_sr or 44100)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Failed to load {path}: {e}")

    if mono and audio.ndim > 1:
        audio = audio.mean(axis=1)

    if target_sr is not None and sr != target_sr:
        audio = _resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def _load_via_ffmpeg(path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    """Load any format via ffmpeg, preserving channels."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(path), "-ar", str(target_sr), "-f", "wav", tmp_path],
        capture_output=True,
        check=True,
    )
    audio, sr = sf.read(tmp_path, dtype="float32")
    Path(tmp_path).unlink(missing_ok=True)
    return audio, sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import torch
    import torchaudio

    if audio.ndim == 1:
        waveform = torch.from_numpy(audio).unsqueeze(0).float()
    else:
        waveform = torch.from_numpy(audio.T).float()  # (channels, samples)

    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    resampled = resampler(waveform)

    if audio.ndim == 1:
        return resampled.squeeze(0).numpy()
    return resampled.T.numpy()  # Back to (samples, channels)


def save_audio(path: str | Path, audio: np.ndarray, sr: int):
    """Save audio. audio shape: (samples,) or (samples, channels)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sr)
