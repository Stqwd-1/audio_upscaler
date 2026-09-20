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

import os
import random
import subprocess
import tempfile

import numpy as np
import torch

from .transforms import MP3Compression, BandwidthLimiter, QuantizationNoise


class AudioAugmentations:
    """Robust training augmentations.

    Simulates real-world degradation: noise, reverb, bandwidth limiting,
    quantization noise, and MP3 compression artifacts.

    Args:
        use_torch: Use PyTorch transforms (no ffmpeg/scipy needed).
                   When False, uses numpy+scipy+ffmpeg (more accurate).
    """
    def __init__(
        self,
        noise_level: float = 0.005,
        reverb_prob: float = 0.3,
        random_resample_prob: float = 0.5,
        bandwidth_limit_prob: float = 0.5,
        quantization_prob: float = 0.3,
        mp3_artifact_prob: float = 0.3,
        dynamic_eq_prob: float = 0.3,
        hard_clip_prob: float = 0.2,
        use_torch: bool = False,
    ):
        self.noise_level = noise_level
        self.reverb_prob = reverb_prob
        self.random_resample_prob = random_resample_prob
        self.bandwidth_limit_prob = bandwidth_limit_prob
        self.quantization_prob = quantization_prob
        self.mp3_artifact_prob = mp3_artifact_prob
        self.dynamic_eq_prob = dynamic_eq_prob
        self.hard_clip_prob = hard_clip_prob
        self.use_torch = use_torch

        # PyTorch transforms (fallback when ffmpeg/scipy unavailable)
        if use_torch:
            self._torch_mp3 = MP3Compression()
            self._torch_bw = BandwidthLimiter(sample_rate=44100)
            self._torch_quant = QuantizationNoise()

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """audio: (samples, channels)"""
        if random.random() < self.random_resample_prob:
            audio = self.random_resample(audio, sr)
        if random.random() < self.reverb_prob:
            audio = self.add_reverb(audio)
        if random.random() < self.bandwidth_limit_prob:
            audio = self.bandwidth_limit(audio, sr)
        if random.random() < self.quantization_prob:
            audio = self.quantization_noise(audio)
        if random.random() < self.mp3_artifact_prob:
            audio = self.mp3_artifacts(audio, sr)
        if random.random() < self.dynamic_eq_prob:
            audio = self.dynamic_eq(audio, sr)
        if random.random() < self.hard_clip_prob:
            audio = self.hard_clip(audio)
        audio = self.add_noise(audio)
        return audio

    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Per-channel independent noise."""
        noise = np.random.randn(*audio.shape).astype(np.float32) * self.noise_level
        return audio + noise

    def add_reverb(self, audio: np.ndarray) -> np.ndarray:
        """Simple delay-based reverb, applied per-channel."""
        delay = random.randint(10, 80)
        decay = random.uniform(0.1, 0.4)
        reverbed = audio.copy()
        if delay < len(audio):
            reverbed[delay:] += audio[:-delay] * decay
        max_val = np.max(np.abs(reverbed))
        if max_val > 0:
            reverbed = reverbed / max_val * np.max(np.abs(audio))
        return reverbed

    def random_resample(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Resample factor, applied uniformly to all channels."""
        factor = random.uniform(0.8, 0.99)
        new_len = int(len(audio) * factor)
        indices = np.linspace(0, len(audio) - 1, new_len).astype(int)
        downsampled = audio[indices]
        indices2 = np.linspace(0, len(downsampled) - 1, len(audio)).astype(int)
        upsampled = downsampled[indices2]
        return upsampled

    def bandwidth_limit(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Low-pass filter to simulate bandwidth-limited sources.

        Randomly cuts frequencies above 4-16 kHz (simulates old recordings, phone audio, etc.)
        Uses PyTorch transforms when use_torch=True (no scipy needed).
        """
        if self.use_torch:
            # (samples, channels) -> (channels, samples)
            t = torch.from_numpy(audio.T).float()
            t = self._torch_bw(t)
            return t.T.numpy().astype(np.float32)

        import scipy.signal as signal

        cutoff = random.uniform(4000, min(16000, sr / 2 - 100))
        nyq = sr / 2
        normalized_cutoff = cutoff / nyq

        order = random.choice([2, 4, 6])
        b, a = signal.butter(order, normalized_cutoff, btype='low')

        if audio.ndim == 1:
            return signal.filtfilt(b, a, audio).astype(np.float32)
        else:
            result = np.zeros_like(audio)
            for ch in range(audio.shape[1]):
                result[:, ch] = signal.filtfilt(b, a, audio[:, ch]).astype(np.float32)
            return result

    def quantization_noise(self, audio: np.ndarray) -> np.ndarray:
        """Simulate low bit-depth quantization noise.

        Reduces to 8-12 bits then back to float32.
        Uses PyTorch transforms when use_torch=True.
        """
        if self.use_torch:
            t = torch.from_numpy(audio.T).float()
            t = self._torch_quant(t)
            return t.T.numpy().astype(np.float32)

        bits = random.randint(8, 12)
        levels = 2 ** bits

        audio_copy = audio.copy()
        min_val = audio_copy.min()
        max_val = audio_copy.max()

        if max_val - min_val < 1e-10:
            return audio_copy

        normalized = (audio_copy - min_val) / (max_val - min_val)
        quantized = np.round(normalized * (levels - 1)) / (levels - 1)
        noise = (quantized - normalized) * 0.3

        return (audio_copy + noise * (max_val - min_val)).astype(np.float32)

    def mp3_artifacts(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Simulate MP3 compression artifacts via roundtrip encode/decode.

        Encodes to MP3 at low bitrate then decodes back.
        Uses PyTorch transforms when use_torch=True (no ffmpeg needed).
        """
        if self.use_torch:
            t = torch.from_numpy(audio.T).float()
            t = self._torch_mp3(t)
            return t.T.numpy().astype(np.float32)

        tmp_paths = []
        try:
            for ext in (".wav", ".mp3", ".wav"):
                # Reserve a name, then release the handle so ffmpeg/soundfile can write to it.
                tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp.close()
                tmp_paths.append(tmp.name)
            tmp_in_path, tmp_mp3_path, tmp_out_path = tmp_paths

            import soundfile as sf
            sf.write(tmp_in_path, audio, sr)

            bitrate = random.choice([64, 96, 128, 160])
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_in_path, "-b:a", f"{bitrate}k", tmp_mp3_path],
                capture_output=True, check=True,
            )

            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_mp3_path, "-ar", str(sr), tmp_out_path],
                capture_output=True, check=True,
            )

            result, _ = sf.read(tmp_out_path, dtype="float32")

            if result.shape[0] < audio.shape[0]:
                result = np.pad(result, ((0, audio.shape[0] - result.shape[0]), (0, 0)))
            else:
                result = result[:audio.shape[0]]

            if result.ndim == 1 and audio.ndim == 2:
                result = np.stack([result, result], axis=1)

            return result.astype(np.float32)

        except Exception:
            return audio
        finally:
            for p in tmp_paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass

    def dynamic_eq(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Dynamic EQ augmentation using torchaudio IIR Biquad filters.

        Randomly applies ONE of three filter types per run:
        1. Low-shelf: random cut/boost on low frequencies
        2. High-shelf: random cut/boost on high frequencies
        3. Peaking EQ: random center freq (100-16kHz), Q, and gain (-12 to +12 dB)

        Args:
            audio: (samples, channels) numpy array
            sr: sample rate

        Returns:
            EQ'd audio as np.float32, normalized to prevent clipping
        """
        import torchaudio.functional as F_audio

        # Convert (samples, channels) -> (channels, samples) for torchaudio
        if audio.ndim == 1:
            waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, samples)
        else:
            waveform = torch.from_numpy(audio.T).float()  # (channels, samples)

        # Choose ONE filter type randomly
        filter_type = random.choice(["low_shelf", "high_shelf", "peaking"])

        if filter_type == "low_shelf":
            cutoff = random.uniform(100, 1000)
            gain_db = random.uniform(-3, 3)
            waveform = F_audio.bass_biquad(waveform, sr, gain_db, cutoff)

        elif filter_type == "high_shelf":
            cutoff = random.uniform(2000, min(16000, sr / 2 - 100))
            gain_db = random.uniform(-3, 3)
            waveform = F_audio.treble_biquad(waveform, sr, gain_db, cutoff)

        else:  # peaking
            center_freq = random.uniform(100, min(16000, sr / 2 - 100))
            gain_db = random.uniform(-2, 2)
            q = random.uniform(0.3, 3.0)
            waveform = F_audio.equalizer_biquad(waveform, sr, center_freq, gain_db, q)

        # Convert back to (samples, channels)
        result = waveform.T.numpy().astype(np.float32)

        # Normalize to prevent clipping
        peak = np.max(np.abs(result))
        if peak > 1.0:
            result = result / peak

        return result

    def hard_clip(self, audio: np.ndarray) -> np.ndarray:
        """Simulate hard digital clipping at random threshold.

        Models distortion from recording levels too hot,
        analog-to-digital converter saturation, or broadcast limiters.
        """
        threshold = random.uniform(0.3, 0.95)
        clipped = np.clip(audio, -threshold, threshold)
        return (clipped / threshold).astype(np.float32)
