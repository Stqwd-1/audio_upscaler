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

import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .augmentations import AudioAugmentations
from .formats import SUPPORTED_FORMATS, load_audio


class AudioDataset(Dataset):
    """Loads stereo audio only.

    All audio (mono, stereo, multi-channel) is converted to 2-channel stereo.
    Model is trained exclusively for stereo inference.

    Args:
        mode: "train" or "val", controls augmentation and file split.
        val_split: fraction of files held out for validation (0.0 = no split).
        split_seed: random seed for reproducible train/val split.
    """
    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 44100,
        segment_length: float = 1.0,
        low_sample_rate: int = 16000,
        augment: bool = True,
        augmentation_config: dict | None = None,
        val_split: float = 0.0,
        split_seed: int = 42,
        mode: str = "train",
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_length)
        self.low_sample_rate = low_sample_rate
        self.augment = augment and mode == "train"
        self.mode = mode

        all_files = sorted(
            p for p in self.data_dir.rglob("*")
            if p.suffix.lower() in SUPPORTED_FORMATS
        )
        valid_files = [f for f in all_files if self._is_loadable(f)]
        if not valid_files:
            raise RuntimeError(f"No audio files found in {data_dir}")

        # Train/val split
        if val_split > 0:
            rng = random.Random(split_seed)
            indices = list(range(len(valid_files)))
            rng.shuffle(indices)
            split_idx = max(1, int(len(valid_files) * (1 - val_split)))
            if mode == "train":
                self.files = [valid_files[i] for i in indices[:split_idx]]
            else:
                self.files = [valid_files[i] for i in indices[split_idx:]]
            if not self.files:
                self.files = [valid_files[indices[0]]]  # fallback
        else:
            self.files = valid_files

        if augmentation_config and self.augment:
            self.augmentations = AudioAugmentations(**augmentation_config)
        else:
            self.augmentations = None

        # Cache resamplers (created once, reused per sample)
        self._resampler_down = torchaudio.transforms.Resample(
            orig_freq=self.sample_rate, new_freq=self.low_sample_rate
        )
        self._resampler_up = torchaudio.transforms.Resample(
            orig_freq=self.low_sample_rate, new_freq=self.sample_rate
        )

    @staticmethod
    def _is_loadable(path: Path) -> bool:
        """Check a file is decodable without decoding the whole thing.

        Reads only headers via torchaudio; falls back to a real (slow)
        load for containers torchaudio can't identify.
        """
        try:
            info = torchaudio.info(path)
            return info.num_channels >= 1
        except Exception:  # noqa: BLE001
            try:
                audio, _ = load_audio(path, None)
                return audio.size > 0
            except Exception:  # noqa: BLE001
                return False

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        audio, _sr = load_audio(self.files[idx], self.sample_rate)

        # --- Always stereo: mono -> duplicate, multi-channel -> take first 2 ---
        if audio.ndim == 1:
            # Mono: duplicate to stereo
            audio = np.stack([audio, audio], axis=1)
        elif audio.ndim == 2 and audio.shape[1] > 2:
            # Multi-channel: take first 2 channels (L, R)
            audio = audio[:, :2]
        elif audio.ndim == 2 and audio.shape[1] == 1:
            # Single channel in (samples, 1) format
            audio = np.concatenate([audio, audio], axis=1)

        # Guarantee shape is (samples, 2)
        assert audio.ndim == 2 and audio.shape[1] == 2, f"Expected (samples, 2), got {audio.shape}"

        if audio.shape[0] < self.segment_samples:
            audio = np.pad(audio, ((0, self.segment_samples - audio.shape[0]), (0, 0)))
        start = np.random.randint(0, max(1, audio.shape[0] - self.segment_samples))
        audio = audio[start : start + self.segment_samples]

        if self.augment and self.augmentations is not None:
            audio = self.augmentations(audio, self.sample_rate)

        low_audio = self._downsample(audio)

        # Always output (2, samples), stereo only
        high = torch.from_numpy(audio.T).float()  # (2, samples)
        low = torch.from_numpy(low_audio.T).float()  # (2, samples)

        return {"audio": high, "low_audio": low}

    def _downsample(self, audio: np.ndarray) -> np.ndarray:
        """Downsample stereo audio: 44100 -> 16000 -> 44100.

        Input must be (samples, 2). Output is (samples, 2).
        """
        assert audio.ndim == 2 and audio.shape[1] == 2, f"Expected (samples, 2), got {audio.shape}"
        waveform = torch.from_numpy(audio.T).float()  # (2, samples)
        downsampled = self._resampler_down(waveform)
        upsampled = self._resampler_up(downsampled)
        return upsampled.T.numpy()  # (samples, 2)


def create_dataloaders(config: dict, data_dir: str) -> tuple[DataLoader, DataLoader | None]:
    """Create train and optional val DataLoaders from config.

    Returns (train_loader, val_loader). val_loader is None if val_split=0.
    """
    data_cfg = config["data"]
    aug_cfg = data_cfg.get("augmentation", None)
    val_split = data_cfg.get("val_split", 0.0)
    split_seed = data_cfg.get("split_seed", 42)

    train_ds = AudioDataset(
        data_dir,
        sample_rate=data_cfg["sample_rate"],
        segment_length=data_cfg["segment_length"],
        low_sample_rate=data_cfg["low_sample_rate"],
        augment=True,
        augmentation_config=aug_cfg,
        val_split=val_split,
        split_seed=split_seed,
        mode="train",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg["num_workers"],
        pin_memory=False,
    )

    val_loader = None
    if val_split > 0:
        val_ds = AudioDataset(
            data_dir,
            sample_rate=data_cfg["sample_rate"],
            segment_length=data_cfg["segment_length"],
            low_sample_rate=data_cfg["low_sample_rate"],
            augment=False,
            augmentation_config=None,
            val_split=val_split,
            split_seed=split_seed,
            mode="val",
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=data_cfg["batch_size"],
            shuffle=False,
            num_workers=data_cfg["num_workers"],
            pin_memory=False,
        )

    return train_loader, val_loader
