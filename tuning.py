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

"""Optuna hyperparameter optimization for audio upscaler.

Usage:
    python tuning.py --data-dir D:\\path\to\\library --trials 20
"""
import argparse
import os
import sys

import optuna
import torch

sys.path.insert(0, '.')

from torch.utils.data import DataLoader

from data.dataset import AudioDataset
from losses import SpectralLoss
from models.sr_network import SRNetwork


def objective(trial, data_dir: str, device):
    """Optuna objective function — minimizes validation loss."""
    # Hyperparameters to search
    base_channels = trial.suggest_categorical("base_channels", [32, 48, 64])
    num_res_blocks = trial.suggest_int("num_res_blocks", 4, 12, step=2)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4])
    segment_length = trial.suggest_float("segment_length", 0.5, 2.0, step=0.5)

    channel_multipliers = [1, 2, 4, 8] if base_channels >= 48 else [1, 2, 4]

    # Build model
    model = SRNetwork(
        in_channels=2,
        out_channels=2,
        base_channels=base_channels,
        channel_multipliers=channel_multipliers,
        num_res_blocks=num_res_blocks,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    if params > 50_000_000:  # Skip if too large
        return float('inf')

    # Dataset
    dataset = AudioDataset(
        data_dir,
        sample_rate=44100,
        segment_length=segment_length,
        low_sample_rate=16000,
        augment=True,
    )

    # Split train/val (80/20)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    # Optimizer + Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.5, 0.9))
    criterion = SpectralLoss()

    # Quick training (3 epochs for search)
    model.train()
    for epoch in range(3):
        for batch in train_loader:
            low = batch["low_audio"].to(device)
            high = batch["audio"].to(device)

            pred = model(low)
            min_len = min(pred.shape[-1], high.shape[-1])
            loss = criterion(pred[..., :min_len], high[..., :min_len])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            low = batch["low_audio"].to(device)
            high = batch["audio"].to(device)
            pred = model(low)
            min_len = min(pred.shape[-1], high.shape[-1])
            val_loss += criterion(pred[..., :min_len], high[..., :min_len]).item()

    val_loss /= max(len(val_loader), 1)
    return val_loss


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--output", default="configs/best_params.yaml")
    args = parser.parse_args()

    try:
        import torch_directml
        device = torch_directml.device()
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Trials: {args.trials}")

    study = optuna.create_study(direction="minimize", study_name="audio_upscaler")
    study.optimize(lambda trial: objective(trial, args.data_dir, device), n_trials=args.trials)

    # Save best params
    best = study.best_params
    print(f"\nBest trial: {study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print(f"Best params: {best}")

    # Save as YAML
    import yaml
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        yaml.dump(best, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
