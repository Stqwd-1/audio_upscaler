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

"""Print model architecture and training metadata from a checkpoint.

Usage:
    python model_info.py [checkpoint_path]
"""
import argparse
import os

import torch

from models.sr_network import SRNetwork


def main():
    parser = argparse.ArgumentParser(description="Show model info from a checkpoint")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = ckpt["config"]
    sr_cfg = config["model"]["sr_network"]

    model = SRNetwork(**sr_cfg)
    total = sum(p.numel() for p in model.parameters())
    ckpt_size = os.path.getsize(args.checkpoint)

    print("=" * 50)
    print("MODEL: Audio Upscaler")
    print("=" * 50)
    print()
    print("--- Architecture ---")
    print("Type: U-Net + ResBlocks")
    print(f"In channels: {sr_cfg['in_channels']}")
    print(f"Out channels: {sr_cfg['out_channels']}")
    print(f"Base channels: {sr_cfg['base_channels']}")
    print(f"Channel multipliers: {sr_cfg['channel_multipliers']}")
    print(f"Num res blocks: {sr_cfg['num_res_blocks']}")
    print()
    print("--- Parameters ---")
    print(f"Total: {total:,} ({total * 4 / 1024**2:.1f} MB FP32)")
    print(f"Checkpoint size: {ckpt_size / 1024**2:.1f} MB")
    print()
    print("--- Checkpoint keys ---")
    print(f"Epoch: {ckpt.get('epoch')}")
    print(f"Keys: {list(ckpt.keys())}")


if __name__ == "__main__":
    main()