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

"""Inspect the model configuration and metadata stored in a checkpoint.

Usage:
    python check_config.py [checkpoint_path]
"""
import sys

import torch

checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/best_model.pt"
ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
print("Model config from checkpoint:")
print(ckpt["config"]["model"]["sr_network"])
print("Epoch:", ckpt["epoch"])