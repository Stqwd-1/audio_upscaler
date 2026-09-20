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

"""Hardware detection and VRAM-aware utilities."""
import gc

import torch


def get_device_info():
    """Get device info (CUDA, DirectML, or CPU)."""
    try:
        import torch_directml
        return {
            "type": "directml",
            "name": torch_directml.device_name(0),
            "device": torch_directml.device(),
        }
    except ImportError:
        pass

    if torch.cuda.is_available():
        return {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
            "device": torch.device("cuda"),
            "vram_gb": torch.cuda.get_device_properties(0).total_mem / 1024**3,
        }

    return {"type": "cpu", "name": "CPU", "device": torch.device("cpu")}


def auto_batch_size(model, max_batch: int = 16, input_channels: int = 2, sample_rate: int = 44100, segment_sec: float = 1.0):
    """Binary search for optimal batch size that fits in VRAM.

    Returns the largest batch_size that doesn't OOM.
    """
    device_info = get_device_info()
    device = device_info["device"]

    if device_info["type"] == "cpu":
        return 1

    segment_samples = int(sample_rate * segment_sec)
    best_batch = 1

    for bs in [1, 2, 4, 8, 16, 32]:
        if bs > max_batch:
            break
        try:
            gc.collect()
            if device_info["type"] == "cuda":
                torch.cuda.empty_cache()

            dummy = torch.randn(bs, input_channels, segment_samples, device=device)
            with torch.no_grad():
                _ = model(dummy)
            del dummy
            best_batch = bs
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            break

    gc.collect()
    if device_info["type"] == "cuda":
        torch.cuda.empty_cache()

    return best_batch
