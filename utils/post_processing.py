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

"""Post-processing for audio upscaling.

- Transient restoration: Restore punch from original
"""
import torch


class AudioPostProcessor:
    """Post-processing utilities for audio enhancement."""

    @staticmethod
    def restore_transients(original, upscaled, strength=0.5):
        """Restore transient punch from original low-res audio.

        strength: 0.0 (no restore) to 1.0 (full restore)
        """
        if strength <= 0:
            return upscaled

        min_len = min(original.shape[-1], upscaled.shape[-1])
        original = original[..., :min_len]
        upscaled = upscaled[..., :min_len]

        env_orig = torch.abs(original)
        env_up = torch.abs(upscaled)

        diff = env_orig - env_up
        mask = (diff > 0).float()

        return upscaled + (diff * mask * strength)
