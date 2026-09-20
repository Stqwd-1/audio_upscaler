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

import torch
import torch.nn as nn


class WaveLoss(nn.Module):
    """L1 + multi-scale L1 loss — works on any device including DirectML."""
    def __init__(self, scales=(1, 2, 4), loss_weight: float = 1.0):
        super().__init__()
        self.scales = scales
        self.l1 = nn.L1Loss()
        self.loss_weight = loss_weight

    def forward(self, pred, target):
        loss = self.l1(pred, target)
        for s in self.scales:
            if s > 1:
                p = pred[..., ::s]
                t = target[..., ::s]
                min_len = min(p.shape[-1], t.shape[-1])
                loss = loss + self.l1(p[..., :min_len], t[..., :min_len])
        return self.loss_weight * loss / len(self.scales)
