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
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, outputs: list[torch.Tensor], is_real: bool = True) -> torch.Tensor:
        loss = 0.0
        for out in outputs:
            if is_real:
                loss += torch.mean((1 - out) ** 2)
            else:
                loss += torch.mean(out ** 2)
        return self.loss_weight * loss / len(outputs)


class FeatureMatchingLoss(nn.Module):
    def __init__(self, loss_weight: float = 2.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, real_features: list, fake_features: list) -> torch.Tensor:
        loss = 0.0
        count = 0
        for real_f, fake_f in zip(real_features, fake_features):
            for r, f in zip(real_f, fake_f):
                loss += F.l1_loss(f, r.detach())
                count += 1
        return self.loss_weight * loss / max(count, 1)
