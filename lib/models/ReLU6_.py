# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class ReLU6_(nn.Module):
    def __init__(self):
        super(ReLU6_, self).__init__()
        self.model = nn.Sequential(
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x).clamp(max=6)
