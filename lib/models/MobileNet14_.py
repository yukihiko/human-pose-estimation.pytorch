# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

class MobileNet14_(nn.Module):
    def __init__(self):
        super(MobileNet14_, self).__init__()
        self.col = 14
        self.Nj = 16

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 1),
            conv_dw(1024, 1024, 1),
        )
        #self.heatmap = nn.Conv2d(1024, self.Nj, 1, bias=False)
        #self.offset = nn.Conv2d(1024, self.Nj*2, 1, bias=False)
        self.output = nn.Conv2d(1024, self.Nj*3, 1, bias=False)

    def forward(self, x):
        x = self.model(x)
        #h = self.heatmap(x)
        #h = F.sigmoid(h)
        #o = self.offset(x)
        o = self.output(x)

        return o
