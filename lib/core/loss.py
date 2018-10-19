# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight, heatmap_size):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
        self.col = float(heatmap_size)
        self.scale = 1./float(self.col)

    def forward(self, offset, heatmap, target, target_weight, meta, isValid=False, useOffset=False):
        batch_size = heatmap.size(0)
        num_joints = heatmap.size(1)
        
        joints = meta['joints']
        joints_vis = meta['joints_vis']
        joints = joints[:, :, :2].float().cuda()
        joints_vis = joints_vis[:, :, :2].float().cuda()
        x = Variable(torch.zeros(joints.size()).float(), requires_grad=True).cuda()

        heatmaps_pred = heatmap.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(heatmap_pred.mul(target_weight[:, idx]), heatmap_gt.mul(target_weight[:, idx]))
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        d1 = loss / num_joints

        if useOffset == False:
            return d1, x

        # loss offset
        reshaped = heatmap.view(-1, num_joints, self.col*self.col)
        _, argmax = reshaped.max(-1)
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col

        for i in range(batch_size):
            for j in range(num_joints):
                #if heatmap[i, j, yCoords[i, j], xCoords[i, j]] > 0.5:
                x[i, j, 0] = (offset[i, j, yCoords[i, j], xCoords[i, j]] + xCoords[i, j].float()) * self.col
                x[i, j, 1] = (offset[i, j + 16, yCoords[i, j], xCoords[i, j]] + yCoords[i, j].float()) * self.col

        diff2 = (x - joints)
        diff2 = diff2*joints_vis/256.
        N2 = (joints_vis.sum()).data[0]/2.0
        diff2 = diff2.view(-1)
        d2 = 0.5 * torch.sqrt(diff2.dot(diff2))/N2

        return d1 + d2, x
