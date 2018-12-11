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
import scipy.ndimage.filters as fi


class JointsMSELossCoco(nn.Module):
    def __init__(self, use_target_weight, heatmap_size):
        super(JointsMSELossCoco, self).__init__()
        self.criterion = nn.MSELoss(size_average=True)
        self.use_target_weight = use_target_weight
        self.col = float(heatmap_size)
        self.scale = 224./float(self.col)
        self.gaussian = 1.0

    def min_max(self, x, axis=None):
        min = x.min(axis=axis, keepdims=True)
        max = x.max(axis=axis, keepdims=True)
        result = (x-min)/(max-min)
        return torch.Tensor(result)

    def checkMatrix(self, xi, yi):
        f = False
        if xi >= 0 and xi <= self.col - 1 and yi >= 0 and yi <= self.col - 1:
            f = True
        return xi, yi, f

    def forward(self, offset, heatmap, target, target_weight, meta, isValid=False, useOffset=False):
        batch_size = heatmap.size(0)
        num_joints = heatmap.size(1)
        
        joints = meta['joints']
        joints_vis = meta['joints_vis']
        joints = joints[:, :, :2].float().cuda()
        joints_vis = joints_vis[:, :, :2].float().cuda()
        x = Variable(torch.zeros(joints.size()).float(), requires_grad=True).cuda()

        '''
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
        '''
        reshaped = heatmap.view(-1, num_joints, int(self.col*self.col))
        _, argmax = reshaped.max(-1)
        yCoords = argmax/self.col
        xCoords = argmax - yCoords*self.col

        s = heatmap.size()
        tt = torch.zeros(s).float()
        ti = joints/self.scale

        for i in range(batch_size):
            for j in range(num_joints):
                #if h[i, j, yCoords[i, j], xCoords[i, j]] > 0.5:
                x[i, j, 0] = (offset[i, j, yCoords[i, j], xCoords[i, j]] + xCoords[i, j].float()) * self.scale
                x[i, j, 1] = (offset[i, j + num_joints, yCoords[i, j], xCoords[i, j]] + yCoords[i, j].float()) * self.scale

                if int(target_weight[i, j, 0]) >= 0.5:
                    xi, yi, f = self.checkMatrix(int(ti[i, j, 0]), int(ti[i, j, 1]))
                    
                    if f == True:
                        # 正規分布に近似したサンプルを得る
                        # 平均は 100 、標準偏差を 1 
                        tt[i, j, yi, xi]  = 1
                        tt[i, j] = self.min_max(fi.gaussian_filter(tt[i, j], self.gaussian))
                    else:
                        target_weight[i, j, 0] = 0
                        #target_weight[i, j, 1] = 0
        
        diff1 = heatmap - target
        '''
        cnt = 0
        for i in range(batch_size):
            for j in range(num_joints):
                if int(target_weight[i, j, 0]) == 0:
                    diff1[i, j] = diff1[i, j]*0
                else:
                    cnt = cnt + 1
        diff1 = diff1.view(-1)
        d1 = diff1.dot(diff1) / cnt
        '''
        diff1 = diff1.view(-1)
        d1 = diff1.dot(diff1) / (batch_size*num_joints)

        if useOffset == False:
            return d1, x, tt, target_weight

        diff2 = (x - joints)
        
        diff2 = diff2*joints_vis/112.
        N2 = (joints_vis.sum()).data[0]/2.0
        diff2 = diff2.view(-1)
        d2 = 0.5 * torch.sqrt(diff2.dot(diff2))/N2
        
        return d1 + d2, x, tt, target_weight
