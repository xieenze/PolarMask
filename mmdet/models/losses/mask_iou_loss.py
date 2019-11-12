import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss
from IPython import embed
import torch

@LOSSES.register_module
class MaskIOULoss(nn.Module):
    def __init__(self):
        super(MaskIOULoss, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred,target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]

        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss




@LOSSES.register_module
class MaskIOULoss_v2(nn.Module):
    def __init__(self):
        super(MaskIOULoss_v2, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred,target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0].clamp(min=1e-6)

        # loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = (l_max / l_min).log().mean(dim=1)
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss

@LOSSES.register_module
class MaskIOULoss_v3(nn.Module):
    def __init__(self):
        super(MaskIOULoss_v3, self).__init__()

    def forward(self, pred, target, weight, avg_factor=None):
        '''
         :param pred:  shape (N,36), N is nr_box
         :param target: shape (N,36)
         :return: loss
         '''

        total = torch.stack([pred,target], -1)
        l_max = total.max(dim=2)[0].pow(2)
        l_min = total.min(dim=2)[0].pow(2)

        # loss = 2 * (l_max.prod(dim=1) / l_min.prod(dim=1)).log()
        # loss = 2 * (l_max.log().sum(dim=1) - l_min.log().sum(dim=1))
        loss = (l_max.sum(dim=1) / l_min.sum(dim=1)).log()
        loss = loss * weight
        loss = loss.sum() / avg_factor
        return loss

