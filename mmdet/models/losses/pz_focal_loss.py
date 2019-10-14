import torch.nn as nn
import torch.nn.functional as F

from mmdet.ops import sigmoid_focal_loss as _sigmoid_focal_loss
from ..registry import LOSSES
from .utils import weight_reduce_loss
from IPython import embed
import torch



@LOSSES.register_module
class pz_SigmoidFocalLoss(nn.Module):
    def __init__(self, use_sigmoid, gamma=2.0, alpha=0.25, loss_weight=1.0):
        super(pz_SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight

    def forward(self, logits, targets,weight=None,avg_factor=None, ious=None):
        '''
         :param logits:  shape (N, C), N is nr_box, C is nr_category
         :param targets: shape (N,)
         :return: loss
         '''

        num_classes = logits.shape[1]
        class_range = torch.arange(1, num_classes + 1, dtype=targets.dtype, device=targets.device).unsqueeze(0)
        t = targets.unsqueeze(1)
        labels = (t == class_range).float()

        # Numerical stability version of sigmoid_cross_entropy_with_logits, details are in
        # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
        loss = torch.max(logits, torch.tensor(0, dtype=logits.dtype, device=logits.device)) - logits * labels + \
               torch.log(torch.tensor(1, dtype=logits.dtype, device=logits.device) + torch.exp(-torch.abs(logits)))

        if ious is None:
            #origin focal loss
            focal_loss_pos = self.alpha * labels * loss * (1 - torch.exp(-loss)) ** self.gamma
            focal_loss_neg = (1.0 - self.alpha) * (1 - labels) * (t >= 0).float() * loss * (1.0 - torch.exp(-loss)) ** self.gamma

            focal_loss = focal_loss_pos + focal_loss_neg

        else:
            #iou focal loss, ious shape = [nr_box] --> [nr_box, 80]
            ious = ious[:,None].expand(ious.size(0), num_classes).detach()
            #--------------------------------------------------------------------------------------------
            '''(1-p)**2 替换成 (1-p)**2 * (1+iou)**2'''
            # focal_loss_pos = self.alpha * labels * loss * (1 - torch.exp(-loss)) ** self.gamma * (1 + ious) ** self.gamma
            '''(1-p)**2 替换成 (1+iou)**2'''
            # focal_loss_pos = self.alpha * labels * loss * (1 + ious) ** self.gamma
            '''(1-p)**2 替换成 (1-p)**2 * (1+iou)**2'''
            # focal_loss_pos = self.alpha * labels * loss * (1 - torch.exp(-loss)) ** self.gamma * (1 + ious) ** (self.gamma*2)
            '''(1-p)**2 替换成 (1+p)**2 '''
            focal_loss_pos = self.alpha * labels * loss * (1 + torch.exp(-loss)) ** self.gamma
            #--------------------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------------------
            '''origin focal loss'''
            focal_loss_neg = (1.0 - self.alpha) * (1 - labels) * (t >= 0).float() * loss * (1.0 - torch.exp(-loss)) ** self.gamma
            '''neg_OHEM 只和pos 1+p搭配使用'''
            # focal_loss_neg = neg_OHEM13(loss,targets)
            # --------------------------------------------------------------------------------------------
            focal_loss = focal_loss_pos + focal_loss_neg


        if weight is not None:
            weight = weight.view(-1, 1)
        loss_cls = self.loss_weight * (focal_loss * weight).sum() / avg_factor


        return loss_cls

def neg_OHEM13(loss, targets, min_nr_bg=128):
    loss = loss.sum(dim=1)
    fg_mask = (targets > 0).int()
    bg_mask = (targets == 0).int()
    nr_fg = fg_mask.sum()
    nr_bg = bg_mask.sum()
    min_nr_bg = torch.tensor(min_nr_bg, dtype=nr_bg.dtype, device=nr_bg.device)
    nr_train_bg = torch.max(min_nr_bg, torch.min(nr_bg, nr_fg * 3)).int()
    bg_loss = loss * bg_mask.float()
    sorted_bg_loss, idx = torch.sort(bg_loss, descending=True)
    keep_idx = idx[:nr_train_bg]
    train_bg_loss = bg_loss[keep_idx]
    top_neg, num_classes = train_bg_loss.size()
    pad_train_bg_loss = torch.zeros(targets.size(0), num_classes, dtype=train_bg_loss.dtype, device=train_bg_loss.device)
    pad_train_bg_loss[:top_neg,:] = train_bg_loss
    return pad_train_bg_loss/3

class SigmoidOhem13(nn.Module):
    def __init__(self, min_nr_bg=128):
        super(SigmoidOhem13, self).__init__()
        self.min_nr_bg = min_nr_bg

    def forward(self, logits, targets):
        '''
        sample all foreground, and 3 times of background
        :param logits:  shape (N, C), N is nr_box, C is nr_category
        :param targets: shape (N,)
        :return: loss
        '''

        num_classes = logits.shape[1]
        class_range = torch.arange(1, num_classes + 1, dtype=targets.dtype, device=targets.device).unsqueeze(0)
        t = targets.unsqueeze(1)
        labels = (t == class_range).float()

        loss = torch.max(logits, torch.tensor(0, dtype=logits.dtype, device=logits.device)) - logits * labels + \
               torch.log(torch.tensor(1, dtype=logits.dtype, device=logits.device) + torch.exp(-torch.abs(logits)))

        loss = loss.sum(dim=1)
        fg_mask = (targets > 0).int()
        bg_mask = (targets == 0).int()
        nr_fg = fg_mask.sum()
        nr_bg = bg_mask.sum()
        min_nr_bg = torch.tensor(self.min_nr_bg, dtype=nr_bg.dtype, device=nr_bg.device)
        nr_train_bg = torch.max(min_nr_bg, torch.min(nr_bg, nr_fg * 3)).int()

        fg_loss = loss * fg_mask.float()
        bg_loss = loss * bg_mask.float()

        sorted_bg_loss, idx = torch.sort(bg_loss, descending=True)
        keep_idx = idx[:nr_train_bg]
        train_bg_loss = bg_loss[keep_idx]

        sum_loss = fg_loss.sum() + train_bg_loss.sum()

        # Since the loss will be divided by nr_fg outside of the function,
        # here 4 means (nr_fg + nr_bg) = (nr_fg + 3 * nr_fg) = 4 * nr_fg
        return sum_loss / 4