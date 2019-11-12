from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import torch.nn as nn

from mmdet.core import bbox2result, bbox_mask2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from IPython import embed
import time
import torch


@DETECTORS.register_module
class PolarMask(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(PolarMask, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      _gt_labels=None,
                      _gt_bboxes=None,
                      _gt_masks=None
                      ):

        if _gt_labels is not None:
            extra_data = dict(_gt_labels=_gt_labels,
                              _gt_bboxes=_gt_bboxes,
                              _gt_masks=_gt_masks)
        else:
            extra_data = None


        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes, gt_labels, img_metas, self.train_cfg)

        losses = self.bbox_head.loss(
            *loss_inputs,
            gt_masks = gt_masks,
            gt_bboxes_ignore=gt_bboxes_ignore,
            extra_data=extra_data
        )
        return losses


    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)

        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)

        results = [
            bbox_mask2result(det_bboxes, det_masks, det_labels, self.bbox_head.num_classes, img_meta[0])
            for det_bboxes, det_labels, det_masks in bbox_list]

        bbox_results = results[0][0]
        mask_results = results[0][1]

        return bbox_results, mask_results

