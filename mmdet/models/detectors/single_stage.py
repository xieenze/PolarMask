import torch.nn as nn

from mmdet.core import bbox2result, bbox_mask2result
from .. import builder
from ..registry import DETECTORS
from .base import BaseDetector
from IPython import embed
import time
import torch

@DETECTORS.register_module
class SingleStageDetector(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SingleStageDetector, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.init_weights(pretrained=pretrained)

        #FPS
        self.time = 0
        self.count = 0

    def init_weights(self, pretrained=None):
        super(SingleStageDetector, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()

    def extract_feat(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

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

    def aug_test(self, imgs, img_metas, rescale=False):
        raise NotImplementedError
