from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import cv2
import math
import Polygon as plg
from tqdm import tqdm

from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS
import os.path as osp
import warnings

import mmcv
import numpy as np
from imagecorruptions import corrupt
from mmcv.parallel import DataContainer as DC
from torch.utils.data import Dataset
import torch

from .extra_aug import ExtraAugmentation
from .registry import DATASETS
from .transforms import (BboxTransform, ImageTransform, MaskTransform,
                         Numpy2Tensor, SegMapTransform, SegmapTransform)
from .utils import random_scale, to_tensor
from IPython import embed

INF = 1e8


def get_angle(v1, v2=[0,0,100,0]):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = int(angle1 * 180/math.pi)
    angle2 = math.atan2(dy2, dx2)
    angle2 = int(angle2 * 180/math.pi)
    included_angle = angle2 - angle1
    if included_angle < 0:
        included_angle += 360
    return included_angle


@DATASETS.register_module
class Coco_Seg_Dataset(CustomDataset):
    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')


    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.


        self.debug = False

        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []

        if self.debug:
            count = 0
            total = 0
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            #filter bbox < 10
            if self.debug:
                total+=1

            if ann['area'] <= 15 or (w < 10 and h < 10) or self.coco.annToMask(ann).sum() < 15:
                # print('filter, area:{},w:{},h:{}'.format(ann['area'],w,h))
                if self.debug:
                    count+=1
                continue

            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)

        if self.debug:
            print('filter:',count/total)
        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        return ann

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # corruption
        if self.corruption is not None:
            img = corrupt(
                img,
                severity=self.corruption_severity,
                corruption_name=self.corruption)
        # load proposals if necessary
        if self.proposals is not None:
            proposals = self.proposals[idx][:self.num_max_proposals]
            # TODO: Handle empty proposals properly. Currently images with
            # no proposals are just ignored, but they can be used for
            # training in concept.
            if len(proposals) == 0:
                return None
            if not (proposals.shape[1] == 4 or proposals.shape[1] == 5):
                raise AssertionError(
                    'proposals should have shapes (n, 4) or (n, 5), '
                    'but found {}'.format(proposals.shape))
            if proposals.shape[1] == 5:
                scores = proposals[:, 4, None]
                proposals = proposals[:, :4]
            else:
                scores = None

        ann = self.get_ann_info(idx)

        gt_bboxes = ann['bboxes']
        gt_labels = ann['labels']
        if self.with_crowd:
            gt_bboxes_ignore = ann['bboxes_ignore']

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0 and self.skip_img_without_anno:
            warnings.warn('Skip the image "%s" that has no valid gt bbox' %
                          osp.join(self.img_prefix, img_info['filename']))
            return None

        # apply transforms
        flip = True if np.random.rand() < self.flip_ratio else False
        # randomly sample a scale
        img_scale = random_scale(self.img_scales, self.multiscale_mode)
        img, img_shape, pad_shape, scale_factor = self.img_transform(img, img_scale, flip, keep_ratio=self.resize_keep_ratio)


        img = img.copy()
        if self.with_seg:
            gt_seg = mmcv.imread(
                osp.join(self.seg_prefix,
                         img_info['filename'].replace('jpg', 'png')),
                flag='unchanged')
            gt_seg = self.seg_transform(gt_seg.squeeze(), img_scale, flip)
            gt_seg = mmcv.imrescale(
                gt_seg, self.seg_scale_factor, interpolation='nearest')
            gt_seg = gt_seg[None, ...]
        if self.proposals is not None:
            proposals = self.bbox_transform(proposals, img_shape, scale_factor,
                                            flip)
            proposals = np.hstack([proposals, scores
                                   ]) if scores is not None else proposals
        gt_bboxes = self.bbox_transform(gt_bboxes, img_shape, scale_factor,
                                        flip)
        if self.with_crowd:
            gt_bboxes_ignore = self.bbox_transform(gt_bboxes_ignore, img_shape,
                                                   scale_factor, flip)
        if self.with_mask:
            gt_masks = self.mask_transform(ann['masks'], pad_shape,
                                           scale_factor, flip)

        ori_shape = (img_info['height'], img_info['width'], 3)
        img_meta = dict(
            ori_shape=ori_shape,
            img_shape=img_shape,
            pad_shape=pad_shape,
            scale_factor=scale_factor,
            flip=flip)

        data = dict(
            img=DC(to_tensor(img), stack=True),
            img_meta=DC(img_meta, cpu_only=True),
            gt_bboxes=DC(to_tensor(gt_bboxes)))

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
        if self.with_crowd:
            data['gt_bboxes_ignore'] = DC(to_tensor(gt_bboxes_ignore))
        if self.with_mask:
            data['gt_masks'] = DC(gt_masks, cpu_only=True)


        #--------------------offline ray label generation-----------------------------

        self.center_sample = True
        self.use_mask_center = True
        self.radius = 1.5
        self.strides = [8, 16, 32, 64, 128]
        self.regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF))
        featmap_sizes = self.get_featmap_size(pad_shape)
        self.featmap_sizes = featmap_sizes
        num_levels = len(self.strides)
        all_level_points = self.get_points(featmap_sizes)
        self.num_points_per_level = [i.size()[0] for i in all_level_points]

        expanded_regress_ranges = [
            all_level_points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                all_level_points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(all_level_points, 0)
        gt_masks = gt_masks[:len(gt_bboxes)]

        gt_bboxes = torch.Tensor(gt_bboxes)
        gt_labels = torch.Tensor(gt_labels)

        _labels, _bbox_targets, _mask_targets = self.fcos_target_single(
            gt_bboxes,gt_masks,gt_labels,concat_points, concat_regress_ranges)

        data['_gt_labels'] = DC(_labels)
        data['_gt_bboxes'] = DC(_bbox_targets)
        data['_gt_masks'] = DC(_mask_targets)
        #--------------------offline ray label generation-----------------------------


        return data

    def get_featmap_size(self, shape):
        h,w = shape[:2]
        featmap_sizes = []
        for i in self.strides:
            featmap_sizes.append([int(h / i), int(w / i)])
        return featmap_sizes

    def get_points(self, featmap_sizes):
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i]))
        return mlvl_points

    def get_points_single(self, featmap_size, stride):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride)
        y_range = torch.arange(
            0, h * stride, stride)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        return points.float()

    def fcos_target_single(self, gt_bboxes, gt_masks, gt_labels, points, regress_ranges):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)

        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        #xs ys 分别是points的x y坐标
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)   #feature map上所有点对于gtbox的上下左右距离 [num_pix, num_gt, 4]


        #mask targets 也按照这种写 同时labels 得从bbox中心修改成mask 重心
        mask_centers = []
        mask_contours = []
        #第一步 先算重心  return [num_gt, 2]
        for mask in gt_masks:
            cnt, contour = self.get_single_centerpoint(mask)
            contour = contour[0]
            contour = torch.Tensor(contour).float()

            y, x = cnt
            mask_centers.append([x,y])
            mask_contours.append(contour)
        mask_centers = torch.Tensor(mask_centers).float()
        # 把mask_centers assign到不同的层上,根据regress_range和重心的位置
        mask_centers = mask_centers[None].expand(num_points, num_gts, 2)

        #-------------------------------------------------------------------------------------------------------------------------------------------------------------
        # condition1: inside a gt bbox
        #加入center sample
        if self.center_sample:
            strides = [8, 16, 32, 64, 128]
            if self.use_mask_center:
                inside_gt_bbox_mask = self.get_mask_sample_region(gt_bboxes,
                                                             mask_centers,
                                                             strides,
                                                             self.num_points_per_level,
                                                             xs,
                                                             ys,
                                                             radius=self.radius)
            else:
                inside_gt_bbox_mask = self.get_sample_region(gt_bboxes,
                                                             strides,
                                                             self.num_points_per_level,
                                                             xs,
                                                             ys,
                                                             radius=self.radius)
        else:
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]


        inside_regress_range = (
            max_regress_distance >= regress_ranges[..., 0]) & (
            max_regress_distance <= regress_ranges[..., 1])

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0         #[num_gt] 介于0-80


        bbox_targets = bbox_targets[range(num_points), min_area_inds]


        pos_inds = labels.nonzero().reshape(-1)

        m = []
        for p in range(num_points):
            if self.debug:
                vis = torch.zeros(gt_masks[0].shape).int().data.numpy()
            #不是pos就随便加
            if p not in pos_inds:
                tmp = torch.zeros(36)
                m.append(tmp)
            #是pos的话  算重心
            else:
                pos_mask_id = min_area_inds[p]
                pos_mask = gt_masks[pos_mask_id]
                pos_mask_contour = mask_contours[pos_mask_id]
                x,y = points[p]
                #计算36射线距离
                dists, coords = self.get_36_coordinates(x,y,pos_mask_contour)
                m.append(dists)

                # debug visual code
                if self.debug:
                    print('debug visual'*10)
                    for j in coords.keys():
                        vis = pos_mask
                        vis = cv2.circle(vis, (x, y), 3, 2, -1)
                        x1, y1 = coords[j]
                        vis = cv2.line(vis, (x, y), (x1, y1), 2, 1)
                    cv2.imwrite('./trash/{}.jpg'.format(pos_mask_id),vis*127)


        mask_targets = torch.stack(m, 0).float()


        return labels, bbox_targets, mask_targets

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1):
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt[beg:end, :, 2], gt[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt[beg:end, :, 3], gt[beg:end, :, 3], ymax)
            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面
        return inside_gt_bbox_mask

    def get_mask_sample_region(self, gt_bb, mask_center, strides, num_points_per, gt_xs, gt_ys, radius=1):
        center_y = mask_center[..., 0]
        center_x = mask_center[..., 1]
        center_gt = gt_bb.new_zeros(gt_bb.shape)
        #no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

        beg = 0
        for level,n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(xmin > gt_bb[beg:end, :, 0], xmin, gt_bb[beg:end, :, 0])
            center_gt[beg:end, :, 1] = torch.where(ymin > gt_bb[beg:end, :, 1], ymin, gt_bb[beg:end, :, 1])
            center_gt[beg:end, :, 2] = torch.where(xmax > gt_bb[beg:end, :, 2], gt_bb[beg:end, :, 2], xmax)
            center_gt[beg:end, :, 3] = torch.where(ymax > gt_bb[beg:end, :, 3], gt_bb[beg:end, :, 3], ymax)
            beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面
        return inside_gt_bbox_mask

    def get_centerpoint(self, lis):
        area = 0.0
        x, y = 0.0, 0.0
        a = len(lis)
        for i in range(a):
            lat = lis[i][0]
            lng = lis[i][1]
            if i == 0:
                lat1 = lis[-1][0]
                lng1 = lis[-1][1]
            else:
                lat1 = lis[i - 1][0]
                lng1 = lis[i - 1][1]
            fg = (lat * lng1 - lng * lat1) / 2.0
            area += fg
            x += fg * (lat + lat1) / 3.0
            y += fg * (lng + lng1) / 3.0
        x = x / area
        y = y / area

        return [int(x), int(y)]

    def get_single_centerpoint(self, mask):
        # if type(mask) == torch.Tensor:
        #     mask = mask.data.cpu().numpy().astype('uint8')
        contour, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contour.sort(key=lambda x: cv2.contourArea(x), reverse=True) #only save the biggest one
        '''debug IndexError: list index out of range'''
        count = contour[0][:, 0, :]
        try:
            center = self.get_centerpoint(count)
        except:
            x,y = count.mean(axis=0)
            center=[int(x), int(y)]
        return center, contour

    def get_36_coordinates(self, c_x, c_y, pos_mask_contour):
        #计算contour上所有点和角度
        coordinate = {}
        for i in pos_mask_contour[:, 0, :]:
            x1, y1 = i
            x2, y2 = c_x, c_y
            v = [x1, y1, x2, y2]
            angle = get_angle(v)
            dis = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if not angle in coordinate.keys():
                coordinate[angle] = [x1, y1, dis]
            elif dis > coordinate[angle][2]:
                coordinate[angle] = [x1, y1, dis]  #get longest dist
            else:
                continue
        #生成36个角度
        new_coordinate = {}
        for i in range(0, 360, 10):
            if i in coordinate.keys():
                angle = i
                x, y, dis = coordinate[i]
                new_coordinate[angle] = [x, y]
            elif i + 1 in coordinate.keys():
                angle = i
                x, y, dis = coordinate[i + 1]
                new_coordinate[angle] = [x, y]
            elif i - 1 in coordinate.keys():
                angle = i
                x, y, dis = coordinate[i - 1]
                new_coordinate[angle] = [x, y]
            elif i + 2 in coordinate.keys():
                angle = i
                x, y, dis = coordinate[i + 2]
                new_coordinate[angle] = [x, y]
            elif i - 2 in coordinate.keys():
                angle = i
                x, y, dis = coordinate[i - 2]
                new_coordinate[angle] = [x, y]
            elif i + 3 in coordinate.keys():
                angle = i
                x, y, dis = coordinate[i + 3]
                new_coordinate[angle] = [x, y]
            elif i - 3 in coordinate.keys():
                angle = i
                x, y, dis = coordinate[i - 3]
                new_coordinate[angle] = [x, y]
        #缺失的角度用临近的点插值
        for a in range(0, 360, 10):
            if not a in new_coordinate.keys():
                try:
                    x1, y1 = new_coordinate[a - 10]
                    x2, y2 = new_coordinate[a + 10]
                    new_coordinate[a] = [(x1 + x2) / 2, (y1 + y2) / 2]
                except:
                    #angle missing bug
                    new_coordinate[a] = [c_x, c_y]
        #计算x,y(fmap上的重心坐标) 和 36个点的距离
        distances = torch.zeros(36)
        for idx in range(36):
            target_x, target_y = new_coordinate[idx*10]
            dist = torch.sqrt((c_x-target_x)**2+(c_y-target_y)**2)
            #确保最小的距离>0
            if dist == 0:
                dist = 1e-6
            distances[idx] = dist

        return distances, new_coordinate



    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)

            if data is None:
                idx = self._rand_another(idx)
                continue
            return data








