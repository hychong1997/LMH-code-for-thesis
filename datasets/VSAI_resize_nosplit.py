import glob
import os
import os.path as osp
import sys
import copy
from .builder import ROTATED_DATASETS


import numpy as np

from .VSAI import VSAIDataset
from mmdet.datasets.pipelines import Compose

from .. import poly2obb_np


@ROTATED_DATASETS.register_module()
class VSAIDataset_resize_nosplit(VSAIDataset):
    def __init__(self,
                 pipeline,
                 multi_scales=[(1200, 1200), (400, 400)],
                 use_random_ms=False,
                 **kwargs):
        super(VSAIDataset_resize_nosplit, self).__init__(pipeline=pipeline, **kwargs)

        self.multi_scales = multi_scales
        self.use_random_ms = use_random_ms
        self.pipeline_cfg = pipeline

    def load_annotations(self, ann_folder):   # ann_folder=ann_file
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations and attributes txt files
        """
        ann_files_path = os.path.join(ann_folder, 'annfiles (copy)')
        distance_path = os.path.join(ann_folder, 'distance_nosplit')

        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_files_path + '/*.txt')


        data_infos = []
        if not ann_files:  # test phase
            # print('111111111111111111111111111111')
            ann_files = glob.glob(ann_folder + '/*.png')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_info['distance'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.png'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []
                distance = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        # print(bbox_info)
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                        label = cls_map[cls_name]  # 0
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)
                            gt_polygons.append(poly)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)
                    data_info['ann']['polygons'] = np.array(
                        gt_polygons, dtype=np.float32)
                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)
                    data_info['ann']['polygons'] = np.zeros((0, 8),
                                                            dtype=np.float32)

                if gt_polygons_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.array(
                        gt_polygons_ignore, dtype=np.float32)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)
                    data_info['ann']['polygons_ignore'] = np.zeros(
                        (0, 8), dtype=np.float32)

                simple_distance = os.path.join(distance_path, img_id + '.txt')

                with open(simple_distance) as dis:
                    line = dis.readline()
                    dis_info = line.split(' ')
                    d = dis_info[0]
                    distance.append(d)
                data_info['distance'] = np.array(distance, dtype=np.float64)

                data_infos.append(data_info)


        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos

    def get_distance(self, idx):

        return self.data_infos[idx]['distance']

    def prepare_train_img(self, idx):

        # print('111111111111111111111111111111')
        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        distance = self.get_distance(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        pipeline_cfg = copy.deepcopy(self.pipeline_cfg)
        for p in pipeline_cfg:
            if p.type == 'RResize':
                if (distance[0]) < 400.0:
                    p.img_scale = self.multi_scales[1]
                elif (distance[0]) > 1400.0:
                    p.img_scale = self.multi_scales[0]

        pipeline = Compose(pipeline_cfg)
        return pipeline(results)

    def prepare_test_img(self, idx):
        """Get testing data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Testing data after pipeline with new keys introduced by \
                pipeline.
        """

        img_info = self.data_infos[idx]
        distance = self.get_distance(idx)
        results = dict(img_info=img_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        pipeline_cfg = copy.deepcopy(self.pipeline_cfg)
        for p in pipeline_cfg:
            if p.type == 'MultiScaleFlipAug':
                for t in p.transforms:
                    if t.type == 'RResize':
                        if (distance[0]) < 400.0:
                            p.img_scale = self.multi_scales[1]
                        elif (distance[0]) > 1400.0:
                            p.img_scale = self.multi_scales[0]
        # for p in pipeline_cfg:
        #     if p.type == 'MultiScaleFlipAug':
        #         for t in p.transforms:
        #             if t.type == 'RResize':
        #                 print(distance[0], p.img_scale)

        pipeline = Compose(pipeline_cfg)
        return pipeline(results)
