
from .dota import DOTADataset

from .builder import ROTATED_DATASETS
import glob
import os
import os.path as osp

import numpy as np

from mmrotate.core import poly2obb_np

@ROTATED_DATASETS.register_module()
class DIORDataset(DOTADataset):
    """DIOR dataset for detection.

    Args:
        ann_file (str): Annotation file path.
        pipeline (list[dict]): Processing pipeline.
        version (str, optional): Angle representations. Defaults to 'oc'.
        difficulty (bool, optional): The difficulty threshold of GT.
    """
    CLASSES = ('airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
             'chimney', 'expressway-service-area', 'expressway-toll-station',
             'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
             'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
             'windmill'
                )
    # PALETTE = [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
    #            (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
    #            (255, 193, 193), (0, 51, 153), (255, 250, 205), (0, 139, 139),
    #            (255, 255, 0), (147, 116, 116), (0, 0, 255), (0, 120, 255),
    #            (255, 200, 0), (0, 50, 50), (0, 0, 128), (42, 42, 255)]

    PALETTE = [(165, 42, 42), (255, 0, 0), (255, 255, 0), (189, 183, 107),
               (255, 128, 0), (153, 51, 0), (255, 0, 255), (0, 255, 255),
               (255, 193, 193), (226, 43, 138), (255, 250, 205), (0, 139, 139),
               (0, 255, 0), (147, 116, 116), (0, 0, 255), (0, 120, 255),
               (255, 200, 0), (0, 50, 50), (0, 0, 128), (42, 42, 255)]

    def __init__(self,
                 ann_file,
                 pipeline,
                 version='oc',
                 difficulty=100,
                 **kwargs):
        self.version = version
        self.difficulty = difficulty

        super(DIORDataset, self).__init__(ann_file, pipeline, **kwargs)

    def load_annotations(self, ann_folder):
        """
            Args:
                ann_folder: folder that contains DOTA v1 annotations txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*.jpg')
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + '.jpg'
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_polygons = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []
                gt_polygons_ignore = []

                if os.path.getsize(ann_file) == 0 and self.filter_empty_gt:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        poly = np.array(bbox_info[:8], dtype=np.float32)
                        try:
                            x, y, w, h, a = poly2obb_np(poly, self.version)
                        except:  # noqa: E722
                            continue
                        cls_name = bbox_info[8]
                        difficulty = int(bbox_info[9])
                        label = cls_map[cls_name]
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

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos