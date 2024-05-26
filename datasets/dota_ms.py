import copy
from .dota import DOTADataset
from .builder import ROTATED_DATASETS
import numpy as np
from mmdet.datasets.pipelines import Compose

@ROTATED_DATASETS.register_module()
class DOTADatasetMS(DOTADataset):
    def __init__(self,
                 pipeline,
                 multi_scales=[[(1536, 1536)], [(512, 512)]],
                 use_random_ms=False,
                 **kwargs):
        super(DOTADatasetMS, self).__init__(pipeline=pipeline, **kwargs)

        self.multi_scales = multi_scales
        self.use_random_ms = use_random_ms
        self.pipeline_cfg = pipeline

    def prepare_train_ms_img(self, idx, branch_idx):

        assert branch_idx < len(self.multi_scales)

        img_info = self.data_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]
        self.pre_pipeline(results)
        pipeline_cfg = self.pipeline_cfg.copy()
        for p in pipeline_cfg:
            if p.type == 'RResize':
                if self.use_random_ms:
                    p.img_scale = self.multi_scales[branch_idx][
                        np.random.randint(0, len(self.multi_scales[branch_idx]))]
                else:
                    p.img_scale = self.multi_scales[branch_idx][0]
        pipeline = Compose(pipeline_cfg)
        return pipeline(results)
    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set \
                True).
        """

        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                print('data is None, continue')
                continue
            else:
                data['data_ms'] = []
                # get ms data after pipeline
                for i in range(len(self.multi_scales)):
                    while True:
                        data_ms_i = self.prepare_train_ms_img(idx, i)
                        if data_ms_i is not None:
                            data['data_ms'].append(data_ms_i)
                            break
                        else:
                            print('data_ms is None, continue: ', 'img_idx ', idx, 'scale idx ',  i)
                return data
