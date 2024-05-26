# Copyright (c) OpenMMLab. All rights reserved.
from .VSAI_resize import VSAIDataset_resize
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset, DOTADatasetv15, DOTADatasetv20  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403
from .dior import DIORDataset
from .VSAI import VSAIDataset, VSAIOriginDataset, VSAIThreeClassDataset
from .VSAI_resizewithf import VSAIDataset_resizewithf
from .VSAI_resize_nosplit import VSAIDataset_resize_nosplit

__all__ = ['SARDataset', 'DOTADataset', 'DOTADatasetv15', 'DOTADatasetv20', 'build_dataset', 'HRSCDataset', 'DIORDataset', 'VSAIDataset', 'VSAIOriginDataset', 'VSAIThreeClassDataset', 'VSAIDataset_resize', 'VSAIDataset_resizewithf', 'VSAIDataset_resize_nosplit']
