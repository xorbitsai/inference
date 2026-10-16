# Copyright (c) OpenMMLab. All rights reserved.
from annotator.mmpkg.mmcv.utils import Registry
from torch.nn.parallel import DataParallel, DistributedDataParallel

MODULE_WRAPPERS = Registry("module wrapper")
MODULE_WRAPPERS.register_module(module=DataParallel)
MODULE_WRAPPERS.register_module(module=DistributedDataParallel)
