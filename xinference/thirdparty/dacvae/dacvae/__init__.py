# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved\n

__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

import audiotools

audiotools.ml.BaseModel.INTERN += ["dacvae.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn
from . import model
from .model import DACVAE
