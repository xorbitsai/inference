from numpy import zeros, int32, float32
from torch import from_numpy

from .core import maximum_path_jit


def maximum_path(neg_cent, mask):
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = neg_cent.data.cpu().numpy().astype(float32)
    path = zeros(neg_cent.shape, dtype=int32)

    t_t_max = mask.sum(1)[:, 0].data.cpu().numpy().astype(int32)
    t_s_max = mask.sum(2)[:, 0].data.cpu().numpy().astype(int32)
    maximum_path_jit(path, neg_cent, t_t_max, t_s_max)
    return from_numpy(path).to(device=device, dtype=dtype)
