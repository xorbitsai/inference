import os
import shutil

import cv2
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download

from .....device_utils import get_available_device
from ..annotator_path import models_path
from .models.mbv2_mlsd_large import MobileV2_MLSD_Large
from .models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
from .utils import pred_lines

mlsdmodel = None
remote_model_path = "https://huggingface.co/lllyasviel/ControlNet/resolve/main/annotator/ckpts/mlsd_large_512_fp32.pth"
old_modeldir = os.path.dirname(os.path.realpath(__file__))
modeldir = os.path.join(models_path, "mlsd")


def unload_mlsd_model():
    global mlsdmodel
    if mlsdmodel is not None:
        mlsdmodel = mlsdmodel.cpu()


def apply_mlsd(input_image, thr_v, thr_d):
    global modelpath, mlsdmodel
    if mlsdmodel is None:
        modelpath = os.path.join(modeldir, "mlsd_large_512_fp32.pth")
        old_modelpath = os.path.join(old_modeldir, "mlsd_large_512_fp32.pth")
        if os.path.exists(old_modelpath):
            modelpath = old_modelpath
        elif not os.path.exists(modelpath):
            dirname = "annotator/ckpts/"
            filename = "mlsd_large_512_fp32.pth"
            hf_hub_download("lllyasviel/ControlNet",
                            dirname + filename,
                            local_dir=modeldir)
            shutil.move(os.path.join(modeldir, dirname, filename),
                        os.path.join(modeldir, filename))
        mlsdmodel = MobileV2_MLSD_Large()
        mlsdmodel.load_state_dict(torch.load(modelpath), strict=True)
    mlsdmodel = mlsdmodel.to(get_available_device()).eval()

    model = mlsdmodel
    assert input_image.ndim == 3
    img = input_image
    img_output = np.zeros_like(img)
    try:
        with torch.no_grad():
            lines = pred_lines(img, model, [img.shape[0], img.shape[1]], thr_v, thr_d)
            for line in lines:
                x_start, y_start, x_end, y_end = [int(val) for val in line]
                cv2.line(
                    img_output, (x_start, y_start), (x_end, y_end), [255, 255, 255], 1
                )
    except Exception as e:
        pass
    return img_output[:, :, 0]
