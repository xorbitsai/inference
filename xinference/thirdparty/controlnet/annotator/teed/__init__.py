from __future__ import print_function

import os
import shutil

import cv2
import numpy as np
import torch
from einops import rearrange
from huggingface_hub import hf_hub_download

from .....device_utils import get_available_device
from ..annotator_path import models_path
from ..util import load_model, safe_step
from .ted import TED  # TEED architecture


class TEEDDetector:
    """https://github.com/xavysp/TEED"""

    model_dir = os.path.join(models_path, "TEED")

    def __init__(self, mteed: bool = False):
        self.device = get_available_device()
        self.model = TED().to(self.device).eval()

        if mteed:
            self.load_mteed_model()
        else:
            self.load_teed_model()

    def load_teed_model(self):
        """Load vanilla TEED model"""
        remote_url = os.environ.get(
            "CONTROLNET_TEED_MODEL_URL",
            "https://huggingface.co/bdsqlsz/qinglong_controlnet-lllite/resolve/main/Annotators/7_model.pth",
        )
        model_path = os.path.join(self.model_dir, "7_model.pth")
        if not os.path.exists(model_path):
            hf_hub_download("bdsqlsz/qinglong_controlnet-lllite", "Annotators/7_model.pth",
                            local_dir=self.model_dir)
            shutil.move(os.path.join(self.model_dir, "Annotators/7_model.pth"), model_path)
        self.model.load_state_dict(torch.load(model_path))

    def load_mteed_model(self):
        """Load MTEED model for Anyline"""
        model_path = os.path.join(self.model_dir, "MTEED.pth")
        if not os.path.exists(model_path):
            hf_hub_download("TheMistoAI/MistoLine", "Anyline/MTEED.pth",
                            local_dir=self.model_dir)
            shutil.move(os.path.join(self.model_dir, "Anyline/MTEED.pth"), model_path)
        self.model.load_state_dict(torch.load(model_path))

    def unload_model(self):
        if self.model is not None:
            self.model.cpu()

    def __call__(self, image: np.ndarray, safe_steps: int = 2) -> np.ndarray:
        self.model.to(self.device)

        H, W, _ = image.shape
        with torch.no_grad():
            image_teed = torch.from_numpy(image.copy()).float().to(self.device)
            image_teed = rearrange(image_teed, "h w c -> 1 c h w")
            edges = self.model(image_teed)
            edges = [e.detach().cpu().numpy().astype(np.float32)[0, 0] for e in edges]
            edges = [
                cv2.resize(e, (W, H), interpolation=cv2.INTER_LINEAR) for e in edges
            ]
            edges = np.stack(edges, axis=2)
            edge = 1 / (1 + np.exp(-np.mean(edges, axis=2).astype(np.float64)))
            if safe_steps != 0:
                edge = safe_step(edge, safe_steps)
            edge = (edge * 255.0).clip(0, 255).astype(np.uint8)
            return edge
