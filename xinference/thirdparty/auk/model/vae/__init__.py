import logging
import os
import sys

import torch

from auk.model.vae.bigvgan_flow_vae import BigVGANFlowVAE, BigVGANFlowVAEConfig


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__file__)


def load_ckpt(model, model_path, map_location="cpu"):
    if model_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        state_dict = load_file(model_path, device=map_location)
    else:
        state_dict = torch.load(model_path, map_location=map_location)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning(f"  [WARN] Missing keys: {missing}")
    if unexpected:
        logging.warning(f"  [WARN] Unexpected keys: {unexpected}")
    return model


def load_vae_model(vae_name, vae_cfg, vae_ckpt, **kwargs):
    if vae_name == "BigVGANFlowVAE":
        model = BigVGANFlowVAE(vae_cfg)
        assert vae_ckpt is not None, "BigVGANFlowVAE requires a checkpoint path"
        model = load_ckpt(model, vae_ckpt, **kwargs)
        model = model.eval()
        return model
    else:
        raise ValueError(f"Unknown VAE name: {vae_name}, Only support BigVGANFlowVAE")


__all__ = ["BigVGANFlowVAE", "BigVGANFlowVAEConfig", "load_vae_model"]
