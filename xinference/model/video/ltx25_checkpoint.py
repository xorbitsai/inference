# Copyright 2022-2026 Xinference Holdings Pte. Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional, Tuple, cast

from ..utils import ModelArtifactSource

if TYPE_CHECKING:
    import torch
    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors


LTX25_MODEL_SOURCES = {
    "huggingface": (
        "Lightricks/LTX-2.5-Diffusers",
        "a6de4b5354f078db24d9cf4778c14846788aea3d",
    ),
    "modelscope": ("Lightricks/LTX-2.5-Diffusers", "master"),
}

_LTX25_CHECKPOINT_PREFIX = "model.diffusion_model."
_CONNECTOR_PREFIXES = {"audio_embeddings_connector", "video_embeddings_connector"}
_TEXT_PROJECTION_KEYS = {
    "audio_text_proj_in.bias",
    "audio_text_proj_in.weight",
    "video_text_proj_in.bias",
    "video_text_proj_in.weight",
}
# Match the LTX-2.5 mappings in Diffusers scripts/convert_ltx2_to_diffusers.py.
_DIFFUSERS_PARAMETER_NAMES = {
    "adaln_single": "time_embed",
    "audio_adaln_single": "audio_time_embed",
    "audio_embeddings_connector": "audio_connector",
    "audio_patchify_proj": "audio_proj_in",
    "audio_prompt_adaln_single": "audio_prompt_adaln",
    "av_ca_a2v_gate_adaln_single": "av_cross_attn_video_a2v_gate",
    "av_ca_audio_scale_shift_adaln_single": "av_cross_attn_audio_scale_shift",
    "av_ca_v2a_gate_adaln_single": "av_cross_attn_audio_v2a_gate",
    "av_ca_video_scale_shift_adaln_single": "av_cross_attn_video_scale_shift",
    "k_norm": "norm_k",
    "patchify_proj": "proj_in",
    "prompt_adaln_single": "prompt_adaln",
    "q_norm": "norm_q",
    "scale_shift_table_a2v_ca_audio": "audio_a2v_cross_attn_scale_shift_table",
    "scale_shift_table_a2v_ca_video": "video_a2v_cross_attn_scale_shift_table",
    "transformer_1d_blocks": "transformer_blocks",
    "video_embeddings_connector": "video_connector",
}


def load_ltx25_transformer_and_connectors(
    checkpoint_path: str, base_model_path: str, dtype: "torch.dtype"
) -> Tuple["LTX2VideoTransformer3DModel", "LTX2TextConnectors"]:
    """Restore the transformer and connectors from a Phai-M1 BF16 checkpoint."""
    import torch
    from accelerate import init_empty_weights
    from diffusers import LTX2VideoTransformer3DModel
    from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors
    from safetensors.torch import load_file

    transformer_state: dict[str, torch.Tensor] = {}
    connector_state: dict[str, torch.Tensor] = {}
    for key, tensor in load_file(checkpoint_path, device="cpu").items():
        if not key.startswith(_LTX25_CHECKPOINT_PREFIX) or tensor.dtype not in (
            torch.bfloat16,
            torch.float32,
        ):
            raise ValueError(f"Unsupported Phai-M1 BF16 checkpoint tensor: {key}")
        parts = key[len(_LTX25_CHECKPOINT_PREFIX) :].split(".")
        converted_key = ".".join(
            _DIFFUSERS_PARAMETER_NAMES.get(part, part) for part in parts
        )
        state = (
            connector_state if parts[0] in _CONNECTOR_PREFIXES else transformer_state
        )
        state[converted_key] = tensor

    config = LTX2VideoTransformer3DModel.load_config(
        base_model_path, subfolder="transformer"
    )
    with init_empty_weights():
        transformer = LTX2VideoTransformer3DModel.from_config(config)
    transformer.load_state_dict(transformer_state, strict=True, assign=True)

    # The four text projection parameters come from the shared Gemma 4 weights.
    connectors = LTX2TextConnectors.from_pretrained(
        base_model_path, subfolder="connectors", dtype=dtype
    )
    missing, unexpected = connectors.load_state_dict(
        connector_state, strict=False, assign=True
    )
    if set(missing) != _TEXT_PROJECTION_KEYS or unexpected:
        raise ValueError(
            f"Incompatible Phai-M1 connector weights. "
            f"Missing parameters: {missing}. Unexpected parameters: {unexpected}."
        )
    # Cast both components after loading, as in the upstream converter.
    return transformer.to(dtype=dtype).eval(), connectors.to(dtype=dtype).eval()


def _download_ltx25_base(model_hub: str) -> str:
    model_id, revision = LTX25_MODEL_SOURCES[model_hub]
    source = ModelArtifactSource(cast(Literal["huggingface", "modelscope"], model_hub))
    return source.snapshot_download(
        model_id,
        revision=revision,
        allow_patterns=[
            "model_index.json",
            "transformer/config.json",
            "connectors/config.json",
            "connectors/diffusion_pytorch_model.safetensors.index.json",
            "connectors/diffusion_pytorch_model-*.safetensors",
            "text_encoder/*",
            "tokenizer/*",
            "scheduler/*",
            "vae/*",
            "audio_vae/*",
            "vocoder/*",
            "duration_head/*",
        ],
    )


def load_phai_m1_components(
    model_path: str,
    transformer_filename: str,
    base_model_path: Optional[str],
    model_hub: str,
    dtype: "torch.dtype",
) -> Tuple[str, "LTX2VideoTransformer3DModel", "LTX2TextConnectors"]:
    checkpoint = Path(model_path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / transformer_filename
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Phai-M1 checkpoint not found: {checkpoint}")

    if base_model_path is None:
        base_model_path = _download_ltx25_base(model_hub)
    if not (Path(base_model_path) / "model_index.json").is_file():
        raise FileNotFoundError(
            f"LTX-2.5 Diffusers model_index.json not found in {base_model_path}"
        )
    transformer, connectors = load_ltx25_transformer_and_connectors(
        str(checkpoint), base_model_path, dtype
    )
    return base_model_path, transformer, connectors
