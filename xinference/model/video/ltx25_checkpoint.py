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

LTX25_BASE_FILES = (
    "model_index.json",
    "transformer/config.json",
    "scheduler/scheduler_config.json",
    "connectors/config.json",
    "connectors/diffusion_pytorch_model.safetensors.index.json",
    "connectors/diffusion_pytorch_model-00001-of-00002.safetensors",
    "connectors/diffusion_pytorch_model-00002-of-00002.safetensors",
    "text_encoder/config.json",
    "text_encoder/generation_config.json",
    "text_encoder/model.safetensors.index.json",
    *(f"text_encoder/model-{part:05d}-of-00005.safetensors" for part in range(1, 6)),
    "tokenizer/chat_template.jinja",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "audio_vae/config.json",
    "audio_vae/diffusion_pytorch_model.safetensors",
    "vocoder/config.json",
    "vocoder/diffusion_pytorch_model.safetensors",
    "duration_head/config.json",
    "duration_head/diffusion_pytorch_model.safetensors",
)

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
        allow_patterns=list(LTX25_BASE_FILES),
    )


def find_phai_m1_checkpoint(model_path: str, filename: str) -> Optional[Path]:
    path = Path(model_path)
    if path.is_file():
        return path
    for directory in (path, path / "diffusion_models"):
        checkpoint = directory / filename
        if checkpoint.is_file():
            return checkpoint
    return None


def load_phai_m1_components(
    model_path: str,
    transformer_filename: str,
    base_model_path: str,
    dtype: "torch.dtype",
) -> Tuple[str, "LTX2VideoTransformer3DModel", "LTX2TextConnectors"]:
    checkpoint = find_phai_m1_checkpoint(model_path, transformer_filename)
    if checkpoint is None:
        raise FileNotFoundError(f"Phai-M1 checkpoint not found in {model_path}")

    if not (Path(base_model_path) / "model_index.json").is_file():
        raise FileNotFoundError(
            f"LTX-2.5 Diffusers model_index.json not found in {base_model_path}"
        )
    transformer, connectors = load_ltx25_transformer_and_connectors(
        str(checkpoint), base_model_path, dtype
    )
    return base_model_path, transformer, connectors
