import torch
from diffusers import AutoencoderKL
import json
import os
from diffusers import FlowMatchEulerDiscreteScheduler
from .transformer import DiffusionTransformer
from .pipeline import ImageGenerationPipeline
import torch.nn as nn
import torch.nn.functional as F
from .autoencoder_kl_qwenimage import AutoencoderKLQwenImage
from ..inference_profile import InferenceProfile

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToClipMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        #self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(input_dim, 2048)
        self.layer_norm1 = nn.LayerNorm(2048)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(2048, output_dim)
        self.layer_norm2 = nn.LayerNorm(output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.relu(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.layer_norm2(hidden_states)
        return hidden_states

class ConditionedTransformer(nn.Module):
    def __init__(self, transformer, vision_dim=1152, use_identity_mlp=False, text_encoder_norm=False):
        super().__init__()
        self.transformer = transformer
        self.mlp = ToClipMLP(vision_dim, 2560) if not use_identity_mlp else nn.Identity()
        self.mlp.to(dtype=self.dtype)
        # self.mlp_pool = ToClipMLP(vision_dim, 768)
        self.config = self.transformer.config
        self.in_channels = self.transformer.in_channels
        self.text_encoder_norm = text_encoder_norm

        # must be used together
        #if text_encoder_norm or use_identity_mlp:
        #    assert use_identity_mlp and text_encoder_norm


    @property
    def dtype(self):
        return next(self.transformer.parameters()).dtype

    def forward(self, hidden_states,
                    timestep,
                    encoder_hidden_states,
                    return_dict,
                    encoder_attention_mask=None,
                    extra_vit_input=None,
                    ref_hidden_states=None,
                    encoder_hidden_states_2=None,
                     **kargs):

        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, list):
                encoder_hidden_states = torch.stack(encoder_hidden_states, dim=0)

            if self.text_encoder_norm:
                encoder_hidden_states = F.normalize(encoder_hidden_states, dim=-1) * 1000.0  # 1000 matches the original text encoder norm

            encoder_hidden_states = self.mlp(encoder_hidden_states)

            if extra_vit_input is not None:
                encoder_hidden_states = torch.cat((encoder_hidden_states, extra_vit_input), dim=1)

            encoder_hidden_states = list(encoder_hidden_states.unbind(dim=0))

        hidden_states = self.transformer(
                    x=hidden_states,
                    cap_feats=encoder_hidden_states,
                    t=timestep,
                    return_dict=False,
                    ref_x=ref_hidden_states,
                    cap_feats_2=encoder_hidden_states_2,
                     **kargs
                )
        return hidden_states

    def enable_gradient_checkpointing(self):
        self.transformer.enable_gradient_checkpointing()

def latent_mean_variance(latents: torch.Tensor, per_channel: bool = True, unbiased_var: bool = False):
    """Compute the mean and variance of latents (shape follows the diffusion
    training ``latents``, usually ``[B, C, H, W]``).

    Args:
        latents: ``[B, C, H, W]``; a ``[B, C, T, H, W]`` tensor with
            ``T==1`` is ``squeeze(2)`` first.
        per_channel: when True aggregate over ``(B, H, W)``, one scalar per
            channel, shape ``[C]``; when False use one global mean/variance.
        unbiased_var: use the unbiased estimator (Bessel).

    Returns:
        ``(mean, variance)``; the std is ``variance.sqrt()`` (use
        ``torch.sqrt(variance.clamp_min(0))`` for numerical stability).
    """
    z = latents
    if z.dim() == 5 and z.shape[2] == 1:
        z = z.squeeze(2)
    if z.dim() != 4:
        raise ValueError(f"Expected 4D latents [B,C,H,W] (or 5D with T=1), got {tuple(z.shape)}")
    if per_channel:
        dims = (0, 2, 3)
        mean_v = z.mean(dim=dims)
        var_v = z.var(dim=dims, unbiased=unbiased_var)
    else:
        mean_v = z.mean()
        var_v = z.var(unbiased=unbiased_var)
    return mean_v, var_v

class ImageGenerator(torch.nn.Module):
    def __init__(self,
            model_path,
            vision_dim=2560,
            scheduler_path=None,
            mlp_state_dict=None,
            torch_dtype=torch.float32,
            device='cpu',
            use_identity_mlp=False,
            text_encoder_norm=False,
            inference_profile=None,
        ):
        super(ImageGenerator, self).__init__()

        if not isinstance(inference_profile, InferenceProfile):
            raise ValueError(
                "inference_profile must be derived from the checkpoint "
                "capability contract (load_checkpoint_capabilities)"
            )
        self.inference_profile = inference_profile

        if device is not None:
            device = torch.device(device)
        else:
            device = torch.device(torch.cuda.current_device())

        self.scheduler_path = scheduler_path

        vae_config_path = os.path.join(model_path, "vae", "config.json")
        assert os.path.exists(vae_config_path)

        with open(vae_config_path, "r") as f:
            vae_config =  json.load(f)
            if "_class_name" in vae_config and vae_config["_class_name"] == "AutoencoderKLQwenImage":
                self.vae = AutoencoderKLQwenImage.from_pretrained(
                    model_path,
                    subfolder="vae",
                    torch_dtype=torch_dtype,
                )
                self.vae_sample_mode = "argmax"
            else:
                self.vae = AutoencoderKL.from_pretrained(
                    model_path,
                    subfolder="vae",
                    torch_dtype=torch_dtype,
                )
                self.vae_sample_mode = "sample"

        self.vae.input_channels = 4 if ('input_channels' in self.vae.config and self.vae.config.input_channels == 4) or ('in_channels' in self.vae.config and self.vae.config.in_channels == 4) else 3
        if self.vae.input_channels != self.inference_profile.vae_input_channels:
            raise ValueError(
                "VAE input channels do not match the checkpoint capability "
                f"contract: checkpoint={self.vae.input_channels}, "
                f"capability={self.inference_profile.vae_input_channels}"
            )
        if self.vae_sample_mode != self.inference_profile.vae_sample_mode:
            raise ValueError(
                "VAE sample mode does not match the checkpoint capability "
                f"contract: checkpoint={self.vae_sample_mode}, "
                f"capability={self.inference_profile.vae_sample_mode}"
            )

        # self.vae.to(self.torch_type).to(self.device)
        self.vae.requires_grad_(False)

        self.train_model = DiffusionTransformer.from_pretrained(
            model_path, subfolder="transformer",
            torch_dtype=torch_dtype,
            alignment_padding_mode=self.inference_profile.alignment_padding_mode,
            multi_frame_output=self.inference_profile.multi_frame_output,
        )
        if (
            self.train_model.alignment_padding_mode
            != self.inference_profile.alignment_padding_mode
            or self.train_model.multi_frame_output
            != self.inference_profile.multi_frame_output
        ):
            raise ValueError(
                "instantiated Transformer capability does not match the "
                "checkpoint capability contract: "
                f"transformer=({self.train_model.alignment_padding_mode!r}, "
                f"{self.train_model.multi_frame_output!r}), "
                f"capability=({self.inference_profile.alignment_padding_mode!r}, "
                f"{self.inference_profile.multi_frame_output!r})"
            )

        self.train_model = ConditionedTransformer(self.train_model, vision_dim=vision_dim, use_identity_mlp=use_identity_mlp, text_encoder_norm=text_encoder_norm)

        assert mlp_state_dict is not None
        self.train_model.mlp.load_state_dict(mlp_state_dict, strict=True)

        self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(self.scheduler_path, subfolder="scheduler")
        self.noise_scheduler.config['use_dynamic_shifting'] = True

        self.pipelines = ImageGenerationPipeline(
            vae=self.vae,
            transformer=self.train_model,
            text_encoder=None,
            tokenizer=None,
            scheduler=self.noise_scheduler,
        ).to(device)

    @property
    def device(self):
        return next(self.train_model.parameters()).device

    def set_trainable_params(self, trainable_params):

        self.vae.requires_grad_(False)

        if trainable_params == 'all':
            self.train_model.requires_grad_(True)
        else:
            self.train_model.requires_grad_(False)
            for name, module in self.train_model.named_modules():
                for trainable_param in trainable_params:
                    if trainable_param in name:
                        for params in module.parameters():
                            params.requires_grad = True

        num_parameters_trainable = 0
        num_parameters = 0
        name_parameters_trainable = []
        for n, p in self.train_model.named_parameters():
            num_parameters += p.data.nelement()
            if not p.requires_grad:
                continue  # frozen weights
            name_parameters_trainable.append(n)
            num_parameters_trainable += p.data.nelement()
        logger.info(f"number of all Diffusion parameters: {num_parameters}, trainable: {num_parameters_trainable}")


    def sample(self, encoder_hidden_states, steps=None, cfg=None, cfg_mode=1, seed=42, height=512, width=512, use_dynamic_shifting=False, extra_vit_input=None, ref_x=None, directvlm_hidden_states=None, num_frames_per_prompt=1):
        sampling = self.inference_profile.resolve_sampling_parameters(
            steps=steps,
            cfg=cfg,
        )
        steps = sampling.steps
        cfg = sampling.cfg
        negative_prompt_embeds = None
        if encoder_hidden_states is not None:
            encoder_hidden_states = encoder_hidden_states.to(
                device=self.device, dtype=self.train_model.dtype
            )
            encoder_hidden_states = list(encoder_hidden_states.unbind(dim=0))
            negative_prompt_embeds= [en * 0 for en in encoder_hidden_states]

        encoder_hidden_states_2 = directvlm_hidden_states
        negative_prompt_embeds_2 = None
        if encoder_hidden_states_2 is not None:
            encoder_hidden_states_2 = encoder_hidden_states_2.to(
                device=self.device, dtype=self.train_model.dtype
            )
            encoder_hidden_states_2 = list(encoder_hidden_states_2.unbind(dim=0))
            negative_prompt_embeds_2= [en * 0 for en in encoder_hidden_states_2]

        image = self.pipelines(
            prompt_embeds=encoder_hidden_states,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_2=encoder_hidden_states_2,
            negative_prompt_embeds_2=negative_prompt_embeds_2,
            guidance_scale=cfg,
            #guidance_scale_mode=cfg_mode,
            generator=torch.manual_seed(seed),
            num_inference_steps=steps,
            height=height,
            width=width,
            max_sequence_length=512,
            device=self.device,
            #extra_vit_input=extra_vit_input,
            ref_hidden_states=ref_x,
            #use_dynamic_shifting=use_dynamic_shifting,
            sample_mode=self.vae_sample_mode,
            num_frames_per_prompt=num_frames_per_prompt,
        ).images

        return image
