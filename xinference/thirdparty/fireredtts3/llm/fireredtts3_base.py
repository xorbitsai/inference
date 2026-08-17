import os
import torch
import torchaudio
import torch.nn.functional as F
from transformers import (
    Qwen3Model, Qwen3Config,
    PretrainedConfig, PreTrainedModel
)
from fireredtts3.llm.patch_encoder import PatchEncoder, RotaryEmbedding
from fireredtts3.llm.dit import DiT
from fireredtts3.redae.redae import RedAE
from fireredtts3.campp.campp import CamppEmbedding
from fireredtts3.utils.utils import fix_seed
from fireredtts3.utils.text_tokenizer import (
    load_text_tokenizer, 
    MULTI_LANG_TAGS, MULTI_DIALECT_TAGS,
)


Qwen3_1_7B_ConfigDict = {
    "architectures": [
        "Qwen3ForCausalLM"
    ],
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "head_dim": 128,
    "hidden_act": "silu",
    "hidden_size": 2048,
    "initializer_range": 0.02,
    "intermediate_size": 6144,
    "max_position_embeddings": 40960,
    "max_window_layers": 28,
    "model_type": "qwen3",
    "num_attention_heads": 16,
    "num_hidden_layers": 28,
    "num_key_value_heads": 8,
    "rms_norm_eps": 1e-06,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "sliding_window": None,
    "tie_word_embeddings": True,
    "torch_dtype": "bfloat16",
    "transformers_version": "5.6.2",
    "use_cache": True,
    "use_sliding_window": False,
    "vocab_size": 151936,
    "attn_implementation": "flash_attention_2",
}


class FireRedTTS3BaseCoreConfig(PretrainedConfig):
    def __init__(
        self,
        redae_dim: int = 64,
        # Shared
        num_history_patches: int = 2,
        spk_in_dim: int = 512,
        # PatchEncoder
        patch_size: int = 4,
        patch_encoder_hidden_size: int = 1024,
        patch_encoder_mlp_ratio: int = 4,
        patch_encoder_depth: int = 8,
        patch_encoder_num_heads: int = 16,
        # DiT
        dit_mlp_ratio: int = 3,
        dit_depth: int = 11,
        dit_num_heads: int = 16,
        dit_hidden_size: int = 1024,
        # Other
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.redae_dim = redae_dim
        # Shared
        self.num_history_patches=num_history_patches
        self.spk_in_dim=spk_in_dim
        # PatchEncoder
        self.patch_size=patch_size
        self.patch_encoder_hidden_size=patch_encoder_hidden_size
        self.patch_encoder_mlp_ratio=patch_encoder_mlp_ratio
        self.patch_encoder_depth=patch_encoder_depth
        self.patch_encoder_num_heads=patch_encoder_num_heads
        # DiT
        self.dit_mlp_ratio=dit_mlp_ratio
        self.dit_depth=dit_depth
        self.dit_num_heads=dit_num_heads
        self.dit_hidden_size=dit_hidden_size


class FireRedTTS3BaseCore(PreTrainedModel):
    config_class = FireRedTTS3BaseCoreConfig
    base_model_prefix = "fireredtts3_base_core"

    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: FireRedTTS3BaseCoreConfig):
        super().__init__(config)
        # Backbone Transformer
        self.backbone_llm_config = Qwen3Config.from_dict(Qwen3_1_7B_ConfigDict)
        self.backbone_llm = Qwen3Model(self.backbone_llm_config)
        # Speaker proj
        self.spk_proj_llm = torch.nn.Linear(config.spk_in_dim, self.backbone_llm_config.hidden_size)
        self.spk_proj_dit = torch.nn.Linear(config.spk_in_dim, config.spk_in_dim)
        # PatchEncoder
        self.patch_encoder = PatchEncoder(
            in_dim=config.redae_dim,
            out_dim=self.backbone_llm_config.hidden_size,
            patch_size=config.patch_size,
            hidden_size=config.patch_encoder_hidden_size,
            mlp_ratio=config.patch_encoder_mlp_ratio,
            depth=config.patch_encoder_depth,
            num_heads=config.patch_encoder_num_heads,
        )
        # DiT
        self.dit_head = torch.nn.Linear(self.backbone_llm_config.hidden_size, config.dit_hidden_size)
        self.dit = DiT(
            in_channels=(config.redae_dim+config.spk_in_dim+config.dit_hidden_size),
            out_channels=config.redae_dim,
            mlp_ratio=config.dit_mlp_ratio,
            depth=config.dit_depth,
            num_heads=config.dit_num_heads,
            hidden_size=config.dit_hidden_size,
        )
        # Stop
        self.stop_head = torch.nn.Linear(self.backbone_llm_config.hidden_size, 1)
        # Shared
        self.redae_dim = config.redae_dim
        self.patch_size = self.patch_encoder.patch_size
        self.history_patches = config.num_history_patches
        self.history_length = config.num_history_patches * self.patch_size
        self.post_init()
    
    # Manually init RotaryEmbedding buffers
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, RotaryEmbedding):
            module.rope_init()
    
    # Backbone Transformer AR wrapper
    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    def _backbone_one_step(self, input_embeds: torch.Tensor, cache = None):
        outs = self.backbone_llm.forward(
            inputs_embeds=input_embeds,
            use_cache=True,
            past_key_values=cache,
        )
        # Only take the last timestep
        hidden_states = outs.last_hidden_state
        new_cache = outs.past_key_values
        return hidden_states, new_cache
    
    # Flow head wrapper
    def _flow_one_step(
        self, 
        hist_latents: torch.Tensor, 
        backbone_cond: torch.Tensor,
        spk_cond: torch.Tensor,
        t_span: torch.Tensor,
        inference_cfg: float,
    ):
        # Compose input
        x0 = torch.randn(1, self.patch_size, self.redae_dim, device=hist_latents.device)
        
        xt = torch.cat([hist_latents, x0], dim=1)   # History clean + current noise
        cond = torch.cat([
            backbone_cond.repeat_interleave(self.patch_size, dim=1),    # Correspond backbone cond
            spk_cond.unsqueeze(1).repeat(1, self.history_length+self.patch_size, 1) # Spk 
        ], dim=-1)
        # Run flow inference
        for ti, t in enumerate(t_span[:-1]):
            dt = t_span[ti+1]-t
            t_in = t.view(-1, 1, 1)
            x_in = torch.cat([xt, cond], dim=2)
            if inference_cfg > 0:
                x_in_cfg = torch.cat([xt, cond * 0], dim=2)
                x_in = torch.cat([x_in, x_in_cfg], dim=0)
                t_in = t_in.expand(2, -1, -1)
            vt = self.dit(x=x_in, t=t_in)
            if inference_cfg > 0:
                vt_cond, vt_cfg = vt.chunk(2, dim=0)
                vt = (1.0 + inference_cfg) * vt_cond - inference_cfg * vt_cfg
            # Only denoise current patch
            xt[:, -self.patch_size:] = xt[:, -self.patch_size:] + dt.view(-1, 1, 1) * vt[:, -self.patch_size:]
        # Remove history
        x1 = xt[:, -self.patch_size:]        
        return x1

    # Core LLM-DiT AR Loop
    @torch.no_grad()
    def generate(
        self, 
        # Input
        spk_emb: torch.Tensor,
        text_tokens: torch.Tensor,
        prompt_latents: torch.Tensor,
        # Inference settings
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
        stop_threshold: float = 0.5,
        # Length control
        min_gen_steps: int = 6,
        max_gen_steps: int = None,
    ):
        device = text_tokens.device

        # Compose input sequence
        input_embeds: torch.Tensor = self.backbone_llm.embed_tokens(text_tokens)
        patch_prompt_latents: torch.Tensor = self.patch_encoder(prompt_latents)
        spk_embs_llm = self.spk_proj_llm(spk_emb)
        input_embeds = torch.cat([spk_embs_llm.unsqueeze(1), input_embeds, patch_prompt_latents], dim=1)
        
        # Prepare DiT decode
        t_span = torch.linspace(0, 1, n_timesteps + 1).to(device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi) # (n_timesteps+1,)
        latents_gen = F.pad(prompt_latents, (0, 0, self.history_length, 0))
        
        # Prepare DiT conditions
        dit_spk_cond = self.spk_proj_dit(spk_emb) # (b=1, c)

        # Prepare Backbone states
        backbone_cond = input_embeds.new_zeros(1, self.history_patches, input_embeds.shape[-1])
        backbone_cache = None

        max_gen_steps = (400 if max_gen_steps is None else max_gen_steps)
        for step_index in range(max_gen_steps):
            # Backbone condition
            backbone_out, backbone_cache = self._backbone_one_step(input_embeds, cache=backbone_cache)   # (b=1, t, c)

            # Stop prediction
            stop_logits = self.stop_head(backbone_out[:, -1]).squeeze(-1)
            stop_score = torch.sigmoid(stop_logits).item()
            if stop_score >= stop_threshold:
                if min_gen_steps is not None:
                    if step_index >= min_gen_steps: break
                else:
                    break
            
            # DiT decode
            if step_index == 0:
                one_backbone_out = backbone_out[:, -patch_prompt_latents.shape[1]:]
            else:
                one_backbone_out = backbone_out[:, -1:]
            backbone_cond = torch.cat([backbone_cond, one_backbone_out], dim=1)

            one_latents = self._flow_one_step(
                hist_latents=latents_gen[:, -self.history_length:],
                backbone_cond=self.dit_head((backbone_cond[:, -(self.history_patches+1):])),
                spk_cond=dit_spk_cond,
                t_span=t_span,
                inference_cfg=inference_cfg,
            )
            
            input_embeds = self.patch_encoder(one_latents)
            latents_gen = torch.cat([latents_gen, one_latents], dim=1)
        
        # Remove dummy history
        latents_gen = latents_gen[:, self.history_length:]
        return latents_gen


# RedAE + TextTokenizer + TTS3Core
class FireRedTTS3Base(object):
    def __init__(self, pretrained_model_dir: str):
        self.device = torch.device('cuda')
        # RedAE
        redae_model_dir = os.path.join(pretrained_model_dir, 'redae')
        assert os.path.exists(redae_model_dir), f'{redae_model_dir} not found'
        self.redae = RedAE.from_pretrained(redae_model_dir)
        self.redae.to(self.device)
        # LLM-DiT 
        tts_model_dir = os.path.join(pretrained_model_dir, 'fireredtts3_base')
        assert os.path.exists(tts_model_dir), f'{tts_model_dir} not found'
        self.tts_core = FireRedTTS3BaseCore.from_pretrained(tts_model_dir)
        self.tts_core.to(self.device)
        # Text Tokenizer
        text_tok_dir = os.path.join(pretrained_model_dir, 'text_tokenizer')
        assert os.path.exists(text_tok_dir), f'{text_tok_dir} not found'
        self.text_tokenizer = load_text_tokenizer(text_tok_dir)
        # Speaker
        spk_ckpt_path = os.path.join(pretrained_model_dir, 'campp/campplus_voxceleb.bin')
        assert os.path.exists(spk_ckpt_path), f'{spk_ckpt_path} not found'
        self.spk_extractor = CamppEmbedding(spk_ckpt_path)
        self.spk_extractor.to(self.device)
    
    def _tokenize_text(self, text:str):
        tokens = self.text_tokenizer(
            text, 
            truncation=False, padding=False, add_special_tokens=False,
        )["input_ids"]
        tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return tokens

    @torch.inference_mode()
    def generate(
        self,
        # Input
        language: str,
        prompt_text: str,
        prompt_audio: torch.Tensor,
        prompt_audio_sr: int,
        text: str,
        # Inference
        stop_threshold: float = 0.5,
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
        seed: int = 1234,
    ):
        # ICL Text
        lang_tag = f'<|{language}|>'
        assert lang_tag in MULTI_LANG_TAGS+MULTI_DIALECT_TAGS, f'invalid language: {language}'
        input_text = f'{lang_tag}<|sot|>{prompt_text}{text}<|eot|>'
        text_tokens = self._tokenize_text(input_text)
        # Prompt audio
        prompt_audio = prompt_audio[:1]
        prompt_audio = torchaudio.functional.resample(prompt_audio, prompt_audio_sr, self.redae.sample_rate)
        prompt_audio_sr = self.redae.sample_rate
        prompt_audio = self.redae.pad_to_multiple_of(prompt_audio, self.redae.downsample_rate*self.tts_core.patch_size)
        prompt_audio = prompt_audio.to(self.device)
        prompt_latents = self.redae.encode(prompt_audio, prompt_audio_sr)
        prompt_latents = prompt_latents.to(torch.float32)
        # Spk emb
        spk_emb = self.spk_extractor.forward(prompt_audio, prompt_audio_sr)
        spk_emb = spk_emb.to(self.device)
        # TTS
        if seed is not None:
            fix_seed(seed)
        gen_latents = self.tts_core.generate(
            # Input
            spk_emb=spk_emb,
            text_tokens=text_tokens,
            prompt_latents=prompt_latents,
            # Inference settings
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            stop_threshold=stop_threshold,
            # Length control
            min_gen_steps=6,
            max_gen_steps=None,
        )
        gen_audio, gen_audio_sr = self.redae.decode(gen_latents)
        # Remove prompts
        gen_audio = gen_audio[:, prompt_audio.shape[1]:]
        return gen_audio, gen_audio_sr
