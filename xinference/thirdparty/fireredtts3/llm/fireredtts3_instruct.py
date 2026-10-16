import os
import torch
import torchaudio
import torch.nn.functional as F
from transformers import (
    Qwen3ForCausalLM, Qwen3Config,
    PretrainedConfig, PreTrainedModel
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from fireredtts3.llm.patch_encoder import PatchEncoder, RotaryEmbedding
from fireredtts3.llm.dit import DiT
from fireredtts3.redae.redae import RedAE
from fireredtts3.utils.utils import fix_seed
from fireredtts3.utils.text_tokenizer import load_text_tokenizer
from fireredtts3.llm.fireredtts3_base import Qwen3_1_7B_ConfigDict
from fireredtts3.utils.chatml import (
    CHATML_LATENT_IN_PAD_ID,
    CHATML_LATENT_OUT_PAD_ID,
    compose_generate_input_tts,
    compose_generate_input_voice_design,
    compose_generate_input_semantic_edit,
    compose_generate_input_acoustic_edit,
)


TEXT_EOT_ID: int = 151677
AUDIO_SOS_ID: int = 151669
REDAE_SCALE = 0.4


def init_text_logits_processor(
    repetition_penalty: float = None,
    do_sample: bool = True,
    temperature: float = None,
    top_p: float = None,
    top_k: int = None,
):
    text_logits_processor = LogitsProcessorList()
    if repetition_penalty is not None and repetition_penalty != 1.0:
        text_logits_processor.append(RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty))
    if do_sample:
        if temperature is not None and temperature != 1.0:
            text_logits_processor.append(TemperatureLogitsWarper(temperature=float(temperature)))
        if top_k is not None and top_k > 0:
            text_logits_processor.append(TopKLogitsWarper(top_k=top_k))
        if top_p is not None and top_p < 1.0:
            text_logits_processor.append(TopPLogitsWarper(top_p=top_p))
    return text_logits_processor


class FireRedTTS3InstructCoreConfig(PretrainedConfig):
    def __init__(
        self,
        redae_dim: int = 64,
        # Shared
        num_history_patches: int = 2,
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


class FireRedTTS3InstructCore(PreTrainedModel):
    config_class = FireRedTTS3InstructCoreConfig
    base_model_prefix = "fireredtts3_instruct_core"

    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: FireRedTTS3InstructCoreConfig):
        super().__init__(config)
        # Backbone Transformer
        self.backbone_llm_config = Qwen3Config.from_dict(Qwen3_1_7B_ConfigDict)
        self.backbone_llm = Qwen3ForCausalLM(self.backbone_llm_config)
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
            in_channels=(config.redae_dim+config.dit_hidden_size),
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
        outs = self.backbone_llm.model.forward(
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
        t_span: torch.Tensor,
        inference_cfg: float,
    ):
        # Compose input
        x0 = torch.randn(1, self.patch_size, self.redae_dim, device=hist_latents.device)
        
        xt = torch.cat([hist_latents, x0], dim=1)   # History clean + current noise
        cond = backbone_cond.repeat_interleave(self.patch_size, dim=1)    # Correspond backbone cond
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
        text_tokens: torch.Tensor,
        latents_in: torch.Tensor = None,
        latents_in_mask: torch.Tensor = None,   # For filling text_tokens
        # Output (ICL)
        latents_out: torch.Tensor = None,
        latents_out_mask: torch.Tensor = None,  # For filling text_tokens
        # Text inference settings
        infer_text: bool = False,
        text_repetition_penalty: float = None,
        text_do_sample: bool = True,
        text_temperature: float = None,
        text_top_p: float = None,
        text_top_k: int = None,
        # Audio inference settings
        n_timesteps: int = 10,
        inference_cfg: float = 2.0,
        stop_threshold: float = 0.5,
        # Audio length control
        min_gen_steps: int = 6,
        max_gen_steps: int = None,
    ):
        device = text_tokens.device

        # Compose input sequence
        input_embeds: torch.Tensor = self.backbone_llm.model.embed_tokens(text_tokens)
        # Any input audio
        if latents_in is not None:
            latents_patch_in: torch.Tensor = self.patch_encoder(latents_in)
            input_embeds = input_embeds.masked_scatter(
                latents_in_mask.unsqueeze(-1),
                latents_patch_in.reshape(-1).to(input_embeds),
            )
        # Any output audio (ICL prompt)
        if latents_out is not None:
            latents_patch_out = self.patch_encoder.forward(latents_out)
            input_embeds = input_embeds.masked_scatter(
                latents_out_mask.unsqueeze(-1),
                latents_patch_out.reshape(-1).to(input_embeds),
            )
        else:
            latents_out = torch.zeros(1, 0, self.config.redae_dim, device=device)
            latents_patch_out = None
        
        # Prepare DiT decode
        t_span = torch.linspace(0, 1, n_timesteps + 1).to(device)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi) # (n_timesteps+1,)

        # Init Backbone states
        backbone_cache = None

        # --- Infer text 
        if infer_text:
            # Text logits processor
            text_logits_processor = init_text_logits_processor(
                text_repetition_penalty, text_do_sample,
                text_temperature, text_top_p, text_top_k,
            )
            text_gen_ids = torch.empty((1, 0), dtype=torch.long, device=device)
            for text_step_index in range(200):
                backbone_out, backbone_cache = self._backbone_one_step(input_embeds, cache=backbone_cache)   # (b=1, t, c)
                # Sampling
                logits = self.backbone_llm.lm_head(backbone_out[:, -1, :])  # (1, V)
                scores = text_logits_processor(text_gen_ids, logits)
                if text_do_sample:
                    probs = torch.softmax(scores.float(), dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)[:, 0]  # (1,)
                else:
                    next_token = scores.argmax(dim=-1)                          # (1,)
                # Next step
                input_embeds = self.backbone_llm.model.embed_tokens(next_token.unsqueeze(0))
                # Whether stop
                if next_token.item() == TEXT_EOT_ID:
                    break
                text_gen_ids = torch.cat([text_gen_ids, next_token.unsqueeze(0)], dim=1)
            # Finalize text inference
            _, backbone_cache = self._backbone_one_step(input_embeds, cache=backbone_cache)   # (b=1, t, c)
            # Process <|sosp|> for audio start
            next_token = next_token * 0 + AUDIO_SOS_ID
            input_embeds = self.backbone_llm.model.embed_tokens(next_token.unsqueeze(0))
            _, backbone_cache = self._backbone_one_step(input_embeds, cache=backbone_cache)   # (b=1, t, c)
        
        # --- Infer audio
        latents_gen =  F.pad(latents_out, (0, 0, self.history_length, 0))
        backbone_cond = input_embeds.new_zeros(1, self.history_patches, input_embeds.shape[-1])

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
            if step_index == 0 and latents_patch_out is not None:
                one_backbone_out = backbone_out[:, -latents_patch_out.shape[1]:]
            else:
                one_backbone_out = backbone_out[:, -1:]
            backbone_cond = torch.cat([backbone_cond, one_backbone_out], dim=1)

            one_latents = self._flow_one_step(
                hist_latents=latents_gen[:, -self.history_length:],
                backbone_cond=self.dit_head((backbone_cond[:, -(self.history_patches+1):])),
                t_span=t_span,
                inference_cfg=inference_cfg,
            )
            
            input_embeds = self.patch_encoder(one_latents)
            latents_gen = torch.cat([latents_gen, one_latents], dim=1)
        
        # Remove dummy history
        latents_gen = latents_gen[:, self.history_length:]

        if infer_text:
            return latents_gen, text_gen_ids
        else:
            return latents_gen


# RedAE + TextTokenizer + TTS3Core
class FireRedTTS3Instruct(object):
    def __init__(self, pretrained_model_dir: str):
        self.device = torch.device('cuda')
        # RedAE
        redae_model_dir = os.path.join(pretrained_model_dir, 'redae')
        assert os.path.exists(redae_model_dir), f'{redae_model_dir} not found'
        self.redae = RedAE.from_pretrained(redae_model_dir)
        self.redae.to(self.device)
        # LLM-DiT 
        tts_model_dir = os.path.join(pretrained_model_dir, 'fireredtts3_instruct')
        assert os.path.exists(tts_model_dir), f'{tts_model_dir} not found'
        self.tts_core = FireRedTTS3InstructCore.from_pretrained(tts_model_dir)
        self.tts_core.to(self.device)
        # Text Tokenizer
        text_tok_dir = os.path.join(pretrained_model_dir, 'text_tokenizer')
        assert os.path.exists(text_tok_dir), f'{text_tok_dir} not found'
        self.text_tokenizer = load_text_tokenizer(text_tok_dir)
    
    def _tokenize_text(self, text:str):
        tokens = self.text_tokenizer(
            text, 
            truncation=False, padding=False, add_special_tokens=False,
        )["input_ids"]
        tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
        return tokens

    def _tokenize_audio(self, audio: torch.Tensor, audio_sr: int):
        audio = audio[:1]
        audio = torchaudio.functional.resample(audio, audio_sr, self.redae.sample_rate)
        audio_sr = self.redae.sample_rate
        audio = self.redae.pad_to_multiple_of(audio, self.redae.downsample_rate*self.tts_core.patch_size)
        audio = audio.to(self.device)
        latents = self.redae.encode(audio, audio_sr) * REDAE_SCALE
        latents = latents.to(torch.float32)
        return latents

    # --- Inference Interface
    @torch.inference_mode()
    def generate_tts(
        self,
        # Input
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
        prompt_latents = self._tokenize_audio(prompt_audio, prompt_audio_sr)
        text_in = compose_generate_input_tts(prompt_latents.shape[1]//self.tts_core.patch_size, prompt_text, text)
        text_tokens = self._tokenize_text(text_in)
        # AR generate
        if seed is not None:
            fix_seed(seed)
        gen_latents = self.tts_core.generate(
            # Input
            text_tokens=text_tokens,
            latents_out=prompt_latents,
            latents_out_mask=(text_tokens==CHATML_LATENT_OUT_PAD_ID),
            # Text inference settings
            infer_text=False,
            # Audio inference settings
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            stop_threshold=0.5,
            # Audio length control
            min_gen_steps=6,
            max_gen_steps=None,
        )
        gen_audio, gen_audio_sr = self.redae.decode(gen_latents / REDAE_SCALE)
        # Remove prompts
        gen_audio = gen_audio[:, (self.redae.downsample_rate*prompt_latents.shape[1]):]
        return gen_audio, gen_audio_sr

    def generate_voice_design(
        self,
        # Input
        instruction: str,
        text: str,
        # Audio Inference Settings
        n_timesteps: int = 10,
        inference_cfg: float = 1.2,
        # Random
        seed: int = 2,
    ):
        text_in = compose_generate_input_voice_design(instruction, text)
        text_tokens = self._tokenize_text(text_in)

        # AR generate
        if seed is not None:
            fix_seed(seed)
        gen_latents, gen_text_ids = self.tts_core.generate(
            # Input
            text_tokens=text_tokens,
            # Text inference settings
            infer_text=True,
            text_repetition_penalty=1.0,
            text_do_sample=True,
            text_temperature=0.7,
            text_top_p=0.8,
            text_top_k=20,
            # Audio inference settings
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            stop_threshold=0.5,
            # Audio length control
            min_gen_steps=6,
            max_gen_steps=None,
        )
        gen_audio, gen_audio_sr = self.redae.decode(gen_latents / REDAE_SCALE)
        gen_text = self.text_tokenizer.decode(gen_text_ids.squeeze(0).cpu())
        
        return gen_audio, gen_audio_sr, gen_text

    def generate_semantic_edit(
        self,
        # Input
        instruction: str,
        audio_in: torch.Tensor,
        audio_in_sr: torch.Tensor,
        # Audio Inference Settings
        n_timesteps: int = 10,
        inference_cfg: float = 1.2,
        # Random
        seed: int = 1234,
    ):
        latents_in = self._tokenize_audio(audio_in, audio_in_sr)
        text_in = compose_generate_input_semantic_edit(instruction, latents_in.shape[1]//self.tts_core.patch_size)
        text_tokens = self._tokenize_text(text_in)
        # AR generate
        if seed is not None:
            fix_seed(seed)
        gen_latents, gen_text_ids = self.tts_core.generate(
            # Input
            text_tokens=text_tokens,
            latents_in=latents_in,
            latents_in_mask=(text_tokens==CHATML_LATENT_IN_PAD_ID),
            # Text inference settings
            infer_text=True,
            text_repetition_penalty=1.0,
            text_do_sample=False,
            # Audio inference settings
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            stop_threshold=0.5,
            # Audio length control
            min_gen_steps=6,
            max_gen_steps=None,
        )
        gen_audio, gen_audio_sr = self.redae.decode(gen_latents / REDAE_SCALE)
        gen_text = self.text_tokenizer.decode(gen_text_ids.squeeze(0).cpu())
        
        return gen_audio, gen_audio_sr, gen_text

    def generate_acoustic_edit(
        self,
        # Input
        instruction: str,
        audio_in: torch.Tensor,
        audio_in_sr: torch.Tensor,
        # Audio Inference Settings
        n_timesteps: int = 10,
        inference_cfg: float = 1.2,
        # Random
        seed: int = 1234,
    ):
        latents_in = self._tokenize_audio(audio_in, audio_in_sr)
        text_in = compose_generate_input_acoustic_edit(instruction, latents_in.shape[1]//self.tts_core.patch_size)
        text_tokens = self._tokenize_text(text_in)
        # AR generate
        if seed is not None:
            fix_seed(seed)
        gen_latents = self.tts_core.generate(
            # Input
            text_tokens=text_tokens,
            latents_in=latents_in,
            latents_in_mask=(text_tokens==CHATML_LATENT_IN_PAD_ID),
            # Text inference settings
            infer_text=False,
            # Audio inference settings
            n_timesteps=n_timesteps,
            inference_cfg=inference_cfg,
            stop_threshold=0.5,
            # Audio length control
            min_gen_steps=6,
            max_gen_steps=None,
        )
        gen_audio, gen_audio_sr = self.redae.decode(gen_latents / REDAE_SCALE)
        return gen_audio, gen_audio_sr
