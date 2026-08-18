import math
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    Qwen3Config, Qwen3Model, 
    PretrainedConfig, PreTrainedModel
)

# --- Encoder
class Qwen3ClsDownsample(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        downsample_rate: int = 2,
        # Qwen
        hidden_size: int = 896,
        intermediate_size: int = 896*4,
        num_hidden_layers: int = 4,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.downsample_rate = downsample_rate
        self.qwen3_config = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            attn_implementation='flash_attention_2',
        )
        self.qwen3 = Qwen3Model(self.qwen3_config)
        self.cls_tok = torch.nn.Parameter(torch.ones(1, 1, hidden_size))
        # Input & output proj
        self.in_proj = (
            torch.nn.Linear(in_dim, hidden_size)
            if in_dim != hidden_size else 
            torch.nn.Identity()
        )
        self.out_proj = (
            torch.nn.Linear(hidden_size, out_dim)
            if out_dim != hidden_size else 
            torch.nn.Identity()
        )
    
    def forward(self, xs: torch.Tensor):
        """Downsample latents as BERT [CLS] token

        Args:
            xs(torch.Tensor): shape (b, t, c).
        Returns:
            ys(torch.Tensor): shape (b, t//downsample_rate, c).
        """
        assert xs.shape[1] % self.downsample_rate == 0, xs.shape
        b = xs.shape[0]
        # Patchify, (b, t, c) -> (b*t/down, down, c)
        xs = xs.reshape(-1, self.downsample_rate, self.in_dim)
        xs = self.in_proj(xs)
        cls_tok = self.cls_tok.expand(xs.shape[0], -1, -1)
        xs = torch.cat([xs, cls_tok], dim=1)  # (b*t/down, down+1, c)
        # Forward LLM
        ys = []
        for xs_chunk in torch.split(xs, 32768, dim=0):  # FlashAttn limits
            outs_chunk = self.qwen3(inputs_embeds=xs_chunk)
            ys_chunk = outs_chunk.last_hidden_state[:, -1]   # (b*t/down, c)
            ys.append(ys_chunk)
        ys = torch.cat(ys, dim=0)
        # ---
        ys = self.out_proj(ys)
        ys = ys.reshape(b, -1, self.out_dim)
        return ys


class RedAEAudioEncoder(torch.nn.Module):
    def __init__(
        self,
        # Output
        out_dim: int = 1024,
        # Input reshape
        audio_patch_size: int = 480,    # 50Hz
        audio_sample_rate: int = 24000,
        # Qwen
        hidden_size: int = 896,
        intermediate_size: int = 896*4,
        num_hidden_layers: int = 24,
        max_position_embeddings: int = 32768,
        max_window_layers: int = 0,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        sliding_window: int = 64,
        use_sliding_window: bool = True,
        # Extra downsample
        extra_downsample_rate: int = 2,   # 50Hz -> 25Hz
        downsample_num_hidden_layers: int = 4,
    ):
        super().__init__()
        self.audio_patch_size = audio_patch_size
        self.audio_sample_rate = audio_sample_rate
        self.extra_downsample_rate = extra_downsample_rate
        self.out_dim = out_dim
        self.in_proj = nn.Sequential(
            nn.Linear(self.audio_patch_size, hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.qwen3_config = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=max_window_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            use_sliding_window=use_sliding_window,
            attn_implementation='flash_attention_2',
        )
        self.qwen3 = Qwen3Model(self.qwen3_config)
        if self.extra_downsample_rate > 1:
            self.downsample = Qwen3ClsDownsample(
                in_dim=hidden_size,
                out_dim=hidden_size,
                downsample_rate=extra_downsample_rate,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_hidden_layers=downsample_num_hidden_layers,
                max_position_embeddings=max_position_embeddings,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
            )
        self.out_proj = nn.Linear(hidden_size, out_dim)

    # --- Some property
    @property
    def downsample_rate(self):
        return int(self.audio_patch_size * self.extra_downsample_rate)
    
    @property
    def sample_rate(self):
        return self.audio_sample_rate
    
    @property
    def hidden_size(self):
        return self.out_dim
    
    # --- Forward
    def forward(self, audio: torch.Tensor):
        """
        Args:
            audio(torch.Tensor): shape (b, t)
        Returns:
            xs(torch.Tensor): shape (b, t//self.downsample_rate, self.hidden_size)
        """ 
        assert audio.shape[1] % self.downsample_rate == 0, audio.shape
        # Patchify
        xs = audio.unfold(
            dimension=1,
            size=self.audio_patch_size,
            step=self.audio_patch_size,
        )   # (b, num_patch, patch_size) ~ (b, t, c)
        # LLM
        xs = self.in_proj(xs)
        outs = self.qwen3(
            inputs_embeds=xs,
            attention_mask=None,
        )
        xs = outs.last_hidden_state # (b, t, c)
        # Downsample
        if self.extra_downsample_rate > 1:
            xs = self.downsample(xs)
        xs = self.out_proj(xs)
        return xs


# --- Decoder
class ISTFT(nn.Module):
    def __init__(
        self, 
        n_fft: int, 
        hop_length: int, 
        win_length: int, 
        padding: str = "same"
    ):
        super().__init__()
        assert padding in ["center", "same"], "Padding must be 'center' or 'same'."
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if self.padding == "center":
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


class ISTFTHead(nn.Module):
    def __init__(
        self, 
        dim: int, 
        n_fft: int, 
        hop_length: int, 
        padding: str = "same"
    ):
        super().__init__()
        self.hop_length = hop_length
        out_dim = n_fft + 2
        self.out = torch.nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pred = self.out(x)
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2)  # safeguard to prevent excessively large magnitudes
        # wrapping happens here. These two lines produce real and imaginary value
        x = torch.cos(p)
        y = torch.sin(p)
        # recalculating phase here does not produce anything new
        # only costs time
        # phase = torch.atan2(y, x)
        # S = mag * torch.exp(phase * 1j)
        # better directly produce the complex value 
        S = mag * (x + 1j * y)
        audio = self.istft(S)
        return audio


class RedAEAudioDecoder(torch.nn.Module):
    def __init__(
        self,
        in_dim: int = 64,
        upsample_rate: int = 2, # 25Hz -> 50Hz
        # Output reshape
        audio_patch_size: int = 480,    # 50Hz
        audio_sample_rate: int = 24000,
        # Qwen(mirrors encoder)
        hidden_size: int = 896,
        intermediate_size: int = 896*4,
        num_hidden_layers: int = 18,
        max_position_embeddings: int = 32768,
        max_window_layers: int = 0,
        num_attention_heads: int = 14,
        num_key_value_heads: int = 2,
        sliding_window: int = 64,
        use_sliding_window: bool = True,
    ):
        super().__init__()
        self.upsample_rate = upsample_rate
        self.audio_patch_size = audio_patch_size
        self.audio_sample_rate = audio_sample_rate
        # Upsample MLP
        self.in_proj = nn.Linear(in_dim, upsample_rate * hidden_size)
        # Qwen3
        self.qwen3_config = Qwen3Config(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            max_position_embeddings=max_position_embeddings,
            max_window_layers=max_window_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            sliding_window=sliding_window,
            use_sliding_window=use_sliding_window,
        )
        self.qwen3 = Qwen3Model(self.qwen3_config)

        self.istft_head = ISTFTHead(
            dim=hidden_size, 
            n_fft=audio_patch_size*4,
            hop_length=audio_patch_size,
            padding='same',
        )

    def forward(
        self,
        xs: torch.Tensor,
    ):
        """
        Args:
            xs(torch.Tensor): shape (b, t, c)
        Returns:
            audio(torch.Tensor): shape (b, t*upsample_rate*patch_size)
            audio_len(torch.Tensor): shape (b,)
        """
        # Upsample
        xs = self.in_proj(xs)   # (b, t, upsample_rate, c)
        xs = xs.reshape(xs.shape[0], -1, self.qwen3_config.hidden_size) # (b, t*upsample_rate, c)
        # Qwen3
        outs = self.qwen3(
            inputs_embeds=xs,
            attention_mask=None,
        )
        xs = outs.last_hidden_state # (b, t*upsample_rate, c)
        audio = self.istft_head(xs)
        return audio


# --- RedAE
class RedAEConfig(PretrainedConfig):
    model_type = "redae"

    def __init__(
        self,
        # Shared
        audio_patch_size: int = 480,
        audio_sample_rate: int = 24000,
        # Encoder
        enc_hidden_size: int = 896,
        enc_intermediate_size: int = 3584,
        enc_num_hidden_layers: int = 18,
        enc_max_position_embeddings: int = 32768,
        enc_max_window_layers: int = 0,
        enc_num_attention_heads: int = 14,
        enc_num_key_value_heads: int = 2,
        enc_sliding_window: int = 64,
        enc_use_sliding_window: bool = True,
        enc_extra_downsample_rate: int = 2,   # 50Hz -> 25Hz
        enc_downsample_num_hidden_layers: int = 4,
        # Intermediate
        bottleneck_dim: int = 64,
        # Decoder
        dec_hidden_size: int = 896,
        dec_intermediate_size: int = 3584,
        dec_num_hidden_layers: int = 18,
        dec_max_position_embeddings: int = 32768,
        dec_max_window_layers: int = 0,
        dec_num_attention_heads: int = 14,
        dec_num_key_value_heads: int = 2,
        dec_sliding_window: int = 64,
        dec_use_sliding_window: bool = True,
        # Other
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Shared
        self.audio_patch_size=audio_patch_size
        self.audio_sample_rate=audio_sample_rate
        # Encoder
        self.enc_hidden_size=enc_hidden_size
        self.enc_intermediate_size=enc_intermediate_size
        self.enc_num_hidden_layers=enc_num_hidden_layers
        self.enc_max_position_embeddings=enc_max_position_embeddings
        self.enc_max_window_layers=enc_max_window_layers
        self.enc_num_attention_heads=enc_num_attention_heads
        self.enc_num_key_value_heads=enc_num_key_value_heads
        self.enc_sliding_window=enc_sliding_window
        self.enc_use_sliding_window=enc_use_sliding_window
        self.enc_extra_downsample_rate=enc_extra_downsample_rate
        self.enc_downsample_num_hidden_layers=enc_downsample_num_hidden_layers
        # Intermediate
        self.bottleneck_dim=bottleneck_dim
        # Decoder
        self.dec_hidden_size=dec_hidden_size
        self.dec_intermediate_size=dec_intermediate_size
        self.dec_num_hidden_layers=dec_num_hidden_layers
        self.dec_max_position_embeddings=dec_max_position_embeddings
        self.dec_max_window_layers=dec_max_window_layers
        self.dec_num_attention_heads=dec_num_attention_heads
        self.dec_num_key_value_heads=dec_num_key_value_heads
        self.dec_sliding_window=dec_sliding_window
        self.dec_use_sliding_window=dec_use_sliding_window

        
class RedAE(PreTrainedModel):
    config_class = RedAEConfig
    base_model_prefix = "redae"

    _supports_flash_attn = True
    _supports_sdpa = True

    def __init__(self, config: RedAEConfig):
        super().__init__(config)

        self.encoder = RedAEAudioEncoder(
            out_dim=config.bottleneck_dim,
            # Input reshape
            audio_patch_size=config.audio_patch_size,    # 50Hz
            audio_sample_rate=config.audio_sample_rate,
            # Qwen
            hidden_size=config.enc_hidden_size,
            intermediate_size=config.enc_intermediate_size,
            num_hidden_layers=config.enc_num_hidden_layers,
            max_position_embeddings=config.enc_max_position_embeddings,
            max_window_layers=config.enc_max_window_layers,
            num_attention_heads=config.enc_num_attention_heads,
            num_key_value_heads=config.enc_num_key_value_heads,
            sliding_window=config.enc_sliding_window,
            use_sliding_window=config.enc_use_sliding_window,
            # Extra downsample
            extra_downsample_rate=config.enc_extra_downsample_rate,   # 50Hz -> 25Hz
            downsample_num_hidden_layers=config.enc_downsample_num_hidden_layers,
        )
        self.decoder = RedAEAudioDecoder(
            in_dim=config.bottleneck_dim,
            upsample_rate=config.enc_extra_downsample_rate,
            audio_patch_size=config.audio_patch_size,
            audio_sample_rate=config.audio_sample_rate,
            hidden_size=config.dec_hidden_size,
            intermediate_size=config.dec_intermediate_size,
            num_hidden_layers=config.dec_num_hidden_layers,
            max_position_embeddings=config.dec_max_position_embeddings,
            max_window_layers=config.dec_max_window_layers,
            num_attention_heads=config.dec_num_attention_heads,
            num_key_value_heads=config.dec_num_key_value_heads,
            sliding_window=config.dec_sliding_window,
            use_sliding_window=config.dec_use_sliding_window,
        )
        self.post_init()
    
    # --- Shared property
    @property
    def downsample_rate(self):
        return self.encoder.downsample_rate
    
    @property
    def sample_rate(self):
        return self.decoder.audio_sample_rate
    
    @property
    def hidden_size(self):
        return self.encoder.out_dim
    
    @staticmethod
    def pad_to_multiple_of(audio: torch.Tensor, multiple_of: int):
        target_samples = math.ceil(audio.shape[-1] / multiple_of) * multiple_of
        pad_len = target_samples - audio.shape[-1]
        if pad_len > 0:
            audio = F.pad(audio, (pad_len, 0))  # NOTE left pad
        return audio

    @torch.autocast(device_type='cuda', dtype=torch.bfloat16)
    @torch.no_grad()
    def encode(self, audio: torch.Tensor, audio_sr:int):
        """
        Args:
            audio: shape (b, t)
            audio_sr: int
        Returns:
            latents: shape (b, l=t//960, c=64)
        """
        audio = audio[:1]
        audio = torchaudio.functional.resample(audio, audio_sr, self.sample_rate)
        audio = self.pad_to_multiple_of(audio, self.downsample_rate)
        latents = self.encoder.forward(audio)
        return latents
    
    @torch.no_grad()
    def decode(self, latents: torch.Tensor):
        """
        Args:
            latents: shape (b, l=t//960, c=64)
        Returns:
            audio: shape (b, t)
            audio_sr: int
        """
        audio = self.decoder.forward(latents)
        return audio, self.sample_rate
