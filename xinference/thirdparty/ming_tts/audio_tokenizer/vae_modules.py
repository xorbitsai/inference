import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Model, Qwen2Config
import torch

from .istft import ISTFTHead


class StreamingLinearUpsample(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor
        self.upsampler = nn.Upsample(scale_factor=scale_factor, mode='linear', align_corners=False)

    def forward(self, x, state=None, is_last=False):
        # 初始化状态
        if state is None:
            state = {
                'prev_chunk': None,
                'history_last': None,
                'is_first': True
            }

        if x is None and not is_last:
            return None, state

        if state['is_first'] and is_last:
            out = self.upsampler(x.transpose(1, 2)).transpose(1, 2)
            return out, None # 结束后清除状态

        output_chunks = []

        if state['is_first']:
            state['prev_chunk'] = x
            state['is_first'] = False
            if not is_last:
                return None, state

        if state['prev_chunk'] is not None:
            p = state['prev_chunk'].transpose(1, 2)

            if state['history_last'] is None:
                lookahead = x[:, :1, :].transpose(1, 2)
                inp = torch.cat([p, lookahead], dim=2)
                up = self.upsampler(inp)
                out_prev = up[:, :, :p.size(2) * self.scale_factor]
            else:
                lookahead = x[:, :1, :].transpose(1, 2)
                inp = torch.cat([state['history_last'], p, lookahead], dim=2)
                up = self.upsampler(inp)
                start = self.scale_factor
                end = start + p.size(2) * self.scale_factor
                out_prev = up[:, :, start:end]

            output_chunks.append(out_prev.transpose(1, 2))
            state['history_last'] = p[:, :, -1:]
            state['prev_chunk'] = x

        if is_last:
            p = state['prev_chunk'].transpose(1, 2)
            inp = torch.cat([state['history_last'], p], dim=2)
            up = self.upsampler(inp)
            out_last = up[:, :, self.scale_factor:]
            output_chunks.append(out_last.transpose(1, 2))
            state = None # 结束

        final_out = torch.cat(output_chunks, dim=1) if output_chunks else None
        return final_out, state


class Encoder(nn.Module):
    def __init__(self, encoder_args, input_dim=320, hop_size=320, latent_dim=64, patch_size=-1):
        super().__init__()
        config = Qwen2Config.from_dict(config_dict=encoder_args)
        self.encoder = Qwen2Model(config)
        self.input_dim = input_dim
        self.hop_size = hop_size
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(input_dim, config.hidden_size, bias=False)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, latent_dim*2)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.patch_size = patch_size
        if patch_size != -1:
            config.num_hidden_layers = 4
            self.aggregator = Qwen2Model(config)
            self.cls_embed = nn.Parameter(torch.rand(1, 1, config.hidden_size))
            self.cls_embed.data.normal_(0, 0.02)

    def get_frames(self, x):
        num_frames_total = (x.size(-1) + self.hop_size - 1) // self.hop_size  # 向上取整的帧数
        expected_len = (num_frames_total - 1) * self.hop_size + self.input_dim
        padding_needed = expected_len - x.size(-1)
        waveform = F.pad(x, (0, padding_needed), value=0.0)

        frames = waveform.unfold(dimension=-1, size=self.input_dim, step=self.hop_size)   # [B, T, d]
        return frames

    def pad_patch_insert_cls(self, x):
        bsz, _, dim = x.size()
        num_frame = x.size(1)
        r = num_frame % self.patch_size
        pad_num = self.patch_size-r if r else 0
        x = F.pad(x, (0, 0, 0, pad_num), value=0.0)  # 帧数对齐到patch_size倍数
        x = x.reshape(-1, self.patch_size, dim)
        x = torch.cat((x, self.cls_embed.expand(x.size(0), -1, -1)), dim=1)  # 每个patch后插入一个cls
        x = x.reshape(bsz, -1, dim)
        return x

    def forward(self, waveform):
        x = self.get_frames(waveform)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.encoder(inputs_embeds=x)
        x = x.last_hidden_state

        # downsample
        if self.patch_size != -1:
            x = self.pad_patch_insert_cls(x)
            x = self.aggregator(inputs_embeds=x)
            x = x.last_hidden_state
            bsz, _, dim = x.size()
            x = x.reshape(-1, self.patch_size+1, dim)
            x = x[:, -1:, :].reshape(bsz, -1, dim)

        x = self.fc3(x)
        return x, waveform.unsqueeze(1)


class Decoder(nn.Module):
    def __init__(self, decoder_args, output_dim=320, latent_dim=64, semantic_model=None, patch_size=-1):
        super().__init__()
        config = Qwen2Config.from_dict(config_dict=decoder_args)
        self.decoder = Qwen2Model(config)
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(latent_dim, config.hidden_size)

        if semantic_model is not None:
            self.gelu = nn.GELU()
            self.fc2 = nn.Linear(config.hidden_size, semantic_model.audio_emb_dim)
            self.semantic_model = semantic_model
            self.fc3 = nn.Linear(semantic_model.audio_emb_dim, config.hidden_size)
        else:
            self.semantic_model = None

        self.hop_length = output_dim
        self.head = ISTFTHead(dim=config.hidden_size, n_fft=self.hop_length * 4, hop_length=self.hop_length, padding="same")
        self.patch_size = patch_size
        if self.patch_size != -1:
            self.upsampling = StreamingLinearUpsample(scale_factor=patch_size)


    def forward(self, x, only_semantic_emb=False, past_key_values=None, use_cache=False):
        x = self.fc1(x)

        if self.semantic_model is not None:
            x = self.fc2(self.gelu(x))
            x, past_key_values = self.semantic_model(whipser_feats=x, past_key_values=past_key_values, use_cache=use_cache)
            unified_emb = x
            if only_semantic_emb:
                return unified_emb, past_key_values
            x = self.fc3(x)
        else:
            unified_emb = None

        if self.patch_size != -1:
            x = self.upsampling(x.transpose(1, 2)).transpose(1, 2)

        x = self.decoder(inputs_embeds=x)
        x = x.last_hidden_state

        x, _ = self.head(x)

        return x, unified_emb

    def low_level_reconstruct(self, x, past_key_values=None, use_cache=False, stream_state=None, last_chunk=False):
        upsample_state, audio_buffer, window_buffer = stream_state
        bsz, device, dtype = x.size(0), x.device, x.dtype
        x = self.fc1(x)
        if self.patch_size != -1:
            if use_cache:
                # streaming
                x, upsample_state = self.upsampling(x, state=upsample_state, is_last=last_chunk)
                if x is None:
                    stream_state = (upsample_state, audio_buffer, window_buffer)
                    return torch.empty(bsz, 1, 0, device=device, dtype=dtype), stream_state, past_key_values
            else:
                x = self.upsampling.upsampler(x.transpose(1, 2)).transpose(1, 2)

        outputs = self.decoder(inputs_embeds=x, past_key_values=past_key_values, use_cache=use_cache)
        past_key_values = outputs.past_key_values
        x = outputs.last_hidden_state

        x, _, audio_buffer, window_buffer = self.head(x, streaming=use_cache, audio_buffer=audio_buffer, window_buffer=window_buffer, last_chunk=last_chunk)

        stream_state = (upsample_state, audio_buffer, window_buffer)
        return x, stream_state, past_key_values
