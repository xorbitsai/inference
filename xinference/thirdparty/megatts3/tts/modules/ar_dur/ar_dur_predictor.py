# Copyright 2025 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from tqdm import tqdm

from tts.modules.ar_dur.commons.layers import Embedding, LayerNorm
from tts.modules.ar_dur.commons.nar_tts_modules import PosEmb
from tts.modules.ar_dur.commons.rot_transformer import RotTransformerDecoderLayer
from tts.modules.ar_dur.commons.transformer import SinusoidalPositionalEmbedding
from tts.modules.ar_dur.commons.rel_transformer import RelTransformerEncoder

FS_ENCODERS = {
    'rel_fft': lambda hp, dict_size: RelTransformerEncoder(
        dict_size, hp['hidden_size'], hp['hidden_size'],
        hp['ffn_hidden_size'], hp['num_heads'], hp['enc_layers'],
        hp['enc_ffn_kernel_size'], hp['dropout'], prenet=hp['enc_prenet'], pre_ln=hp['enc_pre_ln']),
}

def fill_with_neg_inf2(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(-1e8).type_as(t)

def expand_states(h, mel2token):
    h = F.pad(h, [0, 0, 1, 0])
    mel2token_ = mel2token[..., None].repeat([1, 1, h.shape[-1]])
    h = torch.gather(h, 1, mel2token_)  # [B, T, H]
    return h


class CodePredictor(nn.Module):
    def __init__(self, hparams, hidden_size, dec_hidden_size, lm_num_layers, dict_size, code_size):
        super().__init__()
        self.hparams = deepcopy(hparams)
        self.hparams['hidden_size'] = hidden_size
        self.hidden_size = hidden_size
        char_dict_size = hparams.get('char_dict_size', 4000)
        if not hparams.get('lm_use_enc'):
            self.encoder = nn.Embedding(dict_size, self.hidden_size, padding_idx=0)
            if hparams.get('mega_use_char', True):
                self.char_encoder = nn.Embedding(char_dict_size,
                                                 self.hidden_size, padding_idx=0)
        else:
            self.encoder = FS_ENCODERS[self.hparams['encoder_type']](self.hparams, dict_size)
            if hparams.get('mega_use_char', True):
                self.char_encoder = FS_ENCODERS[self.hparams['encoder_type']](self.hparams, char_dict_size)
            if hparams['use_ph_pos_embed']:
                self.ph_pos_embed = PosEmb(self.hidden_size)

        self.char_empty_embed = nn.Embedding(1, self.hidden_size)
        if hparams.get('use_bert_input'):
            self.bert_input_proj = nn.Linear(768, self.hidden_size)
        self.ling_label_embed_layers = nn.ModuleDict()
        for k, s in zip(hparams['ling_labels'], hparams['ling_label_dict_size']):
            self.ling_label_embed_layers[k] = Embedding(s + 3, self.hidden_size, padding_idx=0)

        self.dec_hidden_size = dec_hidden_size
        self.enc_proj = nn.Linear(self.hidden_size, dec_hidden_size)
        self.code_emb = Embedding(code_size + 2, dec_hidden_size, 0)
        self.use_pos_embed = hparams.get('use_pos_embed', False)
        if self.use_pos_embed:
            self.embed_positions = SinusoidalPositionalEmbedding(dec_hidden_size, 0, init_size=1024)
        self.use_post_ln = hparams.get('use_post_ln', False)
        self.layers = None
        if not self.use_post_ln:
            self.layer_norm = LayerNorm(dec_hidden_size)
        self.code_size = code_size
        self.project_out_dim = Linear(dec_hidden_size, code_size + 1, bias=True)

    def forward_ling_encoder(
            self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed, spk_id, spk_embed, mels_timbre):
        ph_tokens = txt_tokens
        hparams = self.hparams
        ph_nonpadding = (ph_tokens > 0).float()[:, :, None]  # [B, T_phone, 1]
        x_spk = self.forward_style_embed(spk_embed, spk_id, mels_timbre)

        # enc_ph
        if not hparams.get('lm_use_enc'):
            x_ph = self.encoder(ph_tokens)
            x_ph = x_ph + sum(
                [self.ling_label_embed_layers[k](ling_feas[k]) for k in hparams['ling_labels']]) \
                if len(hparams['ling_labels']) > 0 else 0
            x_ph = x_ph + x_spk
        else:
            # enc_ph
            ph_enc_oembed = sum(
                [self.ling_label_embed_layers[k](ling_feas[k]) for k in hparams['ling_labels']]) \
                if len(hparams['ling_labels']) > 0 else 0
            ph_enc_oembed = ph_enc_oembed + self.ph_pos_embed(
                torch.arange(0, ph_tokens.shape[1])[None,].to(ph_tokens.device))
            ph_enc_oembed = ph_enc_oembed + x_spk
            ph_enc_oembed = ph_enc_oembed * ph_nonpadding
            x_ph = self.encoder(ph_tokens, other_embeds=ph_enc_oembed)

        # enc_char
        if char_tokens is not None and ph2char is not None:
            char_nonpadding = (char_tokens > 0).float()[:, :, None]
            x_char = self.char_encoder(char_tokens)
            empty_char = (ph2char > 100000).long()
            ph2char = ph2char * (1 - empty_char)
            x_char_phlevel = \
                expand_states(x_char * char_nonpadding, ph2char) \
                * (1 - empty_char)[..., None] + \
                self.char_empty_embed(torch.zeros_like(ph_tokens)) * empty_char[..., None]
        else:
            x_char_phlevel = 0
        # x_ling
        x_ling = x_ph + x_char_phlevel
        x_ling = x_ling * ph_nonpadding
        x_ling = self.enc_proj(x_ling)
        return x_ling

    def sample_one_step(self, vq_pred):
        hparams = self.hparams
        if hparams.get('infer_top_k'):
            top_k = hparams.get('infer_top_k')
            temperature = hparams.get('infer_temperature', 1)
            vq_pred = vq_pred[:, -1] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(vq_pred, min(top_k, vq_pred.size(-1)))
                vq_pred[vq_pred < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(vq_pred, dim=-1)
            # sample from the distribution
            vq_pred = torch.multinomial(probs, num_samples=1)
        else:
            vq_pred = torch.argmax(F.softmax(vq_pred[:, -1], dim=-1), 1)
        return vq_pred

    def forward_style_embed(self, spk_embed=None, spk_id=None, mel_ref=None):
        # add spk embed
        style_embed = 0
        if self.hparams['use_spk_embed']:
            style_embed = style_embed + self.spk_embed_proj(spk_embed)[:, None, :]
        if self.hparams['use_spk_id']:
            style_embed = style_embed + self.spk_id_proj(spk_id)[:, None, :]
        if self.hparams['use_spk_enc']:
            style_embed = style_embed + self.spk_enc(mel_ref)[:, None, :]
        return style_embed

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
                not hasattr(self, '_future_mask')
                or self._future_mask is None
                or self._future_mask.device != tensor.device
                or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(fill_with_neg_inf2(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]


class ARDurPredictor(CodePredictor):
    def __init__(self, hparams, hidden_size, dec_hidden_size, lm_num_layers, dict_size, code_size, use_rot_embed=True,
                 op_version=1):
        super().__init__(hparams, hidden_size, dec_hidden_size, lm_num_layers, dict_size, code_size)
        self.use_rot_embed = use_rot_embed
        bias = hparams.get('lm_bias', True)
        if self.use_rot_embed:
            self.layers = nn.ModuleList([])
            self.layers.extend([
                RotTransformerDecoderLayer(
                    dec_hidden_size, 0.0, kernel_size=1, ffn_hidden_size=dec_hidden_size * 4,
                    post_ln=self.use_post_ln, op_version=op_version, bias=bias)
                for _ in range(lm_num_layers)
            ])
        if hparams['dur_model_type'] == 'ar_mse':
            self.project_out_dim = nn.Sequential(torch.nn.Linear(dec_hidden_size, 1), nn.Softplus())
        else:
            self.project_out_dim = torch.nn.Linear(dec_hidden_size, code_size + 1)

    def forward(self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
                prev_code, spk_id=None, spk_embed=None, mels_timbre=None, mel2ph=None,
                incremental_state=None, x_ling=None, attn_mask=None, spk_pos_ids_flat=None,
                prompt_length=None, cache_size=20, streaming=False):
        x = self.code_emb(prev_code)
        if x_ling is None:
            x_ling = self.forward_ling_encoder(
                txt_tokens, ling_feas, char_tokens, ph2char, bert_embed, spk_id, spk_embed, mels_timbre)
            x_ling = x_ling.flatten(0, 1)
            txt_tokens = txt_tokens.flatten(0, 1)
            x_ling = x_ling[txt_tokens > 0][None]

        # run decoder
        self_attn_padding_mask = None
        if self.use_pos_embed:
            positions = self.embed_positions(
                prev_code,
                incremental_state=incremental_state
            )
        if incremental_state is not None:
            x_ling = x_ling[:, x.shape[1] - 1:x.shape[1]]
            if spk_pos_ids_flat is not None:
                spk_pos_ids_flat = spk_pos_ids_flat[:, x.shape[1] - 1:x.shape[1]]
            x = x[:, -1:]
            if self.use_pos_embed:
                positions = positions[:, -1:]
            if streaming:
                # Shift Pos: query pos is min(cache_size, idx)
                spk_pos_ids_flat = torch.min(torch.LongTensor([prompt_length + cache_size]).to(x.device),
                                             spk_pos_ids_flat)

        # # B x T x C -> T x B x C
        if self.use_pos_embed:
            x = x + positions
        x_ling = x_ling[:, :self.hparams['max_tokens']].contiguous()
        T = min(self.hparams.get('max_tokens_per_item', 1e9), x_ling.shape[1])
        x_ling = x_ling.reshape(-1, T, x_ling.shape[-1])
        x = x + x_ling
        x = x.transpose(0, 1)

        for idx, layer in enumerate(self.layers):
            if incremental_state is None:
                self_attn_mask = self.buffered_future_mask(x)
                if attn_mask is not None:
                    self_attn_mask = self_attn_mask + (1 - attn_mask.float()) * -1e8
                self_attn_mask = self_attn_mask.clamp_min(-1e8)
            else:
                self_attn_mask = None

            x, attn_weights = layer(
                x,
                incremental_state=incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                spk_pos_ids_flat=spk_pos_ids_flat
            )

        if streaming and incremental_state != {}:
            for k, v in incremental_state.items():
                if 'attn_state' in k:
                    prev_key, prev_value = incremental_state[k]['prev_key'], incremental_state[k]['prev_value']
                    cur_length = prev_key.shape[2]
                    if cur_length - prompt_length > cache_size:
                        prev_key = torch.cat((prev_key[:, :, :prompt_length], prev_key[:, :, -cache_size:]), dim=2)
                        prev_value = torch.cat((prev_value[:, :, :prompt_length], prev_value[:, :, -cache_size:]),
                                               dim=2)
                    incremental_state[k]['prev_key'], incremental_state[k]['prev_value'] = prev_key, prev_value

        if not self.use_post_ln:
            x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        x = self.project_out_dim(x)
        return x

    def infer(self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
              spk_id=None, spk_embed=None, mels_timbre=None,
              incremental_state=None, ctx_vqcodes=None, spk_pos_ids_flat=None, return_state=False,
              first_step_min=0, return_probs=False, first_decoder_inp=None, dur_disturb=0.0, **kwargs):
        if incremental_state is None:
            incremental_state = {}
        x_ling = self.forward_ling_encoder(
            txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
            spk_id, spk_embed, mels_timbre)
        x_ling = x_ling.flatten(0, 1)
        txt_tokens_ori = txt_tokens
        txt_tokens_withpad = txt_tokens = txt_tokens.flatten(0, 1)
        x_ling = x_ling[txt_tokens > 0][None]
        txt_tokens = txt_tokens[txt_tokens > 0][None]

        decoded = torch.zeros_like(txt_tokens)
        decoded = F.pad(decoded, [1, 0], value=self.code_size + 1)
        if incremental_state != {}:
            if first_decoder_inp is None:
                assert ctx_vqcodes is not None
                decoded[:, :ctx_vqcodes.shape[1]] = ctx_vqcodes
                ctx_vqcodes = None
            else:
                decoded[:, :1] = first_decoder_inp
        probs = []
        for step in range(decoded.shape[1] - 1):
            vq_pred = self(txt_tokens, None, None, None, None,
                           decoded[:, :step + 1], None, None, None,
                           incremental_state=incremental_state, x_ling=x_ling,
                           spk_pos_ids_flat=spk_pos_ids_flat, **kwargs)
            probs.append(vq_pred.cpu())
            if ctx_vqcodes is None or step >= ctx_vqcodes.shape[1]:
                if self.hparams['dur_model_type'] == 'ar_mse':
                    d = vq_pred[:, -1, 0]
                    if dur_disturb > 0 and step >= 1:
                        if random.random() > 0.5:
                            d = d * (1 + random.random() * dur_disturb)
                        else:
                            d = d / (1 + random.random() * dur_disturb)
                        d = torch.clamp_max(d, self.code_size - 1)
                    vq_pred = torch.round(d).long()
                else:
                    vq_pred = self.sample_one_step(vq_pred)
                decoded[:, step + 1] = torch.clamp_min(vq_pred, 1)
                if step == 0:
                    decoded[:, step + 1] = torch.clamp_min(vq_pred, first_step_min)
            else:
                decoded[:, step + 1] = ctx_vqcodes[:, step]
        decoded = decoded[:, 1:]
        decoded_2d = torch.zeros_like(txt_tokens_ori)
        decoded_2d.flatten(0, 1)[txt_tokens_withpad > 0] = decoded
        if return_state:
            return decoded_2d, incremental_state
        if return_probs:
            return decoded_2d, torch.cat(probs, 1)
        return decoded_2d

    def streaming_infer(self, txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
                        spk_id=None, spk_embed=None, mels_timbre=None,
                        incremental_state=None, ctx_vqcodes=None, spk_pos_ids_flat=None, return_state=False,
                        **kwargs):
        if incremental_state is None:
            incremental_state = {}
        x_ling = self.forward_ling_encoder(
            txt_tokens, ling_feas, char_tokens, ph2char, bert_embed,
            spk_id, spk_embed, mels_timbre)
        x_ling = x_ling.flatten(0, 1)
        txt_tokens_ori = txt_tokens
        txt_tokens_withpad = txt_tokens = txt_tokens.flatten(0, 1)
        x_ling = x_ling[txt_tokens > 0][None]
        txt_tokens = txt_tokens[txt_tokens > 0][None]

        vq_decoded = torch.zeros_like(txt_tokens)
        vq_decoded = F.pad(vq_decoded, [1, 0], value=self.code_size + 1)
        if incremental_state != {}:
            assert ctx_vqcodes is not None
            vq_decoded[:, :ctx_vqcodes.shape[1]] = ctx_vqcodes
            ctx_vqcodes = None
        prompt_length = list(incremental_state.items())[0][1]['prev_key'].shape[2]
        for step in tqdm(range(vq_decoded.shape[1] - 1), desc='AR Duration Predictor inference...'):
            vq_pred = self(txt_tokens, None, None, None, None,
                           vq_decoded[:, :step + 1], None, None, None,
                           incremental_state=incremental_state, x_ling=x_ling,
                           spk_pos_ids_flat=spk_pos_ids_flat, prompt_length=prompt_length, streaming=True, **kwargs)
            if ctx_vqcodes is None or step >= ctx_vqcodes.shape[1]:
                if self.hparams['dur_model_type'] == 'ar_mse':
                    vq_pred = torch.round(vq_pred[:, -1, 0]).long()
                else:
                    vq_pred = self.sample_one_step(vq_pred)
                vq_decoded[:, step + 1] = vq_pred
            else:
                vq_decoded[:, step + 1] = ctx_vqcodes[:, step]
        vq_decoded = vq_decoded[:, 1:]
        vq_decoded_2d = torch.zeros_like(txt_tokens_ori)
        vq_decoded_2d.flatten(0, 1)[txt_tokens_withpad > 0] = vq_decoded
        if return_state:
            return vq_decoded_2d, incremental_state
        return vq_decoded_2d