# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu)

from typing import List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TransformerDecoder(nn.Module):
    def __init__(
            self, sos_id, eos_id, pad_id, odim,
            n_layers, n_head, d_model,
            residual_dropout=0.1, pe_maxlen=5000):
        super().__init__()
        self.INF = 1e10
        # parameters
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.n_layers = n_layers

        # Components
        self.tgt_word_emb = nn.Embedding(odim, d_model, padding_idx=self.pad_id)
        self.positional_encoding = PositionalEncoding(d_model, max_len=pe_maxlen)
        self.dropout = nn.Dropout(residual_dropout)

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            block = DecoderLayer(d_model, n_head, residual_dropout)
            self.layer_stack.append(block)

        self.tgt_word_prj = nn.Linear(d_model, odim, bias=False)
        self.layer_norm_out = nn.LayerNorm(d_model)

        self.tgt_word_prj.weight = self.tgt_word_emb.weight
        self.scale = (d_model ** 0.5)

    def batch_beam_search(self, encoder_outputs, src_masks,
                   beam_size=1, nbest=1, decode_max_len=0,
                   softmax_smoothing=1.0, length_penalty=0.0, eos_penalty=1.0,
                   elm=None, elm_weight=0.0):
        B = beam_size
        N, Ti, H = encoder_outputs.size()
        device = encoder_outputs.device
        maxlen = decode_max_len if decode_max_len > 0 else Ti
        assert eos_penalty > 0.0

        # Init
        encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, Ti, H)
        src_mask = src_masks.unsqueeze(1).repeat(1, B, 1, 1).view(N*B, -1, Ti)
        ys = torch.ones(N*B, 1).fill_(self.sos_id).long().to(device)
        t_ys = ys.clone()
        confidences = torch.zeros(N*B, 1).float().to(device)
        caches: List[Optional[Tensor]] = []
        for _ in range(self.n_layers):
            caches.append(None)
        scores = torch.tensor([0.0] + [-self.INF]*(B-1)).float().to(device)
        scores = scores.repeat(N).view(N*B, 1)
        is_finished = torch.zeros_like(scores)
        if elm is not None:
            elm_cache = elm.init_hidden(encoder_outputs, N*B)

        # Autoregressive Prediction
        for t in range(maxlen):
            tgt_mask = self.ignored_target_position_is_0(ys, self.pad_id)

            dec_output = self.dropout(
                self.tgt_word_emb(ys) * self.scale +
                self.positional_encoding(ys))
#            if t > 0:
#                dec_output = dec_output[:, -1:, :]
            i = 0
            for dec_layer in self.layer_stack:
                dec_output = dec_layer.forward(
                    dec_output, encoder_outputs,
                    tgt_mask, src_mask,
                    cache=caches[i])
                caches[i] = dec_output
                i += 1

            dec_output = self.layer_norm_out(dec_output)

            t_logit = self.tgt_word_prj(dec_output[:, -1])
            t_scores = F.log_softmax(t_logit / softmax_smoothing, dim=-1)
            t_origin_scores  = t_scores

            if elm is not None and elm_weight > 0.0:
                elm_logit, elm_cache = elm.forward_model(t_ys, hidden=elm_cache)
                #elm_logit, _ = elm.forward_model(ys)
                t_lm_scores = torch.log_softmax(elm_logit[:, -1], dim=-1) * (1 - is_finished.float()) # mask, (N*B, V)
                t_lm_scores[:, elm.eos_id] *= 3
                t_scores = t_scores + elm_weight * t_lm_scores

            if eos_penalty != 1.0:
                t_scores[:, self.eos_id] *= eos_penalty

            t_topB_scores, t_topB_ys = torch.topk(t_scores, k=B, dim=1)
            t_topB_scores = self.set_finished_beam_score_to_zero(t_topB_scores, is_finished)
            t_topB_ys = self.set_finished_beam_y_to_eos(t_topB_ys, is_finished)

            # Accumulated
            scores = scores + t_topB_scores

            # Pruning
            scores = scores.view(N, B*B)
            scores, topB_score_ids = torch.topk(scores, k=B, dim=1)
            scores = scores.view(-1, 1)

            topB_row_number_in_each_B_rows_of_ys = torch.div(topB_score_ids, B).view(N*B)
            stride = B * torch.arange(N).view(N, 1).repeat(1, B).view(N*B).to(device)
            topB_row_number_in_ys = topB_row_number_in_each_B_rows_of_ys.long() + stride.long()

            # Update ys
            ys = ys[topB_row_number_in_ys]
            t_ys = torch.gather(t_topB_ys.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
            ys = torch.cat((ys, t_ys), dim=1)

            # Update confidences
            confidences = confidences[topB_row_number_in_ys]
            t_confidences = torch.gather(t_topB_scores.view(N, B*B), dim=1, index=topB_score_ids).view(N*B, 1)
            t_confidences = torch.exp(t_confidences)
            assert torch.all(t_confidences <= 1.0)
            assert torch.all(t_confidences >= 0.0)
            confidences = torch.cat((confidences, t_confidences), dim=1)

            # Update caches
            new_caches: List[Optional[Tensor]] = []
            for cache in caches:
                if cache is not None:
                    new_caches.append(cache[topB_row_number_in_ys])
            caches = new_caches
            if elm and elm_weight > 0.0:
                elm_cache = (elm_cache[0][:, topB_row_number_in_ys], elm_cache[1][:, topB_row_number_in_ys])

            # Update finished state
            is_finished = t_ys.eq(self.eos_id)
            if is_finished.sum().item() == N*B:
                break

        # Length penalty (follow GNMT)
        scores = scores.view(N, B)
        ys = ys.view(N, B, -1)
        ys_lengths = self.get_ys_lengths(ys)
        if length_penalty > 0.0:
            penalty = torch.pow((5+ys_lengths.float())/(5.0+1), length_penalty)
            scores /= penalty
        nbest_scores, nbest_ids = torch.topk(scores, k=int(nbest), dim=1)
        nbest_scores = -1.0 * nbest_scores
        index = nbest_ids + B * torch.arange(N).view(N, 1).to(device).long()
        nbest_ys = ys.view(N*B, -1)[index.view(-1)]
        nbest_ys = nbest_ys.view(N, nbest_ids.size(1), -1)
        nbest_ys_lengths = ys_lengths.view(N*B)[index.view(-1)].view(N, -1)
        nbest_confidences = confidences.view(N*B, -1)[index.view(-1)].view(N, nbest_ids.size(1), -1)

        # result
        nbest_hyps: List[List[Dict[str, Tensor]]] = []
        for n in range(N):
            n_nbest_hyps: List[Dict[str, Tensor]] = []
            for i, score in enumerate(nbest_scores[n]):
                confidence = nbest_confidences[n, i, 1:nbest_ys_lengths[n, i]]
                confidence = confidence.mean()
                new_hyp = {
                    "yseq": nbest_ys[n, i, 1:nbest_ys_lengths[n, i]],
                    "confidence": confidence
                }
                n_nbest_hyps.append(new_hyp)
            nbest_hyps.append(n_nbest_hyps)
        return nbest_hyps

    def ignored_target_position_is_0(self, padded_targets, ignore_id):
        mask = torch.ne(padded_targets, ignore_id)
        mask = mask.unsqueeze(dim=1)
        T = padded_targets.size(-1)
        upper_tri_0_mask = self.upper_triangular_is_0(T).unsqueeze(0).to(mask.dtype)
        upper_tri_0_mask = upper_tri_0_mask.to(mask.dtype).to(mask.device)
        return mask.to(torch.uint8) & upper_tri_0_mask.to(torch.uint8)

    def upper_triangular_is_0(self, size):
        ones = torch.ones(size, size)
        tri_left_ones = torch.tril(ones)
        return tri_left_ones.to(torch.uint8)

    def set_finished_beam_score_to_zero(self, scores, is_finished):
        NB, B = scores.size()
        is_finished = is_finished.float()
        mask_score = torch.tensor([0.0] + [-self.INF]*(B-1)).float().to(scores.device)
        mask_score = mask_score.view(1, B).repeat(NB, 1)
        return scores * (1 - is_finished) + mask_score * is_finished

    def set_finished_beam_y_to_eos(self, ys, is_finished):
        is_finished = is_finished.long()
        return ys * (1 - is_finished) + self.eos_id * is_finished

    def get_ys_lengths(self, ys):
        N, B, Tmax = ys.size()
        ys_lengths = torch.sum(torch.ne(ys, self.eos_id), dim=-1)
        return ys_lengths.int()



class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout):
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = DecoderMultiHeadAttention(d_model, n_head, dropout)

        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = DecoderMultiHeadAttention(d_model, n_head, dropout)

        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = PositionwiseFeedForward(d_model, d_model*4, dropout)

    def forward(self, dec_input, enc_output, self_attn_mask, cross_attn_mask,
                cache=None):
        x = dec_input
        residual = x
        x = self.self_attn_norm(x)
        if cache is not None:
            xq = x[:, -1:, :]
            residual = residual[:, -1:, :]
            self_attn_mask = self_attn_mask[:, -1:, :]
        else:
            xq = x
        x = self.self_attn(xq, x, x, mask=self_attn_mask)
        x = residual + x

        residual = x
        x = self.cross_attn_norm(x)
        x = self.cross_attn(x, enc_output, enc_output, mask=cross_attn_mask)
        x = residual + x

        residual = x
        x = self.mlp_norm(x)
        x = residual + self.mlp(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        return x


class DecoderMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_qs = nn.Linear(d_model, n_head * self.d_k)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_k)

        self.attention = DecoderScaledDotProductAttention(
            temperature=self.d_k ** 0.5)
        self.fc = nn.Linear(n_head * self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        q = self.w_qs(q).view(bs, -1, self.n_head, self.d_k)
        k = self.w_ks(k).view(bs, -1, self.n_head, self.d_k)
        v = self.w_vs(v).view(bs, -1, self.n_head, self.d_k)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        output = self.attention(q, k, v, mask=mask)

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.fc(output)
        output = self.dropout(output)

        return output


class DecoderScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.INF = float("inf")

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        if mask is not None:
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -self.INF)
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)
        output = torch.matmul(attn, v)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.w_2(self.act(self.w_1(x)))
        output = self.dropout(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        length = x.size(1)
        return self.pe[:, :length].clone().detach()
