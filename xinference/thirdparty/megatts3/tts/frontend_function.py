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

import torch
import torch.nn.functional as F
import whisper
import librosa
from copy import deepcopy
from tts.utils.text_utils.ph_tone_convert import split_ph_timestamp, split_ph
from tts.utils.audio_utils.align import mel2token_to_dur

''' Graphme to phoneme function '''
def g2p(self, text_inp):
    # prepare inputs
    txt_token = self.g2p_tokenizer('<BOT>' + text_inp + '<BOS>')['input_ids']
    input_ids = torch.LongTensor([txt_token+[145+self.speech_start_idx]]).to(self.device)

    # model forward
    with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
        outputs = self.g2p_model.generate(input_ids, max_new_tokens=256, do_sample=True, top_k=1, eos_token_id=800+1+self.speech_start_idx)
    
    # process outputs
    ph_tokens = outputs[:, len(txt_token):-1]-self.speech_start_idx
    ph_pred, tone_pred = split_ph(ph_tokens[0])
    ph_pred, tone_pred = ph_pred[None, :].to(self.device), tone_pred[None, :].to(self.device)
    return ph_pred, tone_pred

''' Get phoneme2mel align of prompt speech '''
def align(self, wav):
    with torch.inference_mode():
        whisper_wav = librosa.resample(wav, orig_sr=self.sr, target_sr=16000)
        mel = torch.FloatTensor(whisper.log_mel_spectrogram(whisper_wav).T).to(self.device)[None].transpose(1,2)
        prompt_max_frame = mel.size(2) // self.fm * self.fm
        mel = mel[:, :, :prompt_max_frame]
        token = torch.LongTensor([[798]]).to(self.device)
        audio_features = self.aligner_lm.embed_audio(mel)
        for i in range(768):
            with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
                logits = self.aligner_lm.logits(token, audio_features, None)
                token_pred = torch.argmax(F.softmax(logits[:, -1], dim=-1), 1)[None]
                token = torch.cat([token, token_pred], dim=1)
                if token_pred[0] == 799:
                    break
        alignment_tokens = token
    
    ph_ref, tone_ref, dur_ref, _ = split_ph_timestamp(deepcopy(alignment_tokens)[0, 1:-1])
    ph_ref = torch.Tensor(ph_ref)[None].to(self.device)
    tone_ref = torch.Tensor(tone_ref)[None].to(self.device)
    if dur_ref.sum() < prompt_max_frame:
        dur_ref[-1] += prompt_max_frame - dur_ref.sum()
    elif dur_ref.sum() > prompt_max_frame:
        len_diff = dur_ref.sum() - prompt_max_frame
        while True:
            for i in range(len(dur_ref)):
                dur_ref[i] -= 1
                len_diff -= 1
                if len_diff == 0:
                    break
            if len_diff == 0:
                break
    mel2ph_ref = self.length_regulator(dur_ref[None]).to(self.device)
    mel2ph_ref = mel2ph_ref[:, :mel2ph_ref.size(1)//self.fm*self.fm]
    return ph_ref, tone_ref, mel2ph_ref

''' Duration Prompting '''
def make_dur_prompt(self, mel2ph_ref, ph_ref, tone_ref):
    dur_tokens_2d_ = mel2token_to_dur(mel2ph_ref, ph_ref.shape[1]).clamp(
                    max=self.hp_dur_model['dur_code_size'] - 1) + 1

    ctx_dur_tokens = dur_tokens_2d_.clone().flatten(0, 1).to(self.device)
    txt_tokens_flat_ = ph_ref.flatten(0, 1)
    ctx_dur_tokens = ctx_dur_tokens[txt_tokens_flat_ > 0][None]

    last_dur_pos_prompt = ctx_dur_tokens.shape[1]
    dur_spk_pos_ids_flat = range(0, last_dur_pos_prompt)
    dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(self.device)
    with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
        _, incremental_state_dur_prompt = self.dur_model.infer(
            ph_ref, {'tone': tone_ref}, None, None, None,
            ctx_vqcodes=ctx_dur_tokens, spk_pos_ids_flat=dur_spk_pos_ids_flat, return_state=True)
    return incremental_state_dur_prompt, ctx_dur_tokens

''' Duration Prediction '''
def dur_pred(self, ctx_dur_tokens, incremental_state_dur_prompt, ph_pred, tone_pred, seg_i, dur_disturb, dur_alpha, is_first, is_final):
    last_dur_token = ctx_dur_tokens[:, -1:]
    last_dur_pos_prompt = ctx_dur_tokens.shape[1]
    incremental_state_dur = deepcopy(incremental_state_dur_prompt)
    txt_len = ph_pred.shape[1]
    dur_spk_pos_ids_flat = range(last_dur_pos_prompt, last_dur_pos_prompt + txt_len)
    dur_spk_pos_ids_flat = torch.LongTensor([dur_spk_pos_ids_flat]).to(self.device)
    last_dur_pos_prompt = last_dur_pos_prompt + txt_len

    with torch.cuda.amp.autocast(dtype=self.precision, enabled=True):
        dur_pred = self.dur_model.infer(
            ph_pred, {'tone': tone_pred}, None, None, None,
            incremental_state=incremental_state_dur,
            first_decoder_inp=last_dur_token,
            spk_pos_ids_flat=dur_spk_pos_ids_flat,
        )

    dur_pred = dur_pred - 1
    dur_pred = dur_pred.clamp(0, self.hp_dur_model['dur_code_size'] - 1)
    # if is_final:
    #     dur_pred[:, -1] = dur_pred[:, -1].clamp(64, 128)
    # else:
    #     dur_pred[:, -1] = dur_pred[:, -1].clamp(48, 128)
    # if seg_i > 0:
        # dur_pred[:, 0] = 0
    # ['。', '！', '？', 'sil']
    for sil_token in [148, 153, 166, 145]:
        dur_pred[ph_pred==sil_token].clamp_min(32)
    # ['，', '；'] 
    for sil_token in [163, 165]:
        dur_pred[ph_pred==sil_token].clamp_min(16)
    if not is_final:
        # add 0.32ms for crossfade
        dur_pred[:, -1] =  dur_pred[:, -1] + 32
    else:
        dur_pred[:, -1] = dur_pred[:, -1].clamp(64, 128)

    ''' DiT target speech generation '''
    dur_disturb_choice = (torch.rand_like(dur_pred.float()) > 0.5).float()
    dur_disturb_r = 1 + torch.rand_like(dur_pred.float()) * dur_disturb
    dur_pred = dur_pred * dur_disturb_r * dur_disturb_choice + \
            dur_pred / dur_disturb_r * (1 - dur_disturb_choice)
    dur_pred = torch.round(dur_pred * dur_alpha).clamp(0, 127)
    if is_first:
        dur_pred[:, 0] = 8
    
    dur_sum = dur_pred.sum()
    npad = self.fm - dur_sum % self.fm
    if npad < self.fm:
        dur_pred[:, -1] += npad
    mel2ph_pred = self.length_regulator(dur_pred).to(self.device)
    return mel2ph_pred

def prepare_inputs_for_dit(self, mel2ph_ref, mel2ph_pred, ph_ref, tone_ref, ph_pred, tone_pred, vae_latent):
    # Prepare duration token 
    mel2ph_pred = torch.cat((mel2ph_ref, mel2ph_pred+ph_ref.size(1)), dim=1)
    mel2ph_pred = mel2ph_pred[:, :mel2ph_pred.size(1)//self.fm*self.fm].repeat(3, 1)
    # Prepare phone and tone token
    ph_pred = torch.cat((ph_ref, ph_pred), dim=1)
    tone_pred = torch.cat((tone_ref, tone_pred), dim=1)
    # Disable the English tone (set them to 3)"""
    en_tone_idx = ~((tone_pred == 4) | ( (11 <= tone_pred) & (tone_pred <= 15)) | (tone_pred == 0))
    tone_pred[en_tone_idx] = 3
    
    # Prepare cfg inputs
    ph_seq = torch.cat([ph_pred, ph_pred, torch.full(ph_pred.size(), self.cfg_mask_token_phone, device=self.device)], 0)
    tone_seq = torch.cat([tone_pred, tone_pred, torch.full(tone_pred.size(), self.cfg_mask_token_tone, device=self.device)], 0)
    target_size = mel2ph_pred.size(1)//self.vae_stride
    vae_latent_ = vae_latent.repeat(3, 1, 1)
    ctx_mask = torch.ones_like(vae_latent_[:, :, 0:1])
    vae_latent_ = F.pad(vae_latent_, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)
    vae_latent_[1:] = 0.0
    ctx_mask = F.pad(ctx_mask, (0, 0, 0, target_size - vae_latent.size(1)), mode='constant', value=0)

    return {
        'phone': ph_seq,
        'tone': tone_seq,
        "lat_ctx": vae_latent_ * ctx_mask,
        "ctx_mask": ctx_mask,
        "dur": mel2ph_pred,
    }
