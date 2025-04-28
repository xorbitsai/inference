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

def map_phone_to_tokendict(item, pad_bos_eos=True):
    # Merge Chinese phone and tone (Original dict ends at 173, i.e., ph_dict_size=173). 146~173 is punctuations.
    phone = item['txt_token'].clone()
    merged_phone = item['txt_token'].clone()
    tone_tmp = item['tone'].clone()
    # In tone_dict, tone_1 is 4, tone_2 is 11, tone_3 is 12, tone_4 is 13, tone_5 is 14, tone_6 is 15
    tone_tmp[tone_tmp==4] = 1
    tone_tmp[tone_tmp==11] = 2
    tone_tmp[tone_tmp==12] = 3
    tone_tmp[tone_tmp==13] = 4
    tone_tmp[tone_tmp==14] = 5
    tone_tmp[tone_tmp==15] = 6
    # Chinese phones lie in 3~100 in the phone_dict, we map them to 200~788
    ch_phone_idx = (phone >= 3) & (phone <= 100)
    merged_phone[ch_phone_idx] = (merged_phone[ch_phone_idx] - 3) * 6 + 200 + tone_tmp[ch_phone_idx]

    if pad_bos_eos:
        merged_phone = F.pad(merged_phone, (1, 0), mode='constant', value=798)
        merged_phone = F.pad(merged_phone, (0, 1), mode='constant', value=799)
    return merged_phone
    
def split_ph_timestamp(ph_timestamp):
    ''' Input: ph_timestamp, shape [T] '''

    # Map the timestamp of each phone back to its original frame-level lengths
    ph_timestamp[ph_timestamp >= 800] -= 800

    ph_list = []
    tone_list = []
    dur_list = []
    cur_timestamp = 0
    for idx, item in enumerate(ph_timestamp):
        if idx % 2 == 0:
            # Map Chinese phones back to its original phone_dict
            if (200 <= item <= 788):
                ph = (item - 200 - 1) // 6 + 3
                tone = (item - 200 - 1) % 6 + 1
                if tone == 1:
                    tone = 4
                else:
                    tone = tone + 9
            # Set English tone to '3'
            else:
                ph = item
                tone = 3
            ph_list.append(ph)
            tone_list.append(tone)
        else:
            dur_list.append((item - cur_timestamp))
            cur_timestamp = item
    assert len(ph_list) == len(dur_list), f"{len(ph_list)}, {len(dur_list)}"
    ph_seq, tone_seq, dur_seq = torch.LongTensor(ph_list), torch.LongTensor(tone_list), torch.LongTensor(dur_list)
    return ph_seq, tone_seq, dur_seq, ph_timestamp[-1]
    
def split_ph(ph_seq):
    ''' Input: ph_timestamp, shape [T] '''
    ph_list = []
    tone_list = []
    for idx, item in enumerate(ph_seq):
        # Map Chinese phones back to its original phone_dict
        if (200 <= item <= 788):
            ph = (item - 200 - 1) // 6 + 3
            tone = (item - 200 - 1) % 6 + 1
            if tone == 1:
                tone = 4
            else:
                tone = tone + 9
        # Set English tone to '3'
        else:
            ph = item
            tone = 3
        ph_list.append(ph)
        tone_list.append(tone)
        
    assert len(ph_list) == len(tone_list)
    ph_seq, tone_seq = torch.LongTensor(ph_list), torch.LongTensor(tone_list)
    return ph_seq, tone_seq