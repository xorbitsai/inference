import torch
import torch.nn as nn
from typing import Optional

import logging

logger = logging.getLogger(__name__)

class Transpose(nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

def patch_continuous_features(
    input_embeddings: torch.Tensor,
    placeholder_loc_lens: torch.Tensor,
    encoded_feats: torch.Tensor,
    encoded_feat_lens: torch.Tensor,
):
    """
    Patch continuous features into input embeddings, while keeping a valid gradient flow.

    input_embeddings: torch.Tensor, size = [B, C?, T, D]
    placeholder_loc_lens: torch.LongTensor, size = [B, N, 2]
        Each 2-tuple represents (start, length) of a placeholder.
    encoded_feats: torch.Tensor, size = [B, L1 + L2 + ... + LN, ...]
    encoded_feat_lens: torch.LongTensor, size = [B, N]

    Example ('X' for patch placeholder tokens):
    Inputs:
        input_embeddings = [[1, 2, 3, X, X, X, 4, 5, 6, X, X, X, 7, 8]]
        placeholder_loc_lens = [[[3, 3]], [[9, 3]]]
        encoded_feats = [[A, A, A, B, B]]
        encoded_feat_lens = [[3], [2]]
    Outputs:
        embeddings = [[1, 2, 3, A, A, A, 4, 5, 6, B, B, X, 7, 8]]
    """
    batch_size = input_embeddings.size(0)
    for i in range(batch_size):
        audio_feat_start = 0
        for j in range(placeholder_loc_lens.shape[1]):
            placeholder_start: int = int(placeholder_loc_lens[i, j, 0].item())
            placeholder_len: int = int(placeholder_loc_lens[i, j, 1].item())
            if placeholder_len <= 0:
                break
            feat_len = int(encoded_feat_lens[i, j].item())
            real_feat_len = feat_len
            if feat_len > placeholder_len:
                logger.warning(
                    f"Feature length ({feat_len}) > placeholder length ({placeholder_len}). "
                    "This is not expected; please check estimate_audio_feature_length(). "
                    "We truncate the feature to avoid errors."
                )
                feat_len = placeholder_len
            target_len = min(feat_len, placeholder_len)
            input_embeddings[i, placeholder_start:placeholder_start + target_len] = encoded_feats[i, audio_feat_start:audio_feat_start + target_len]
            audio_feat_start += real_feat_len
    return input_embeddings

def build_modality_mask(placeholder_loc_lens: torch.Tensor, shape: torch.Size):
    mask = torch.zeros(shape, dtype=torch.bool)
    for i in range(placeholder_loc_lens.shape[0]):
        for j in range(placeholder_loc_lens.shape[1]):
            start: int = int(placeholder_loc_lens[i, j, 0].item())
            length: int = int(placeholder_loc_lens[i, j, 1].item())
            if length <= 0:
                break
            mask[i, start:start + length] = True
    return mask

def encode_audio_segments(
    encoder,
    proj_layer,
    wav_feats=None,
    wav_feats_lengths=None,
    waveforms=None,
    waveforms_lengths=None,
    use_waveform=False,
    audio_config=None,
):
    """
    Apply audio encoder to input audio features in wrapped format.
    See the documentation of unwrap_feats() for details about 'wrapped format'.
    """

    # Forward audio encoder.
    if use_waveform:
        assert waveforms is not None and waveforms_lengths is not None
        # Unwrap the waveforms so each waveform is placed at an independent row.
        waveform_segs_batch, waveform_seg_lengths = unwrap_feats(waveforms, waveforms_lengths)
        audio_feats_seg, audio_feat_seg_lengths = encoder(waveform_segs_batch, waveform_seg_lengths)[:2]
    else:
        assert wav_feats is not None and wav_feats_lengths is not None
        # Unwrap the features so the feature of each waveform is placed at an independent row.
        feat_segs_batch, feat_seg_lengths = unwrap_feats(wav_feats, wav_feats_lengths)
        # for whisper encoder
        # feat_segs_batch: [B, T, n_mels]
        # feat_seg_lengths: [B]
        audio_feats_seg = encoder(feat_segs_batch)
        audio_feats_seg_proj = proj_layer(audio_feats_seg.transpose(-1, -2)).transpose(-1, -2)
        feat_seg_lengths = feat_seg_lengths.to(feat_segs_batch.device)
        # whisper encoder conv
        audio_feat_seg_lengths = (feat_seg_lengths - 3 + 2 * 1) // 2 + 1
        # project layer conv
        audio_feat_seg_lengths = (audio_feat_seg_lengths - audio_config.ds_kernel_size + 2 *
                                  (audio_config.ds_kernel_size//2)) // audio_config.ds_stride + 1

    # Wrap the features so the 1st dim represents batch_size.
    input_lengths = waveforms_lengths if use_waveform else wav_feats_lengths
    assert input_lengths is not None
    audio_feats, _, audio_feats_lengths = wrap_feats(audio_feats_seg, input_lengths, audio_feat_seg_lengths)
    audio_feats_proj, _, audio_feats_lengths2 = wrap_feats(audio_feats_seg_proj, input_lengths, audio_feat_seg_lengths)
    assert torch.all(audio_feats_lengths == audio_feats_lengths2), f"{audio_feats_lengths}, {audio_feats_lengths2}"

    return audio_feats_proj, audio_feats, audio_feats_lengths

def unwrap_feats(feats: torch.Tensor, feats_lengths: torch.Tensor):
    """
    The input feats are in the "wrapped" format, which means that features from (at most) N audios are concatenated
    as a single sample feats[i]. In this case, each row of feats_lengths contains the lengths of the concatenated
    feature. This function unwraps the features.
    For samples with less than N segments, one should pad feats_lengths with 0. The result will contain valid
    segments only.

    feats: torch.Tensor, size = [B, L1 + L2 + ... + LN, ...]
    feats_lengths: torch.LongTensor, size = [B, N]

    Example ('X' for padding):
    Inputs:
        feats = [[A, A, A, A, X],
                 [B, B, C, C, C]]
        feats_lengths = [[4, 0],
                         [2, 3]]
    Outputs:
        feat_segs = [[A, A, A, A],
                     [B, B, X, X],
                     [C, C, C, X]]
        feat_seg_lengths = [4, 2, 3]
    """
    feat_segs = []
    feat_seg_lengths = []
    for i in range(feats_lengths.shape[0]):
        feat_index = 0
        for j in range(feats_lengths.shape[1]):
            feat_len = feats_lengths[i, j].item()
            if feat_len == 0: break
            feat_segs.append(feats[i, feat_index:feat_index + feat_len])
            feat_seg_lengths.append(feat_len)
            feat_index += feat_len
    feat_segs_batch = torch.nn.utils.rnn.pad_sequence(feat_segs, True).to(feats.device)
    feat_seg_lengths = torch.tensor(feat_seg_lengths, dtype=torch.long, device=feats.device)
    return feat_segs_batch, feat_seg_lengths

def wrap_feats(feat_segs: torch.Tensor, feats_lengths: torch.Tensor, feats_seg_lengths: Optional[torch.Tensor] = None):
    """
    Wrap segmented features back to the wrapped format.
    This function is the inverse operation of unwrap_feats(). See its documentation for details.
    Note that the feats_lengths value does not matter a lot. We only check the location of the first 0 to determine the
    number of feature segments.
    """
    feat_idx = 0
    feats_buffer = []
    feats_locs_buffer = []
    feats_lengths_buffer = []
    for i in range(feats_lengths.shape[0]):
        feat_buffer = []
        feat_locs_buffer = []
        feat_lengths_buffer = []
        feat_total_len = 0
        for j in range(feats_lengths.shape[1]):
            feat_len = feats_lengths[i, j].item()
            if feat_len == 0:
                break
            if feats_seg_lengths is not None:
                feat_len = feats_seg_lengths[feat_idx].item()
            feat_buffer.append(feat_segs[feat_idx, :feat_len])
            feat_locs_buffer.append(feat_total_len)
            feat_lengths_buffer.append(feat_len)
            feat_idx += 1
            feat_total_len += feat_len
        feats_buffer.append(torch.cat(feat_buffer))
        feats_locs_buffer.append(torch.tensor(feat_locs_buffer, dtype=torch.long))
        feats_lengths_buffer.append(torch.tensor(feat_lengths_buffer, dtype=torch.long))
    feats = torch.nn.utils.rnn.pad_sequence(feats_buffer, True).to(feat_segs.device)
    feats_locs = torch.nn.utils.rnn.pad_sequence(feats_locs_buffer, True).to(feats_lengths.device)
    feats_new_lengths = torch.nn.utils.rnn.pad_sequence(feats_lengths_buffer, True).to(feats_lengths.device)
    return feats, feats_locs, feats_new_lengths
